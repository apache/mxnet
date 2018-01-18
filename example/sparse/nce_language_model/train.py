# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import mxnet as mx
import run_utils
import evaluate
from data import Corpus, CorpusIter, DummyIter, MultiSentenceIter
from log_uniform import LogUniformSampler
from model import *
from sampler import *
from sparse_module import SparseModule
import os, math, logging, time, pickle
import data_utils

DEBUG_FLG = False

def DEBUG(s):
    if DEBUG_FLG:
        print(s)

if __name__ == '__main__':
    parser = run_utils.get_parser(is_train=True)
    args = parser.parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info(args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    ngpus = len(ctx)
    #assert(args.dense)

    # data
    vocab = data_utils.Vocabulary.from_file(args.vocab)
    unigram = vocab.unigram()
    ntokens = unigram.size
    os.environ["MXNET_MAGIC_DIM"] = str(ntokens) if not args.dense else "-2"
    #sampler = AliasMethod(unigram)
    sampler = MXLogUniformSampler(ntokens)
    # serialize sampler table
    # pickle.dump(sampler, open(args.checkpoint_dir + "sampler", "w"))

    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                       args.batch_size * ngpus, args.bptt))
    # model
    rnn_module = RNNModel(args.bptt, ntokens, args.emsize, args.nhid, args.nlayers,
                          args.dropout, args.num_proj)
    nce_module = SampledModule(ntokens, args.nhid, args.k, args.bptt, args.num_proj, is_nce=False)

    rnn_out, last_states = rnn_module.forward(args.batch_size)
    #p_target, p_sample = nce_module.forward(rnn_out, args.batch_size)
    #model = nce_criterion(p_target, p_sample, args.batch_size * args.bptt, args.k)
    logits, new_targets = nce_module.forward(rnn_out, args.batch_size)
    model = CrossEntropyLoss().forward(logits, new_targets)
    
    state_names = rnn_module.state_names

    sparse_params=['encoder_weight', 'decoder_weight', 'decoder_bias']
    data_names = ['data', 'mask']
    label_names = ['label']

    # module
    last_states.append(model)
    extra_states = ['sample', 'p_noise_sample', 'p_noise_target', 'hit_mask']

    import numpy as np
    param_map = {'decoder_bias': 'decoder.params.bias','lstm_l0_pj_bias': 'proj.bias','decoder_weight': 'decoder.params.weight','lstm_l0_h2h_bias': 'rnn.bias_hh_l0','encoder_weight': 'encoder.weight','lstm_l0_pj_weight': 'proj.weight','lstm_l0_i2h_weight': 'rnn.weight_ih_l0','lstm_l0_i2h_bias': 'rnn.bias_ih_l0','lstm_l0_h2h_weight' : 'rnn.weight_hh_l0'}

    # TODO load optimizer state
    if args.load_epoch < 0:
        module = SparseModule(symbol=mx.sym.Group(last_states), context=ctx,
                              state_names=(state_names + extra_states),
                              data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
        # currently params are initialized explicitly, choice of init has no impact
        arg_params = {}
        if DEBUG_FLG:
            count = 0
            for k in module._exec_group.execs[0]._arg_dict.keys():
                if k in param_map:
                    count += 1
                    torch_k = param_map[k]
                    torch_v = np.load(torch_k + '.npy')
                    if 'decoder_bias' == k:
                        torch_v = torch_v.reshape((-1, 1))
                    double_sum = np.array(torch_v, dtype=np.float64).sum()
                    arg_params[k] = mx.nd.array(torch_v, dtype=np.float64).astype(np.float32)
                    print('loading sum(%s) = %.7f -> %.7f | %.7f' % (k, torch_v.sum(), arg_params[k].sum().asnumpy()[0], double_sum))
            assert(count == len(param_map))
        module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, allow_missing=True)

    else:
        module = SparseModule.load(args.checkpoint_dir, 0, context=ctx, state_names=(state_names + extra_states),
                                   data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)

    # parameters
    all_args = model.list_arguments()
    trainable_args = set(all_args) - set(state_names) - set(extra_states) - set(data_names) - set(label_names)
    lstm_args = []
    for arg in trainable_args:
        if 'lstm' in arg and 'pj' not in arg:
            lstm_args.append(arg)
    print(lstm_args)

    kvstore = None if args.kvstore is None else mx.kv.create(args.kvstore)
    require_rsp_pull = kvstore and not args.dense
    # TODO support custom eps
    optimizer = mx.optimizer.create('adagrad', learning_rate=args.lr, rescale_grad=1.0/ngpus, eps=args.eps, wd=args.wd)

    module.init_optimizer(optimizer=optimizer, kvstore=kvstore)
    speedometer = mx.callback.Speedometer(args.batch_size * ngpus * args.bptt, args.log_interval)
    ############### eval module ####################

    if args.profile:
        config = ['nhid', args.nhid, 'k', args.k, 'nlayers', args.nlayers,
                  'dense', args.dense, 'ngpus', ngpus]
        config_str = map(lambda x: str(x), config)
        filename = '-'.join(config_str) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=filename)
        mx.profiler.profiler_set_state('run')

    # train
    def listify(x):
        return x if isinstance(x, list) else [x]

    def prep_samples(label):
        label_list = listify(label.split(ngpus, axis=0))
        sample = sampler.sample(long(ngpus * args.k))
        sample_list = listify(sample.split(ngpus, axis=0))
        p_noise_sample = sampler.probability(sample).reshape((ngpus * args.k,))
        p_noise_target = sampler.probability(label).reshape((args.bptt * args.batch_size * ngpus, 1))
        # remove accidental hits
        accidental_hit_mask_list = []
        for i in range(ngpus):
            accidental_hit_mask_list.append(mx.nd.contrib.accidental_hits(label_list[i].reshape((-1,)), sample_list[i]))

        p_noise_sample_list = listify(p_noise_sample.split(ngpus, axis=0))
        p_noise_target_list = listify(p_noise_target.split(ngpus, axis=0))
        return (sample_list, p_noise_sample_list, p_noise_target_list, accidental_hit_mask_list), sample

    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        total_L = mx.nd.array([0.0])
        nbatch = 0
        module.set_states(value=0)
        state_cache = module.get_states(merge_multi_context=False)[:-len(extra_states)]
        next_batch = train_data.next()
        next_lists, next_sample = prep_samples(next_batch.label[0])
        stop_iter = False
        while not stop_iter:
            batch = next_batch
            label = batch.label[0]
            lists, sample = next_lists, next_sample
            state_cache += lists
            module.set_states(states=state_cache)
            if require_rsp_pull:
                data_1d = batch.data[0].reshape((-1,)).astype(np.float32)
                label_1d = label.reshape((-1,))
                sample_1d = sample.reshape((-1,)).astype(np.float32)
                row_ids = mx.nd.concat(label_1d, sample_1d, dim=0)
                param_rowids = {'encoder_weight': data_1d, 'decoder_weight': row_ids, 'decoder_bias': row_ids}
                # sync_sparse_params should be part of forward API
                module.sync_sparse_params(param_rowids)

            module.forward(batch)
            try:
                next_batch = train_data.next()
                next_lists, next_sample = prep_samples(next_batch.label[0])
            except StopIteration:
                stop_iter = True
            outputs = module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            module.backward()
            # TODO haibin add_n
            for g in range(ngpus):
                total_L += outputs[-1][g].copyto(mx.cpu()) / ngpus

            # update all parameters (including the weight parameter)
            norm = module.clip_by_global_norm(max_norm=args.clip, param_names=['encoder_weight'])
            norm = module.clip_by_global_norm(max_norm=args.clip, param_names=lstm_args if args.clip_lstm else None)
            norm = module.clip_by_global_norm(max_norm=args.clip, param_names=['lstm_l0_pj_weight', 'lstm_l0_pj_bias'])
            norm = module.clip_by_global_norm(max_norm=args.clip, param_names=['decoder_weight', 'decoder_bias'])
            #print('norm is ', norm)
            module.update()
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=None, locals=locals())

            speedometer(speedometer_param)
            # update training metric
            # TODO (revert >=)
            x = -1 if DEBUG_FLG else 0
            if nbatch % args.log_interval == 0 and nbatch > x:
                cur_L = total_L.asscalar() / args.log_interval
                try:
                    ppl = math.exp(cur_L) if cur_L < 100 else -1.0
                except OverflowError:
                    ppl = -1.0
                logging.info('Iter[%d] Batch [%d] \tloss %.7f, ppl %.7f'%(
                    epoch, nbatch, cur_L, ppl))
                #print('Batch [%d] \tloss %.7f, ppl %.7f \n'%(nbatch, cur_L, ppl))
                total_L[:] = 0.0
            nbatch += 1
            if nbatch == args.checkpoint_interval:
                #exit()
                pass
        if (epoch + 1) % args.checkpoint_interval == 0:
            module.save_checkpoint(args.checkpoint_dir, epoch % 1, save_optimizer_states=True)
            nce_mod = SparseModule.load(args.checkpoint_dir, 0, context=mx.cpu(), state_names=(state_names + extra_states),
                                        data_names=data_names, label_names=label_names, sparse_params=sparse_params)
            checkpoint_iter = MultiSentenceIter(args.data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                                args.batch_size, args.bptt)
            nce_mod.bind(data_shapes=checkpoint_iter.provide_data, label_shapes=checkpoint_iter.provide_label)

            ############### eval model ####################
            eval_rnn_out, eval_last_states = rnn_module.forward(32)
            eval_model = ce_loss(eval_rnn_out, ntokens, args.dense)
            eval_last_states.append(eval_model)
            ############### eval module ####################
            eval_module = SparseModule(symbol=mx.sym.Group(eval_last_states), context=mx.cpu(), data_names=data_names,
                                       label_names=label_names, state_names=state_names, sparse_params=sparse_params)
            test_data_path = "/home/ubuntu/gbw-validation/heldout-monolingual.tokenized.shuffled/*"
            eval_data = mx.io.PrefetchingIter(MultiSentenceIter(test_data_path if not args.bench else "./data/ptb.tiny.txt", vocab,
                                              32, args.bptt))
            eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label, shared_module=nce_mod, for_training=False)
            val_L = evaluate.evaluate(eval_module, eval_data, epoch, args.log_interval, early_stop=None)
        train_data.reset()
    logging.info("Training completed. ")
    if args.profile:
        mx.profiler.profiler_set_state('stop')
