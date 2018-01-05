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
import argparse
from data import Corpus, CorpusIter, DummyIter, MultiSentenceIter
from model import *
from sampler import *
from sparse_module import SparseModule
import os, math, logging, time, pickle
import data_utils

parser = argparse.ArgumentParser(description='PennTreeBank LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--train-data', type=str, default='./data/ptb.train.txt',
                    help='location of the data corpus')
parser.add_argument('--eval-data', type=str, default='./data/ptb.valid.txt',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/ptb_vocab.txt',
                    help='location of the corpus vocab')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--num_proj', type=int, default=0,
                    help='number of projection units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--mom', type=float, default=0.0,
                    help='mom')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1')
parser.add_argument('--wd', type=float, default=0.0,
                    help='wd')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=6000,
                    help='upper epoch limit')
parser.add_argument('--eval-every', type=int, default=1,
                    help='evalutaion every x epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=15,
                    help='number of noise samples to estimate')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--lr-decay', type=float, default=0.25,
                    help='learning rate decay')
parser.add_argument('--minlr', type=float, default=0.00001,
                    help='min learning rate')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--load-epoch', type=int, default=-1,
                    help='load epoch')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='which optimizer to use')
parser.add_argument('--profile', action='store_true',
                    help='whether to use profiler')
parser.add_argument('--kvstore', type=str, default='device',
                    help='type of kv-store to use')
parser.add_argument('--init', type=str, default='uniform',
                    help='type of initialization for embed and softmax weight')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint/',
                    help='dir for checkpoint')
parser.add_argument('--bench', action='store_true',
                    help='whether to use tiny data')
parser.add_argument('--clip-lstm', action='store_true',
                    help='only clip lstm layers')
parser.add_argument('--tf-nce', action='store_true',
                    help='use tf nce impl')
parser.add_argument('--skip-eval', action='store_true',
                    help='skip evaluation')
args = parser.parse_args()


best_val = 100000

def evaluate(mod, data_iter, epoch, mode, args):
    import time
    start = time.time()
    total_L = 0.0
    nbatch = 0
    mod.set_states(value=0)
    for batch in data_iter:
        mod.forward(batch, is_train=False)
        outputs = mod.get_outputs(merge_multi_context=False)
        state_cache = outputs[:-1]
        # (args.batch_size * args.bptt)
        for g in range(ngpus):
            total_L += mx.nd.sum(outputs[-1][g]).asscalar()
        mod.set_states(states=state_cache)
        nbatch += 1
        logging.info("eval batch %d : %.7f" % (nbatch, total_L / args.bptt / nbatch))
        # TODO support eval batch size > 1
    data_iter.reset()
    loss = total_L / args.bptt / nbatch
    try:
        ppl = math.exp(loss)
    except Exception:
        ppl = -1
    end = time.time()
    logging.info('Iter[%d] %s\t\tloss %.7f, ppl %.7f. Cost = %.2f'%(epoch, mode, loss, ppl, end - start))
    return loss

if __name__ == '__main__':
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else mx.cpu()
    ngpus = len(ctx)
    if args.init == 'uniform':
        init = mx.init.Uniform(0.1)
    elif args.init == 'uniform_unit':
        init = mx.init.UniformUnitScaling()
    else:
        raise NotImplementedError()
    #assert(args.dense)

    # data
    vocab = data_utils.Vocabulary.from_file(args.vocab)
    unigram = vocab.unigram()
    ntokens = unigram.size
    os.environ["MXNET_MAGIC_DIM"] = str(ntokens) if not args.dense else "-2"
    sampler = AliasMethod(unigram)
    # serialize sampler table
    # pickle.dump(sampler, open(args.checkpoint_dir + "sampler", "w"))

    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.train_data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                       args.batch_size * ngpus, args.bptt))
    eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.eval_data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                      1, args.bptt))
    # model
    rnn_out, last_states = rnn(args.bptt, ntokens, args.emsize, args.nhid,
                               args.nlayers, args.dropout, args.dense, args.batch_size, init, args.num_proj)
    NCE = nce_loss_tf if args.tf_nce else nce_loss
    model = NCE(rnn_out, ntokens, args.nhid, args.k, args.batch_size, args.bptt, args.dense, init, args.num_proj)


    state_names = ['lstm_l0_0', 'lstm_l0_1', 'lstm_l1_0', 'lstm_l1_1'] if args.nlayers == 2 else ['lstm_l0_0', 'lstm_l0_1']
    sparse_params=['encoder_weight', 'decoder_weight', 'decoder_bias']
    data_names = ['data', 'mask']
    label_names = ['label']

    # module
    last_states.append(model)
    extra_states = ['sample', 'p_noise_sample', 'p_noise_target']
    # TODO load optimizer state
    if args.load_epoch < 0:
        module = SparseModule(symbol=mx.sym.Group(last_states), context=ctx,
                              state_names=(state_names + extra_states),
                              data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
        module.init_params(initializer=mx.init.Xavier())
    else:
        module = SparseModule.load(args.checkpoint_dir, 0, context=ctx, state_names=(state_names + extra_states),
                                   data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)

    # parameters
    all_args = model.list_arguments()
    trainable_args = set(all_args) - set(state_names) - set(extra_states) - set(data_names) - set(label_names)
    lstm_args = []
    for arg in trainable_args:
        if 'lstm' in arg:
            lstm_args.append(arg)
    print(lstm_args)

    kvstore = None if args.kvstore is None else mx.kv.create(args.kvstore)
    require_rsp_pull = kvstore and not args.dense
    if args.optimizer == 'sgd':
        optimizer = mx.optimizer.create('sgd', learning_rate=args.lr,
                                        rescale_grad=1.0/ngpus, wd=args.wd, momentum=args.mom)
    elif args.optimizer == 'adam':
        optimizer = mx.optimizer.create('adam', learning_rate=args.lr, rescale_grad=1.0/ngpus, beta1=args.beta1)
    elif args.optimizer == 'adagrad':
        optimizer = mx.optimizer.create('adagrad', learning_rate=args.lr, rescale_grad=1.0/ngpus)
    else:
         raise NotImplementedError()

    module.init_optimizer(optimizer=optimizer, kvstore=kvstore)
    speedometer = mx.callback.Speedometer(args.batch_size * ngpus, args.log_interval)
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

    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        total_L = 0.0
        nbatch = 0
        module.set_states(value=0)
        state_cache = module.get_states(merge_multi_context=False)[:-len(extra_states)]
        for batch in train_data:
            label = batch.label[0]
            sample = sampler.draw(args.k).reshape((args.k, )).copyto(mx.cpu())
            p_noise_sample = unigram[sample].reshape((1, args.k))
            p_noise_target = unigram[label].reshape((args.bptt * args.batch_size * ngpus, 1))

            sample_list = [sample] * ngpus
            p_noise_sample_list = [p_noise_sample] * ngpus
            p_noise_target_list = listify(p_noise_target.split(ngpus, axis=0))

            state_cache += [sample_list, p_noise_sample_list, p_noise_target_list]
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
            outputs = module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            module.backward()
            # TODO haibin add_n
            for g in range(ngpus):
                total_L += mx.nd.sum(outputs[-1][g]).asscalar() / ngpus

            # update all parameters (including the weight parameter)
            norm = module.clip_by_global_norm(max_norm=args.clip, param_names=lstm_args if args.clip_lstm else None)
            module.update()
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            # update training metric
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_L = total_L / args.log_interval
                try:
                    ppl = math.exp(cur_L)
                except OverflowError:
                    ppl = -1.0
                logging.info('Iter[%d] Batch [%d] \tloss %.7f, ppl %.7f'%(
                    epoch, nbatch, cur_L, ppl))
                total_L = 0.0
            nbatch += 1
        if (epoch + 1) % args.eval_every == 0:
            module.save_checkpoint(args.checkpoint_dir, 0, save_optimizer_states=True)
            if not args.skip_eval:
                nce_mod = SparseModule.load(args.checkpoint_dir, 0, context=mx.cpu(), state_names=(state_names + extra_states),
                                            data_names=data_names, label_names=label_names, sparse_params=sparse_params)
                nce_mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)

                ############### eval model ####################
                eval_rnn_out, eval_last_states = rnn(args.bptt, ntokens, args.emsize, args.nhid,
                                                     args.nlayers, 0, args.dense, 1, init, args.num_proj)
                eval_model = ce_loss(eval_rnn_out, ntokens, args.dense)
                eval_last_states.append(eval_model)
                ############### eval module ####################
                eval_module = SparseModule(symbol=mx.sym.Group(eval_last_states), context=mx.cpu(), data_names=['data', 'mask'],
                                           label_names=['label'], state_names=state_names, sparse_params=sparse_params)
                eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label, shared_module=nce_mod, for_training=False)
                val_L = evaluate(eval_module, eval_data, epoch, 'Valid', args)
                if val_L < best_val:
                    best_val = val_L
                else:
                    optimizer.lr *= args.lr_decay
                    optimizer.lr = max(args.minlr, optimizer.lr)
                    logging.info("epoch %d with lr decay, lr = %.6f" % (epoch, optimizer.lr))
                eval_data.reset()
        train_data.reset()
    logging.info("Training completed. ")
    if args.profile:
        mx.profiler.profiler_set_state('stop')
