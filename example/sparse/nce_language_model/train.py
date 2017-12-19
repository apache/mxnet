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
import os, math
import data_utils

parser = argparse.ArgumentParser(description='PennTreeBank LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/ptb_vocab.txt',
                    help='location of the corpus vocab')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
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
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--scale', type=int, default=1,
                    help='scaling factor for vocab size')
parser.add_argument('--k', type=int, default=15,
                    help='number of noise samples to estimate')
parser.add_argument('--use-gpu', type=int, default=0,
                    help='which gpu to use')
parser.add_argument('--use-dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')
parser.add_argument('--use-full-softmax', action='store_true',
                    help='use full softmax ce loss instead of noise contrastive estimation')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--lr-decay', type=float, default=0.25,
                    help='learning rate decay')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus to use')
parser.add_argument('--rescale-grad', type=float, default=1,
                    help='rescale grad')
parser.add_argument('--gpu', type=int, default=1,
                    help='which gpu')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='which optimizer to use')
parser.add_argument('--Z', type=int, default=1,
                    help='Z')
parser.add_argument('--profile', action='store_true',
                    help='whether to use profiler')
parser.add_argument('--kvstore', type=str, default=None,
                    help='type of kvstore to use')
parser.add_argument('--dummy-iter', action='store_true',
                    help='whether to dummy data iterator')
args = parser.parse_args()


best_val = 100000

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    ctx = [mx.gpu(args.gpu) for i in range(args.num_gpus)] if args.num_gpus > 0 else mx.cpu()
    full_softmax = args.use_full_softmax
    batch_size = args.batch_size if args.num_gpus == 0 else args.batch_size * args.num_gpus

    # data
    vocab = data_utils.Vocabulary.from_file(args.vocab)
    unigram = vocab.unigram()
    ntokens = unigram.size * args.scale
    sampler = AliasMethod(unigram)
    # TODO serialize sampler table
    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data + "train.txt", vocab,
                                       batch_size, args.bptt))
    eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data + "valid.txt", vocab,
                                      batch_size, args.bptt))
    test_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data + "test.txt", vocab,
                                      batch_size, args.bptt))

    if args.dummy_iter:
        train_data = DummyIter(train_data)
        eval_data = DummyIter(train_data)
        test_data = DummyIter(train_data)

    # model
    on_cpu = False
    #group2ctxs={'cpu_dev':[mx.cpu(0) for i in range(args.num_gpus)], 'gpu_dev':ctx} if on_cpu else None
    rnn_out, weight, last_states = rnn(args.bptt, ntokens, args.emsize, args.nhid,
                                       args.nlayers, args.dropout, args.use_dense, on_cpu, batch_size)
    logging.debug(str(last_states))
    if full_softmax:
        model = ce_loss(rnn_out, ntokens, args.tied, args.use_dense, weight)
    else:
        model = nce_loss(rnn_out, ntokens, args.nhid, args.k, on_cpu, batch_size, args.bptt,
                         args.use_dense, decoder_w=weight if args.tied else None)
    state_names = ['lstm_l0_0', 'lstm_l0_1', 'lstm_l1_0', 'lstm_l1_1'] if args.nlayers == 2 else ['lstm_l0_0', 'lstm_l0_1']
    # module
    last_states.append(model)
    extra_states = ['sample', 'p_noise_sample', 'p_noise_target']
    module = mx.mod.Module(symbol=mx.sym.Group(last_states), context=ctx,
                           state_names=(state_names + extra_states) if not full_softmax else state_names,
                           data_names=['data', 'mask'], label_names=['label'])#, group2ctxs=group2ctxs)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier())

    kvstore = None #if args.kvstore is None else mx.kv.create(args.kvstore)
    #optimizer = mx.optimizer.create('sgd', learning_rate=args.lr, rescale_grad=1.0/batch_size)
    if args.optimizer == 'sgd':
        optimizer = mx.optimizer.create('sgd', learning_rate=args.lr,
                                    rescale_grad=args.rescale_grad, wd=args.wd, momentum=args.mom)
    elif args.optimizer == 'adam':
        optimizer = mx.optimizer.create('adam', learning_rate=args.lr, rescale_grad=args.rescale_grad, beta1=args.beta1)
    else:
         raise NotImplementedError()

    module.init_optimizer(optimizer=optimizer, kvstore=kvstore)
    speedometer = mx.callback.Speedometer(batch_size, args.log_interval)

    ############### eval model ####################
    eval_rnn_out, eval_weight, eval_last_states = rnn(args.bptt, ntokens, args.emsize, args.nhid,
                                                      args.nlayers, 0, args.use_dense, on_cpu, batch_size)
    eval_model = ce_loss(eval_rnn_out, ntokens, args.tied, args.use_dense, eval_weight)
    eval_last_states.append(eval_model)
    ############### eval module ####################
    eval_module = mx.mod.Module(symbol=mx.sym.Group(eval_last_states), context=ctx, data_names=['data', 'mask'],
                                label_names=['label'], state_names=state_names)
    eval_module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, shared_module=module, for_training=False)
    ############### eval module ####################

    # get the sparse weight parameter
    if kvstore:
        assert(False)
        encoder_w_index = module._exec_group.param_names.index('encoder_weight')
        encoder_w_param = module._exec_group.param_arrays[encoder_w_index]
        if not full_softmax:
            decoder_w_index = module._exec_group.param_names.index('decoder_weight')
            decoder_w_param = module._exec_group.param_arrays[decoder_w_index]

    if args.profile:
        config = ['scale', args.scale, 'nhid', args.nhid, 'k', args.k, 'nlayers', args.nlayers,
                  'use_dense', args.use_dense, 'use_full_softmax', args.use_full_softmax, 'ngpu', args.num_gpus]
        config_str = map(lambda x: str(x), config)
        filename = '-'.join(config_str) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=filename)
        mx.profiler.profiler_set_state('run')

    def evaluate(eval_module, data_iter, epoch, mode):
        total_L = 0.0
        nbatch = 0
        eval_module.set_states(value=0)
        for batch in data_iter:
            eval_module.forward(batch, is_train=False)
            outputs = eval_module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            total_L += mx.nd.sum(outputs[-1][0]).asscalar()
            eval_module.set_states(states=state_cache)
            nbatch += 1
        data_iter.reset()
        loss = total_L / args.bptt / batch_size / nbatch
        logging.info('Iter[%d] %s\t\tloss %.7f, ppl %.7f'%(epoch, mode, loss, math.exp(loss)))
        return loss

    # train
    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        total_L = 0.0
        nbatch = 0
        module.set_states(value=0)
        state_cache = module.get_states(merge_multi_context=False)[:-len(extra_states)]
        for batch in train_data:
            if not full_softmax:
                label = batch.label[0]
                samples = sampler.draw(args.bptt * batch_size * args.k).reshape((args.bptt * batch_size, args.k))
                p_noise_sample = unigram[samples].reshape((args.bptt * batch_size, args.k))
                p_noise_target = unigram[label].reshape((args.bptt * batch_size, 1))
                state_cache += [[samples.astype(np.float32)], [p_noise_sample], [p_noise_target]]
                module.set_states(states=state_cache)
            '''
            if kvstore:
                # TODO use kvstore
                assert(False)
                data = batch.data[0].reshape((-1, ))
                kvstore.row_sparse_pull('encoder_weight', encoder_w_param, row_ids=[data for i in range(args.num_gpus)],
                                        priority=-encoder_w_index)
                if not full_softmax:
                    label = batch.label[0].reshape((-1, )).astype(np.int32)
                    sample = batch.data[1].reshape((-1, ))
                    row_ids = mx.nd.concat(data, label, sample, dim=0)
                    kvstore.row_sparse_pull('decoder_weight', decoder_w_param, row_ids=[row_ids for i in range(args.num_gpus)],
                                            priority=-decoder_w_index)
            '''
            module.forward(batch)
            outputs = module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            module.backward()
            total_L += mx.nd.sum(outputs[-1][0]).asscalar()
            # update all parameters (including the weight parameter)
            module.clip_by_global_norm(max_norm=args.clip)
            module.update()
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            # update training metric
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_L = total_L / args.log_interval
                logging.info('Iter[%d] Batch [%d] \tloss %.7f, ppl %.7f'%(
                    epoch, nbatch, cur_L, math.exp(cur_L)))
                total_L = 0.0
            nbatch += 1
        if kvstore:
            assert(False)
        val_L = evaluate(eval_module, eval_data, epoch, 'Valid')
        if val_L < best_val:
            best_val = val_L
            test_L = evaluate(eval_module, test_data, epoch, 'Test')
        else:
            optimizer.lr *= args.lr_decay
            logging.info("epoch %d with lr decay, lr = %.4f" % (epoch, optimizer.lr))
        eval_data.reset()
        train_data.reset()
    logging.info("Training completed. ")
    if args.profile:
        mx.profiler.profiler_set_state('stop')
