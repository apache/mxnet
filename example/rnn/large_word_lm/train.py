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
import mxnet.symbol as S
import run_utils
from data import MultiSentenceIter, Vocabulary
from model import *
from custom_module import CustomModule
import os, math, logging, sys
from sampler import LogUniformSampler

if __name__ == '__main__':
    # parser
    parser = run_utils.get_parser()
    args = parser.parse_args()
    head = '%(asctime)-15s %(message)s'
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.gpu()]
    ngpus = len(ctx)
    rescale_loss = args.bptt

    # logging
    logging.basicConfig(level=logging.INFO, format=head)
    logging.info(args)
    logging.debug(sys.argv)

    # seeding
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # data
    vocab = Vocabulary.from_file(args.vocab)
    ntokens = vocab.num_tokens
    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data, vocab,
                                       args.batch_size * ngpus, args.bptt))
    # model
    model = Model(ntokens, rescale_loss, args.bptt, args.emsize, args.nhid,
                  args.nlayers, args.dropout, args.num_proj, args.batch_size, args.k)
    train_loss_and_states = model.train()
    eval_loss_and_states = model.eval()
    sampler = LogUniformSampler(ntokens, args.k)

    # training module
    data_names, label_names = ['data', 'mask'], ['label']
    eval_state_names = model.state_names
    num_sample_names = len(model.sample_names)
    train_state_names = model.state_names + model.sample_names

    module = CustomModule(symbol=train_loss_and_states, context=ctx,
                          state_names=train_state_names,
                          data_names=data_names, label_names=label_names)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier(factor_type='out'))

    # create kvstore and sparse optimizer
    kvstore = mx.kv.create('device')
    optimizer = mx.optimizer.create('adagrad', learning_rate=args.lr, \
                                    rescale_grad=1.0/ngpus, eps=args.eps)
    module.init_optimizer(optimizer=optimizer, kvstore=kvstore)

    # speedometer
    num_words_per_batch = args.batch_size * ngpus * args.bptt
    speedometer = mx.callback.Speedometer(num_words_per_batch, args.log_interval)

    # training loop
    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        total_L = mx.nd.array([0.0])
        nbatch = 0
        # reset initial LSTMP states
        module.set_states(value=0)
        state_cache = module.get_states(merge_multi_context=False)[:-num_sample_names]
        next_batch = train_data.next()
        next_sampled_values = generate_samples(next_batch.label[0], ngpus, sampler)
        stop_iter = False
        while not stop_iter:
            batch = next_batch
            state_cache += next_sampled_values
            # propagate LSTMP states from the previous batch
            module.set_states(states=state_cache)
            # selectively pull row_sparse weight to multiple devices based on the data batch
            target_ids = [batch.label[0]]
            sampled_ids = next_sampled_values[0]
            param_rowids = {'encoder_weight': batch.data[0],
                            'decoder_weight': sampled_ids + target_ids,
                            'decoder_bias': sampled_ids + target_ids}
            module.prepare_sparse_params(param_rowids)
            # forward
            module.forward(batch)
            try:
                # prefetch the next batch of data and samples
                next_batch = train_data.next()
                next_sampled_values = generate_samples(next_batch.label[0], ngpus, sampler)
            except StopIteration:
                stop_iter = True
            # cache LSTMP states of the current batch
            outputs = module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            # backward
            module.backward()
            for g in range(ngpus):
                total_L += outputs[-1][g].copyto(mx.cpu()) / ngpus

            # rescaling the gradient for embedding layer emperically leads to faster convergence
            module.rescale_grad(args.rescale_embed, 'encoder_weight')
            # clip lstm params on each device based on norm
            norm = module.clip_by_global_norm_per_ctx(max_norm=args.clip, param_names=model.lstm_args)
            # update parameters
            module.update()
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            # update training metric
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_L = total_L.asscalar() / args.log_interval / rescale_loss
                ppl = math.exp(cur_L) if cur_L < 100 else 1e36
                logging.info('Iter[%d] Batch [%d] \tloss %.7f, ppl %.7f'%(epoch, nbatch, cur_L, ppl))
                total_L[:] = 0.0
            nbatch += 1

        # run evaluation with full softmax on cpu
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        ckp = os.path.join(args.checkpoint_dir, 'ckp')
        module.save_checkpoint(ckp, epoch, save_optimizer_states=False)

        # use batch_size = 1 for testing
        eval_batch_size = 1
        load_model = Model(ntokens, rescale_loss, args.bptt, args.emsize, args.nhid,
                           args.nlayers, args.dropout, args.num_proj, eval_batch_size, args.k)
        cpu_train_mod = CustomModule.load(ckp, epoch, context=mx.cpu(),
                                          state_names=train_state_names, data_names=data_names,
                                          label_names=label_names, symbol=load_model.train())
        # eval data iter
        eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.test, vocab,
                                          eval_batch_size, args.bptt))
        cpu_train_mod.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label)

        # eval module
        eval_module = CustomModule(symbol=load_model.eval(), context=mx.cpu(), data_names=data_names,
                                   label_names=label_names, state_names=eval_state_names)
        # use `shared_module` to share parameter with the training module
        eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label,
                         shared_module=cpu_train_mod, for_training=False)
        val_L = run_utils.evaluate(eval_module, eval_data, epoch, 1000)
        train_data.reset()
    logging.info("Training completed. ")
