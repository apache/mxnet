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

import mxnet as mx
from mxnet.test_utils import *
from config import *
from data import get_uci_adult
from model import wide_deep_model
import argparse
import os


parser = argparse.ArgumentParser(description="Run sparse wide and deep classification ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=100,
                    help='number of examples per batch')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='what optimizer to use',
                    choices=["ftrl", "sgd", "adam"])
parser.add_argument('--log-interval', type=int, default=100,
                    help='number of batches to wait before logging training status')


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = args.optimizer
    log_interval = args.log_interval
    lr = args.lr
    ctx = mx.gpu(0) if args.gpu else mx.cpu()

    # dataset    
    data_dir = os.path.join(os.getcwd(), 'data')
    train_data = os.path.join(data_dir, ADULT['train'])
    val_data = os.path.join(data_dir, ADULT['test'])
    train_csr, train_dns, train_label = get_uci_adult(data_dir, ADULT['train'], ADULT['url'])
    val_csr, val_dns, val_label = get_uci_adult(data_dir, ADULT['test'], ADULT['url'])

    model = wide_deep_model(ADULT['num_linear_features'], ADULT['num_embed_features'],
                            ADULT['num_cont_features'], ADULT['embed_input_dims'],
                            ADULT['hidden_units'])

    # data iterator
    train_data = mx.io.NDArrayIter({'csr_data': train_csr, 'dns_data': train_dns},
                                   {'softmax_label': train_label}, batch_size,
                                   shuffle=True, last_batch_handle='discard')
    eval_data = mx.io.NDArrayIter({'csr_data': val_csr, 'dns_data': val_dns},
                                  {'softmax_label': val_label}, batch_size,
                                  shuffle=True, last_batch_handle='discard')
    
    # module
    mod = mx.mod.Module(symbol=model, context=ctx, data_names=['csr_data', 'dns_data'],
                        label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    optim = mx.optimizer.create(optimizer, learning_rate=lr, rescale_grad=1.0/batch_size)
    mod.init_optimizer(optimizer=optim)
    # use accuracy as the metric
    metric = mx.metric.create(['acc'])
    # get the sparse weight parameter
    speedometer = mx.callback.Speedometer(batch_size, log_interval)

    logging.info('Training started ...')
    
    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in data_iter:
            nbatch += 1
            mod.forward_backward(batch)
            # update all parameters (including the weight parameter)
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # evaluate metric on validation dataset
        score = mod.score(eval_data, ['acc'])
        logging.info('epoch %d, accuracy = %s' % (epoch, score[0][1]))
        
        mod.save_checkpoint("checkpoint", epoch, save_optimizer_states=True)
        # reset the iterator for next pass of data
        data_iter.reset()
    
    logging.info('Training completed.')
