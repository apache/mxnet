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
from get_data import get_libsvm_data
from fm_model import *
import argparse
import os

parser = argparse.ArgumentParser(description="Run factorization machine",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=8192,
                    help='number of examples per batch')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='what optimizer to use',
                    choices=["ftrl", "sgd", "adam"])

AVAZU = {
    'train': 'avazu-app',
    'test': 'avazu-app.t',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    # 1000000 + 1 since LibSVMIter uses zero-based indexing
    'num_features': 1000001,
}

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
    factor_size = 4

    # dataset
    num_features = AVAZU['num_features']
    data_dir = os.path.join(os.getcwd(), 'data')
    train_data = os.path.join(data_dir, AVAZU['train'])
    val_data = os.path.join(data_dir, AVAZU['test'])
    get_libsvm_data(data_dir, AVAZU['train'], AVAZU['url'])
    get_libsvm_data(data_dir, AVAZU['test'], AVAZU['url'])

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=train_data, data_shape=(num_features,),
                                  batch_size=batch_size)
    eval_data = mx.io.LibSVMIter(data_libsvm=val_data, data_shape=(num_features,),
                                 batch_size=batch_size)

    # model
    norm_init = mx.initializer.Normal(sigma=0.01)
    model = fm_model(factor_size, num_features, norm_init)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    optim = mx.optimizer.create(optimizer, learning_rate=0.01, rescale_grad=1.0/batch_size)
    mod.init_optimizer(optimizer=optim)
    # use accuracy as the metric
    metric = mx.metric.create(['mse'])
    speedometer = mx.callback.Speedometer(batch_size, 100)

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
        ## evaluate metric on validation dataset
        #score = mod.score(eval_data, ['nll_loss'])
        #logging.info('epoch %d, eval nll = %s ' % (epoch, score[0][1]))
        # reset the iterator for next pass of data
        data_iter.reset()
    logging.info('Training completed.')
