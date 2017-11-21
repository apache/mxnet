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
from data import DummyIter
from factorization_machine_model import *
import argparse, os

parser = argparse.ArgumentParser(description="Run factorization machine",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default="./data/",
                    help='training LibSVM files to use.')
parser.add_argument('--num-epoch', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='number of examples per batch')
parser.add_argument('--input-size', type=int, default=2000000,
                    help='number of features in the input')
parser.add_argument('--factor-size', type=int, default=16,
                    help='number of latent variables')
parser.add_argument('--kvstore', type=str, default='local',
                    help='what kvstore to use', choices=["dist_async", "local"])

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    kvstore = args.kvstore
    factor_size = args.factor_size
    num_features = args.input_size

    # create kvstore
    kv = mx.kvstore.create(kvstore)
    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=args.data, data_shape=(num_features,),
                                  batch_size=batch_size)
    # model
    model = factorization_machine_model(factor_size, num_features)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    # TODO LR, lambda?
    optim = mx.optimizer.create('adam', learning_rate=0.001, wd=0.0001,
                                beta1=0.9, beta2=0.999, epsilon=1e-8)
    mod.init_optimizer(optimizer=optim, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create(['accuracy'])
    speedometer = mx.callback.Speedometer(batch_size, 100)

    # get the sparse weight parameter
    w_index = mod._exec_group.param_names.index('w')
    w_param = mod._exec_group.param_arrays[w_index]
    v_index = mod._exec_group.param_names.index('v')
    v_param = mod._exec_group.param_arrays[v_index]

    logging.info('Training started ...')
    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in data_iter:
            nbatch += 1
            # for distributed training, we need to manually pull sparse weights from kvstore
            row_ids = batch.data[0].indices
            kv.row_sparse_pull('w', w_param, row_ids=[row_ids], priority=-w_index)
            kv.row_sparse_pull('v', v_param, row_ids=[row_ids], priority=-v_index)
            mod.forward_backward(batch)
            # update all parameters (including the weight parameter)
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # reset the iterator for next pass of data
        data_iter.reset()
    logging.info('Training completed.')
