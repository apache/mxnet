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
from metric import *
from mxnet.test_utils import *
from model import factorization_machine_model
import argparse, os

parser = argparse.ArgumentParser(description="Run factorization machine with criteo dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-train', type=str, default=None,
                    help='training dataset in LibSVM format.')
parser.add_argument('--data-test', type=str, default=None,
                    help='test dataset in LibSVM format.')
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='number of examples per batch')
parser.add_argument('--input-size', type=int, default=1000000,
                    help='number of features in the input')
parser.add_argument('--factor-size', type=int, default=16,
                    help='number of latent variables')
parser.add_argument('--factor-lr', type=float, default=0.0001,
                    help='learning rate for factor terms')
parser.add_argument('--linear-lr', type=float, default=0.001,
                    help='learning rate for linear terms')
parser.add_argument('--bias-lr', type=float, default=0.1,
                    help='learning rate for bias terms')
parser.add_argument('--factor-wd', type=float, default=0.00001,
                    help='weight decay rate for factor terms')
parser.add_argument('--linear-wd', type=float, default=0.001,
                    help='weight decay rate for linear terms')
parser.add_argument('--bias-wd', type=float, default=0.01,
                    help='weight decay rate for bias terms')
parser.add_argument('--factor-sigma', type=float, default=0.001,
                    help='standard deviation for initialization of factor terms')
parser.add_argument('--linear-sigma', type=float, default=0.01,
                    help='standard deviation for initialization of linear terms')
parser.add_argument('--bias-sigma', type=float, default=0.01,
                    help='standard deviation for initialization of bias terms')
parser.add_argument('--log-interval', type=int, default=100,
                    help='number of batches between logging messages')
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
    log_interval = args.log_interval
    assert(args.data_train is not None and args.data_test is not None), \
          "dataset for training or test is missing"

    def batch_row_ids(data_batch):
        """ Generate row ids based on the current mini-batch """
        idx = data_batch.data[0].indices
        return {'w': idx, 'v': idx}

    def all_row_ids(data_batch):
        """ Generate row ids for all rows """
        all_rows = mx.nd.arange(0, num_features, dtype='int64')
        return {'w': all_rows, 'v': all_rows}

    # create kvstore
    kv = mx.kvstore.create(kvstore)
    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=args.data_train, data_shape=(num_features,),
                                  batch_size=batch_size)
    eval_data = mx.io.LibSVMIter(data_libsvm=args.data_test, data_shape=(num_features,),
                                 batch_size=batch_size)
    # model
    lr_config = {'v': args.factor_lr, 'w': args.linear_lr, 'w0': args.bias_lr}
    wd_config = {'v': args.factor_wd, 'w': args.linear_wd, 'w0': args.bias_wd}
    init_config = {'v': mx.initializer.Normal(args.factor_sigma),
                   'w': mx.initializer.Normal(args.linear_sigma),
                   'w0': mx.initializer.Normal(args.bias_sigma)}
    model = factorization_machine_model(factor_size, num_features, lr_config, wd_config, init_config)

    # module
    mod = mx.mod.Module(symbol=model)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    optimizer_params=(('learning_rate', 1), ('wd', 1), ('beta1', 0.9),
                      ('beta2', 0.999), ('epsilon', 1e-8))
    mod.init_optimizer(optimizer='adam', kvstore=kv, optimizer_params=optimizer_params)

    # metrics
    metric = mx.metric.create(['log_loss', 'auc'])
    speedometer = mx.callback.Speedometer(batch_size, log_interval)

    logging.info('Training started ...')
    train_iter = iter(train_data)
    eval_iter = iter(eval_data)
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in train_iter:
            try:
                nbatch += 1
                # manually pull sparse weights from kvstore so that _square_sum
                # only computes the rows necessary
                mod.prepare(batch, sparse_row_id_fn=batch_row_ids)
                mod.forward_backward(batch)
                # update all parameters (including the weight parameter)
                mod.update()
                # update training metric
                mod.update_metric(metric, batch.label)
                speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                            eval_metric=metric, locals=locals())
                speedometer(speedometer_param)
            except:
                continue

        # pull all updated rows before validation
        mod.prepare(None, all_row_ids)
        # evaluate metric on validation dataset
        score = mod.score(eval_iter, ['log_loss'])
        logging.info("epoch %d, eval log loss = %s" % (epoch, score[0][1]))
        # reset the iterator for next pass of data
        train_iter.reset()
        eval_iter.reset()
    logging.info('Training completed.')
