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
import time
import argparse
import os

parser = argparse.ArgumentParser(description="Run sparse linear classification " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--profiler', type=int, default=0,
                    help='whether to use profiler')
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=8192,
                    help='number of examples per batch')
parser.add_argument('--num-batch', type=int, default=99999999,
                    help='number of batches per epoch')
parser.add_argument('--dummy-iter', type=int, default=0,
                    help='whether to use dummy iterator to exclude io cost')
parser.add_argument('--kvstore', type=str, default='dist_sync',
                    help='what kvstore to use [local, dist_sync, etc]')
parser.add_argument('--log-level', type=str, default='DEBUG',
                    help='logging level [debug, info, error]')
parser.add_argument('--dataset', type=str, default='avazu',
                    help='what test dataset to use')

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

# testing dataset sources
avazu = {
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
}

kdda = {
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
}

datasets = { 'kdda' : kdda, 'avazu' : avazu }

def linear_model(feature_dim):
     x = mx.symbol.Variable("data", stype='csr')
     norm_init = mx.initializer.Normal(sigma=0.01)
     weight = mx.symbol.Variable("weight", shape=(feature_dim, 1), init=norm_init, stype='row_sparse')
     bias = mx.symbol.Variable("bias", shape=(1,), init=norm_init)
     dot = mx.symbol.dot(x, weight)
     pred = mx.symbol.broadcast_add(dot, bias)
     y = mx.symbol.Variable("softmax_label")
     model = mx.symbol.SoftmaxOutput(data=pred, label=y, name="out")
     return model

if __name__ == '__main__':
    # arg parser
    args = parser.parse_args()
    num_epoch = args.num_epoch
    num_batch = args.num_batch
    kvstore = args.kvstore
    profiler = args.profiler > 0
    batch_size = args.batch_size
    dummy_iter = args.dummy_iter
    dataset = args.dataset
    log_level = args.log_level

    # create kvstore
    kv = mx.kvstore.create(kvstore)
    rank = kv.rank
    num_worker = kv.num_workers

    # only print log for rank 0 worker
    import logging
    if rank != 0:
        log_level = logging.ERROR
    elif log_level == 'DEBUG':
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=log_level, format=head)

    # dataset
    assert(dataset in datasets), "unknown dataset " + dataset
    metadata = datasets[dataset]
    feature_dim = metadata['feature_dim']
    if logging:
        logging.debug('preparing data ... ')
    data_dir = os.path.join(os.getcwd(), 'data')
    path = os.path.join(data_dir, metadata['data_name'])
    if not os.path.exists(path):
        get_libsvm_data(data_dir, metadata['data_name'], metadata['url'],
                        metadata['data_origin_name'])
        assert os.path.exists(path)

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=path, data_shape=(feature_dim,),
                                  batch_size=batch_size, num_parts=num_worker,
                                  part_index=rank)
    if dummy_iter:
        train_data = DummyIter(train_data)

    # model
    model = linear_model(feature_dim)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    sgd = mx.optimizer.SGD(momentum=0.0, clip_gradient=5.0,
                           learning_rate=0.1, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=sgd, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create('Accuracy')

    # start profiler
    if profiler:
        name = 'profile_output_' + str(num_worker) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=name)
        mx.profiler.profiler_set_state('run')

    logging.debug('start training ...')
    start = time.time()
    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        data_iter.reset()
        metric.reset()
        for batch in data_iter:
            nbatch += 1
            row_ids = batch.data[0].indices
            # pull sparse weight
            index = mod._exec_group.param_names.index('weight')
            kv.row_sparse_pull('weight', mod._exec_group.param_arrays[index],
                               priority=-index, row_ids=[row_ids])
            mod.forward_backward(batch)
            # update parameters
            mod.update()
            # accumulate prediction accuracy
            mod.update_metric(metric, batch.label)
            if nbatch == num_batch:
                break
        logging.info('epoch %d, %s' % (epoch, metric.get()))
    if profiler:
        mx.profiler.profiler_set_state('stop')
    end = time.time()
    time_cost = end - start
    logging.info('num_worker = ' + str(num_worker) + ', time cost = ' + str(time_cost))
