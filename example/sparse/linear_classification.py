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
import argparse
import os

parser = argparse.ArgumentParser(description="Run sparse linear classification " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=8192,
                    help='number of examples per batch')
parser.add_argument('--kvstore', type=str, default=None,
                    help='what kvstore to use [local, dist_async, etc]')

AVAZU = {
    'train': 'avazu-app',
    'test': 'avazu-app.t',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    # 1000000 + 1 since LibSVMIter uses zero-based indexing
    'num_features': 1000001,
}

def linear_model(num_features):
    # data with csr storage type to enable feeding data with CSRNDArray
    x = mx.symbol.Variable("data", stype='csr')
    norm_init = mx.initializer.Normal(sigma=0.01)
    # weight with row_sparse storage type to enable sparse gradient updates
    weight = mx.symbol.Variable("weight", shape=(num_features, 2), init=norm_init, stype='row_sparse')
    bias = mx.symbol.Variable("bias", shape=(2, ))
    dot = mx.symbol.sparse.dot(x, weight)
    pred = mx.symbol.broadcast_add(dot, bias)
    y = mx.symbol.Variable("softmax_label")
    model = mx.symbol.SoftmaxOutput(data=pred, label=y, multi_output=True, name="out")
    return model

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    num_epoch = args.num_epoch
    kvstore = args.kvstore
    batch_size = args.batch_size

    # create kvstore
    kv = mx.kvstore.create(kvstore) if kvstore else None
    rank = kv.rank if kv else 0
    num_worker = kv.num_workers if kv else 1

    # dataset
    num_features = AVAZU['num_features']
    data_dir = os.path.join(os.getcwd(), 'data')
    train_data = os.path.join(data_dir, AVAZU['train'])
    val_data = os.path.join(data_dir, AVAZU['test'])
    get_libsvm_data(data_dir, AVAZU['train'], AVAZU['url'])
    get_libsvm_data(data_dir, AVAZU['test'], AVAZU['url'])

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=train_data, data_shape=(num_features,),
                                  batch_size=batch_size, num_parts=num_worker,
                                  part_index=rank)
    eval_data = mx.io.LibSVMIter(data_libsvm=val_data, data_shape=(num_features,),
                                 batch_size=batch_size)

    # model
    model = linear_model(num_features)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    sgd = mx.optimizer.SGD(momentum=0.0, clip_gradient=5.0,
                           learning_rate=0.001, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=sgd, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create('log_loss')

    logging.info('Training started ...')
    data_iter = iter(train_data)
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in data_iter:
            nbatch += 1
            # for distributed training, we need to explicitly pull sparse weights from kvstore
            if kv:
                row_ids = batch.data[0].indices
                # pull sparse weight based on the indices
                index = mod._exec_group.param_names.index('weight')
                kv.row_sparse_pull('weight', mod._exec_group.param_arrays[index],
                                   priority=-index, row_ids=[row_ids])
            mod.forward_backward(batch)
            # update parameters
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            if nbatch % 100 == 0:
                logging.info('epoch %d batch %d, train log loss = %s' % (epoch, nbatch, metric.get()[1]))
        # evaluate metric on validation dataset
        score = mod.score(eval_data, ['log_loss'])
        logging.info('epoch %d, eval log loss = %s' % (epoch, score[0][1]))
        # reset the iterator for next pass of data
        data_iter.reset()
    logging.info('Training completed.')
