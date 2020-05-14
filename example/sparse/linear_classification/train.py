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
from data import get_avazu_data
from linear_model import *
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
                    help='what kvstore to use',
                    choices=["dist_sync", "dist_async", "local"])
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='what optimizer to use',
                    choices=["adagrad", "sgd", "adam"])

AVAZU = {
    'train': 'avazu-app',
    'test': 'avazu-app.t',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    # 1000000 + 1 since LibSVMIter uses zero-based indexing
    'num_features': 1000001,
}

def batch_row_ids(data_batch):
    """ Generate row ids based on the current mini-batch """
    return {'weight': data_batch.data[0].indices}

def all_row_ids(data_batch):
    """ Generate row ids for all rows """
    all_rows = mx.nd.arange(0, AVAZU['num_features'], dtype='int64')
    return {'weight': all_rows}

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    kvstore = args.kvstore
    batch_size = args.batch_size
    optimizer = args.optimizer

    # create kvstore
    kv = mx.kvstore.create(kvstore) if kvstore else None
    rank = kv.rank if kv else 0
    num_worker = kv.num_workers if kv else 1

    # dataset
    num_features = AVAZU['num_features']
    data_dir = os.path.join(os.getcwd(), 'data')
    train_data = os.path.join(data_dir, AVAZU['train'])
    val_data = os.path.join(data_dir, AVAZU['test'])
    get_avazu_data(data_dir, AVAZU['train'], AVAZU['url'])
    get_avazu_data(data_dir, AVAZU['test'], AVAZU['url'])

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=train_data, data_shape=(num_features,),
                                  batch_size=batch_size, num_parts=num_worker,
                                  part_index=rank)
    eval_data = mx.io.LibSVMIter(data_libsvm=val_data, data_shape=(num_features,),
                                 batch_size=batch_size)

    # model
    # The positive class weight, says how much more we should upweight the importance of
    # positive instances in the objective function.
    # This is used to combat the extreme class imbalance.
    positive_class_weight = 2
    model = linear_model(num_features, positive_class_weight)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params()
    optim = mx.optimizer.create(optimizer, learning_rate=0.01, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=optim, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create(['nll_loss'])

    # get the sparse weight parameter
    speedometer = mx.callback.Speedometer(batch_size, 100)

    logging.info('Training started ...')
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in train_data:
            nbatch += 1
            # for distributed training, we need to manually pull sparse weights from kvstore
            mod.prepare(batch, sparse_row_id_fn=batch_row_ids)
            mod.forward_backward(batch)
            # update all parameters (including the weight parameter)
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)

        # prepare the module weight with all row ids for inference. Alternatively, one could call
        # score = mod.score(val_iter, ['MSE'], sparse_row_id_fn=batch_row_ids)
        # to fetch the weight per mini-batch
        mod.prepare(None, all_row_ids)
        # evaluate metric on validation dataset
        score = mod.score(eval_data, ['nll_loss'])
        logging.info('epoch %d, eval nll = %s ' % (epoch, score[0][1]))

        # prepare the module weight with all row ids before making a checkpoint.
        mod.prepare(None, all_row_ids)
        mod.save_checkpoint("checkpoint", epoch)
        # reset the iterator for next pass of data
        train_data.reset()
        eval_data.reset()
    logging.info('Training completed.')
