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

import time
import argparse
import os
import multiprocessing
from mxnet.test_utils import *

MAX_NUM_BATCH = 99999999
COMP = "compute"
COMM = "communication"
IO = "io"

parser = argparse.ArgumentParser(description="Run sparse linear regression " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--profiler', type=int, default=0,
                    help='whether to use profiler')
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='number of examples per batch')
parser.add_argument('--num-batch', type=int, default=MAX_NUM_BATCH,
                    help='number of batches per epoch')
parser.add_argument('--dummy-iter', type=int, default=0,
                    help='whether to use dummy iterator to exclude io cost')
parser.add_argument('--kvstore', type=str, default=None,
                    help='what kvstore to use [local, dist_sync, etc]')
parser.add_argument('--sparse-log-level', type=str, default='DEBUG',
                    help='logging level [DEBUG, INFO, ERROR]')
parser.add_argument('--dataset', type=str, default='avazu',
                    help='what test dataset to use')
parser.add_argument('--num-gpu', type=int, default=0,
                    help='number of gpus to use. 0 means using cpu(0);'
                         'otherwise, use gpu(0),...,gpu(num_gpu-1)')
parser.add_argument('--output-dim', type=int, default=4,
                    help='number of columns of the forward output')
parser.add_argument('--dummy-metric', type=int, default=0,
                    help='whether to call update_metric')
parser.add_argument('--enable-logging-for', default="0",
                    help="Enable logging for the specified list of workers")
parser.add_argument('--measure-only', default=None,
                    help="Measure only",
                    choices=[IO, COMP, COMM])
parser.add_argument('--omit-row-sparse-push', action='store_true',
                    help="omit row_sparse_push")

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
    'feature_dim': 1000001,
    'lc': 1719304,
}

kdda = {
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216831,
    'lc': 510302,
}

criteo = {
    'data_name': 'criteo.t',
    'data_origin_name': 'criteo.t.bz2',
    'url': "https://s3-us-west-2.amazonaws.com/sparse-dataset/criteo.t.bz2",
    'feature_dim': 8388621,
    'lc': 548787,
}

datasets = { 'kdda' : kdda, 'avazu' : avazu , 'criteo': criteo }


def get_sym(feature_dim):
    inputs = mx.symbol.Variable("data", stype='csr')
    norm_init = mx.initializer.Normal(sigma=0.01)
    weights = mx.symbol.Variable("w", shape=(feature_dim, args.output_dim),
                                 init=norm_init, stype='row_sparse')
    embed = mx.symbol.sparse.dot(inputs, weights)
    softmax_output = mx.symbol.Variable("softmax_label")
    model = mx.symbol.SoftmaxOutput(data=embed, label=softmax_output, name="out")
    return model


def row_sparse_push(kv, param_arrays, grad_arrays, param_names):
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        name = param_names[index]
        kv.push(name, grad_list, priority=-index)


def row_sparse_pull(kv, key, data, slices, weight_array, priority):
    # if have kvstore, need to pull corresponding rows of
    # the weights to each context
    # column indices (NDArray type) of the csr data
    # used as the row_idx of the weight row-sparse matrix
    row_indices = data.indices
    if len(slices) == 1:
        kv.row_sparse_pull(key, weight_array, priority=priority, row_ids=row_indices)
    else:  # more than one slices, multi-GPU training. Need to retain weight rows according to data slices
        # TODO(junwu):
        # the following line blocks, may need to pre-compute
        # and cache it outside the for loop
        indptr = data.indptr.asnumpy()
        row_idx_array = []
        for s in slices:
            row_idx_array.append(row_indices[indptr[s.start]:indptr[s.stop]])
        kv.row_sparse_pull(key, weight_array, priority=priority, row_ids=row_idx_array)


if __name__ == '__main__':

    # arg parser
    args = parser.parse_args()
    num_epoch = args.num_epoch
    num_batch = args.num_batch
    kvstore = args.kvstore
    profiler = args.profiler > 0
    batch_size = args.batch_size if args.num_gpu == 0 else args.num_gpu * args.batch_size
    dummy_iter = args.dummy_iter
    dataset = args.dataset
    log_level = args.sparse_log_level
    measure_only = args.measure_only
    num_cores = multiprocessing.cpu_count()
    omit_row_sparse_push = args.omit_row_sparse_push
    if measure_only == COMP or measure_only == IO:
        assert not kvstore, "when compute_only or io_only is set, kvstore should be None"
        num_batch = datasets[dataset]['lc'] / batch_size if num_batch == MAX_NUM_BATCH else num_batch
    if measure_only == COMM:
        assert (kvstore == "dist_async"), "when communication_only is set kvstore should be dist_async"
        num_batch = datasets[dataset]['lc'] / batch_size if num_batch == MAX_NUM_BATCH else num_batch


    contexts = mx.context.cpu(0) if args.num_gpu < 1\
        else [mx.context.gpu(i) for i in range(args.num_gpu)]

    # create kvstore when there are gpus
    kv = mx.kvstore.create(kvstore) if kvstore else None
    rank = kv.rank if kv is not None else 0
    num_worker = kv.num_workers if kv is not None else 1

    # only print log for rank 0 worker
    import logging
    if log_level == 'ERROR':
        log_level = logging.ERROR
    elif log_level == 'DEBUG':
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Only log if it is in the list of workers to be logged
    logging_workers_list = [int(i) for i in args.enable_logging_for.split(",")]
    log_level = log_level if rank in logging_workers_list else logging.CRITICAL

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
        get_bz2_data(data_dir, metadata['data_name'], metadata['url'],
                        metadata['data_origin_name'])
        assert os.path.exists(path)

    # data iterator
    train_data = mx.io.LibSVMIter(data_libsvm=path, data_shape=(feature_dim,),
                                  batch_size=batch_size, num_parts=num_worker,
                                  part_index=rank)
    if dummy_iter or measure_only == COMP or measure_only  == COMM:
        train_data = DummyIter(train_data)

    # model
    model = get_sym(feature_dim)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'],
                        label_names=['softmax_label'], context=contexts)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    sgd = mx.optimizer.SGD(momentum=0.0, clip_gradient=5.0,
                           learning_rate=0.1, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=sgd, kvstore=kv)
    # use accuracy as the metric
    metric = mx.metric.create('acc')

    index = mod._exec_group.param_names.index('w')
    # weight_array bound to executors of the contexts
    weight_array = mod._exec_group.param_arrays[index]

    mx.nd.waitall()  # sync point for initialization
    # start profiler
    if profiler:
        device = 'cpu'
        if args.num_gpu > 0:
            device = 'gpu' + str(args.num_gpu)
        name = 'profile_' + args.dataset + '_' + device + '_nworker' + str(num_worker)\
               + '_batchsize' + str(args.batch_size) + '_outdim' + str(args.output_dim) + '.json'
        mx.profiler.set_config(profile_all=True, filename=name)
        mx.profiler.set_state('run')

    logging.debug('start training ...')
    start = time.time()
    data_iter = iter(train_data)
    time_cost_epoch = 0.
    sum_cost_epoch = 0.
    average_cost_epoch = 0.

    for epoch in range(num_epoch):
        start_time_epoch = time.time()
        nbatch = 0
        end_of_batch = False
        metric.reset()
        next_batch = next(data_iter)
        if kv is not None:
            row_sparse_pull(kv, 'w', next_batch.data[0], mod._exec_group.slices, weight_array, -index)
        while not end_of_batch:
            nbatch += 1
            batch = next_batch

            if measure_only != IO and measure_only != COMM:
                mod.forward_backward(batch)
                # update parameters
                mod.update()
            if measure_only == COMM:
                if nbatch == 1:
                    mod.forward_backward(batch)
                    mod.update()
                elif not omit_row_sparse_push:
                    row_sparse_push(kv, mod._exec_group.param_arrays, mod._exec_group.grad_arrays, mod._exec_group.param_names)


            try:
                # pre fetch next batch
                next_batch = next(data_iter)
                if nbatch == num_batch:
                    raise StopIteration
                if kv is not None:
                    row_sparse_pull(kv, 'w', next_batch.data[0], mod._exec_group.slices, weight_array, -index)
            except StopIteration:
                end_of_batch = True
            # accumulate prediction accuracy
            if args.dummy_metric == 0:
                mod.update_metric(metric, batch.label)
            else:  # call waitall to replace update_metric as sync point
                mx.nd.waitall()  # sync point for the current minibatch
        logging.info('epoch {}, {}'.format(epoch, metric.get()))
        end_time_epoch = time.time()
        if epoch == 0:
            logging.debug("num_batches = {}".format(nbatch))
            logging.info('|device|num_worker|average_cost_epoch|rank|')
        time_cost_epoch = end_time_epoch - start_time_epoch
        if epoch > 0:
            sum_cost_epoch = sum_cost_epoch + time_cost_epoch
            average_cost_epoch = float(sum_cost_epoch) / epoch
        logging.info('num_worker = {}, time cost per epoch = {}'.format(str(num_worker), str(time_cost_epoch)))
        if args.num_gpu < 1:
            logging.info('|cpu/{} cores| {} | {} | {} |'.format(str(num_cores), str(num_worker), str(average_cost_epoch), rank))
        data_iter.reset()
    if profiler:
        mx.profiler.set_state('stop')
    end = time.time()
    time_cost = end - start
    logging.info('num_worker = {}, rank = {}, time cost = {}'.format(str(num_worker), str(rank), str(time_cost)))
