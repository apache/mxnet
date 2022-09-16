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

import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, "../../python"))
sys.path.insert(0, os.path.join(curr_path, "../../example/image-classification/symbols"))
import mxnet as mx
import logging
import argparse
import time
import numpy as np
from importlib import import_module
from collections import namedtuple
from functools import reduce

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="command for benchmark kv-store")
    parser.add_argument('--network', type=str, default="resnet",
                        help='the neural network to test')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='the gpus to be used, e.g "0,1,2,3"')
    parser.add_argument('--num-layers', type=int, default=152,
                        help='number of layers, can be used for resnet')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='number of batches to run')
    parser.add_argument('--disp-batches', type=int, default=1,
                        help='show averaged results for every n batches')
    parser.add_argument('--test-results', type=int, default=1,
                        help='if or not evalute the results correctness')
    parser.add_argument('--image-shape', type=str, default='3,224,224',
                        help='input images shape')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='None',
                        help='the optimizer set to kvstore. None means no optimizer')
    parser.add_argument('--gc-type', type=str, default='none',
                        help='type of gradient compression')
    args = parser.parse_args()
    logging.info(args)
    return args

def get_shapes(symbol, data_shape):
    arg_name = symbol.list_arguments()
    arg_shape, _, _ = symbol.infer_shape(data=data_shape)
    shapes = [s for n,s in zip(arg_name, arg_shape) if 'weight' in n or 'bias' in n]
    return shapes

def diff(a, b):
    return np.sum(np.abs(a.asnumpy() - b.asnumpy()))

def error(gpu_res, cpu_res):
    res = sum([sum([diff(a, b) for a in w]) for w, b in zip(gpu_res, cpu_res)])
    res /= sum([np.sum(np.abs(g.asnumpy())) for g in cpu_res])
    return res

def run(network, optimizer, gpus, kv_store, image_shape, disp_batches,
        num_batches, test_results, gc_type, **kwargs):
    # create kvstore and optimizer
    devs = [mx.gpu(int(i)) for i in gpus.split(',')]
    kv = mx.kv.create(kv_store)
    if gc_type != 'none':
        kv.set_gradient_compression({'type': gc_type})
    if optimizer is None or optimizer == 'None':
        opt = None
    else:
        opt = mx.optimizer.Optimizer.create_optimizer(optimizer)
        kv.set_optimizer(opt)
        updater = mx.optimizer.get_updater(mx.optimizer.Optimizer.create_optimizer(optimizer))

    # create network
    symbol = import_module(network).get_symbol(image_shape=image_shape, **kwargs)
    # a fake batch size 32, which does not affect the results
    data_shape = (32,) + tuple([int(s) for s in image_shape.split(',')])
    shapes = get_shapes(symbol, data_shape)

    size = float(sum([reduce(lambda x,y : x*y, s, 1) for s in shapes])) * 4 / 1e6
    logging.info(f'num of arrays = {len(shapes)}, total size = {size} MB')

    for i, s in enumerate(shapes):
        kv.init(i, mx.nd.zeros(s))

    grads_val = [[mx.random.uniform(-1,1,shape=s) for d in devs] for s in shapes]
    grads = [[g.as_in_context(d) for g, d in zip(gs, devs)] for gs in grads_val]
    weights = [[mx.nd.zeros(s, d) for d in devs] for s in shapes]

    cpu_grads = [mx.nd.array(sum([g.asnumpy() for g in gs]))*kv.num_workers for gs in grads_val]
    cpu_weights = [mx.nd.zeros(s) for s in shapes]
    toc = 0

    Results = namedtuple('Results', ['iter', 'time', 'bandwidth', 'error'])
    res = []
    for b in range(0, num_batches+1):
        tic = time.time()
        for i,g in enumerate(grads):
            kv.push(i, g, i)

        for i,w in enumerate(weights):
            kv.pull(i, w, i)
        for ws in weights:
            for w in ws:
                w.wait_to_read()
        toc += time.time() - tic
        if test_results:
            if opt == None:
                err = error(weights, cpu_grads)
            else:
                for i, wg in enumerate(zip(cpu_weights, cpu_grads)):
                    updater(i, wg[1], wg[0])
                err = error(weights, cpu_weights)
        else:
            err = -1

        if b % disp_batches == 0:
            toc /= disp_batches
            if b != 0:
                # 0 is used for warmup, ignored
                r = Results(iter=b, time=toc, error=err,
                            bandwidth=size*2*(len(devs)-1)/len(devs)/toc/1e3)
                logging.info(f'iter {r.iter}, {r.time} sec, {r.bandwidth} GB/sec per gpu, error {r.error}')
                res.append(r)
            toc = 0
    return res

if __name__ == "__main__":
    args = parse_args();
    run(**vars(args))
