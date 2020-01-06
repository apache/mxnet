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
# 
import os
import time
import argparse
import logging
import math
import random
import numpy as np
import mxnet as mx
from core.model import get_model
from core.dataset import NCFTestData
from core.evaluate import *
from mxnet.contrib.quantization import *

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run matrix factorization with embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m',
                    help='The dataset name.')
parser.add_argument('--max-user', type=int, default=138493,
                    help='max number of user index.')
parser.add_argument('--max-item', type=int, default=26744,
                    help='max number of item index.')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of examples per batch')
parser.add_argument('--topk', type=int, default=10,
                    help="topk for accuracy evaluation.")
parser.add_argument('--gpu', type=int, default=None,
                    help="index of gpu to run, e.g. 0 or 1. None means using cpu().")
parser.add_argument('--benchmark', action='store_true',  help="whether to benchmark performance only")
parser.add_argument('--epoch', type=int, default=7, help='model checkpoint index for inference')
parser.add_argument('--prefix', default='./model/ml-20m/neumf', help="model checkpoint prefix")
parser.add_argument('--calibration', action='store_true', help="whether to calibrate model")
parser.add_argument('--calib-mode', type=str, choices=['naive', 'entropy'], default='naive',
                    help='calibration mode used for generating calibration table for the quantized symbol; supports'
                            ' 1. naive: simply take min and max values of layer outputs as thresholds for'
                            ' quantization. In general, the inference accuracy worsens with more examples used in'
                            ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                            ' inference results.'
                            ' 2. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                            ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                            ' kinds of quantized models if the calibration dataset is representative enough of the'
                            ' inference dataset.')
parser.add_argument('--quantized-dtype', type=str, default='auto',
                    choices=['auto', 'int8', 'uint8'],
                    help='quantization destination data type for input data')
parser.add_argument('--num-calib-batches', type=int, default=10,
                    help='number of batches for calibration')

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    max_user = args.max_user
    max_item = args.max_item
    batch_size = args.batch_size
    benchmark = args.benchmark
    calibration = args.calibration
    calib_mode = args.calib_mode
    quantized_dtype = args.quantized_dtype
    num_calib_batches = args.num_calib_batches
    ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)
    topK = args.topk

    # prepare dataset
    if benchmark or calibration:
        logging.info('Prepare movielens dataset')
        val_iter = get_movielens_iter(args.path + args.dataset + '/test-ratings.csv', batch_size, ctx=ctx, logger=logging)
    else:
        logging.info('Prepare validation dataset')
        data = NCFTestData(args.path + args.dataset)
        testRatings, testNegatives= data.testRatings, data.testNegatives
        logging.info("Load validation data done. #user=%d, #item=%d, #test=%d" 
                    %(max_user, max_item, len(testRatings)))
        logging.info('Prepare validation dataset completed')
    
    # construct the model
    net, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    if ctx == mx.cpu() and calibration:
        net = net.get_backend_symbol('MKLDNN_QUANTIZE')

    # initialize the module
    mod = mx.module.Module(net, context=ctx, data_names=['user', 'item'], label_names=['softmax_label'])
    provide_data = [mx.io.DataDesc(name='item', shape=((batch_size,))),
                    mx.io.DataDesc(name='user', shape=((batch_size,)))]
    provide_label = [mx.io.DataDesc(name='softmax_label', shape=((batch_size,)))]
    mod.bind(for_training=False, data_shapes=provide_data, label_shapes=provide_label)
    mod.set_params(arg_params, aux_params)

    if calibration:
        logging.info('Quantizing FP32 model')
        excluded_sym_names = ['post_gemm_concat', 'fc_final']
        cqsym, cqarg_params, aux_params, collector = quantize_graph(sym=net, arg_params=arg_params, aux_params=aux_params,
                                                                    excluded_sym_names=excluded_sym_names,
                                                                    calib_mode=calib_mode,
                                                                    quantized_dtype=quantized_dtype, logger=logging)
        max_num_examples = num_calib_batches * batch_size
        mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
        num_batches = 0
        num_examples = 0
        for batch in val_iter:
            mod.forward(batch)
            num_batches += 1
            num_examples += batch_size
            if num_examples >= max_num_examples:
                break
        logging.info("Collected statistics from %d batches with batch_size=%d"
                    % (num_batches, batch_size))
        cqsym, cqarg_params, aux_params = calib_graph(qsym=cqsym, arg_params=arg_params, aux_params=aux_params,
                                                      collector=collector, calib_mode=calib_mode,
                                                      quantized_dtype=quantized_dtype, logger=logging)                                                       
        sym_name = '%s-symbol.json' % (args.prefix + '-quantized')
        cqsym = cqsym.get_backend_symbol('MKLDNN_QUANTIZE')
        mx.model.save_checkpoint(args.prefix + '-quantized', args.epoch, cqsym, cqarg_params, aux_params)
    elif benchmark:
        logging.info('Benchmarking...')
        data = [mx.random.randint(0, 1000, shape=shape, ctx=ctx) for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, []) # empty label
        for i in range(2000):
            mod.forward(batch, is_train=False)
        logging.info('Benchmarking...')
        num_samples = 0
        for ib, batch in enumerate(val_iter):
            if ib == 5:
                num_samples = 0
                tic = time.time()
            mod.forward(batch, is_train=False)
            mx.nd.waitall()
            num_samples += batch_size
        toc = time.time()
        fps = num_samples/(toc - tic)
        logging.info('Evaluating completed')
        logging.info('Inference speed %.4f fps' % fps)
    else:
        logging.info('Evaluating...')
        (hits, ndcgs) = evaluate_model(mod, testRatings, testNegatives, topK, batch_size, ctx, logging)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        logging.info('Evaluate: HR = %.4f, NDCG = %.4f'  % (hr, ndcg))

