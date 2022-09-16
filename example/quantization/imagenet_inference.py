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

import argparse
import logging
import time

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms


def download_dataset(dataset_url, dataset_dir, logger=None):
    if logger is not None:
        logger.info(f'Downloading dataset for inference from {dataset_url} to {dataset_dir}')
    mx.test_utils.download(dataset_url, dataset_dir)


def score(symblock, data, ctx, max_num_examples, skip_num_batches, logger=None):
    metrics = [gluon.metric.create('acc'),
               gluon.metric.create('top_k_accuracy', top_k=5)]

    # make sure that fp32 inference works on the same images as calibrated quantized model
    logger.info(f'Skipping the first {skip_num_batches} batches')

    tic = time.time()
    num = 0
    for i, input_data in enumerate(data):
        if i < skip_num_batches:
            continue
        x = input_data[0].to_device(ctx)
        label = input_data[1].to_device(ctx)
        outputs = symblock.forward(x)
        for m in metrics:
            m.update(label, outputs)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    speed = num / (time.time() - tic)

    if logger is not None:
        logger.info(f'Finished inference with {num} images')
        logger.info(f'Finished with {speed} images per second')
        for m in metrics:
            logger.info(m.get())

def initialize_block_params(block, initializer):
    for _, param in block.collect_params('.*gamma|.*moving_var|.*running_var').items():
        param.initialize(mx.init.Constant(1))
    for _, param in block.collect_params('.*beta|.*moving_mean|.*running_mean|.*bias').items():
        param.initialize(mx.init.Constant(0))
    for _, param in block.collect_params('.*weight').items():
        param.initialize(initializer)

def benchmark_score(symblock, ctx, batch_size, warmup_batches, num_batches, data_layer_type):
    if data_layer_type == "int8":
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.int8)
    elif data_layer_type == 'uint8':
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.uint8)
    else:  # float32
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.float32)

    # get data
    if data_layer_type == "float32":
        data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx, dtype=data_layer_type)
                for _, shape in [dshape]]
    else:
        data = [mx.nd.full(shape=shape, val=127, ctx=ctx, dtype=data_layer_type)
                for _, shape in [dshape]]

    # run
    for i in range(warmup_batches+num_batches):
        if i == warmup_batches:
            tic = time.time()
        outputs = symblock.forward(*data)
        for output in outputs:
            output.wait_to_read()

    # return num images per second
    return num_batches * batch_size / (time.time() - tic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='cpu')
    parser.add_argument('--benchmark', type=bool, default=False, help='dummy data benchmark')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=False, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
    parser.add_argument('--num-warmup-batches', type=int, default=5, help='number of warmup batches used for benchmark')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the score dataset')
    parser.add_argument('--data-layer-type', type=str, default='float32',
                        choices=['float32', 'int8', 'uint8'],
                        help='data type for data layer (only with --benchmark)')

    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    if args.device == 'cpu':
        ctx = mx.cpu(0)
    elif args.device == 'gpu':
        ctx = mx.gpu(0)
        logger.warning('Notice that oneDNN optimized and quantized model may not work with GPU context')
    else:
        raise ValueError(f'ctx {args.device} is not supported in this script')

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info(f'batch size = {batch_size} for inference')

    rgb_mean = args.rgb_mean
    logger.info(f'rgb_mean = {rgb_mean}')
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    rgb_std = args.rgb_std
    logger.info(f'rgb_std = {rgb_std}')
    rgb_std = [float(i) for i in rgb_std.split(',')]

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info(f'Input data shape = {str(data_shape)}')

    data_layer_type = args.data_layer_type

    if not args.benchmark:
        dataset = args.dataset
        download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset)
        logger.info(f'Dataset for inference: {dataset}')

        dataset = mx.gluon.data.vision.ImageRecordDataset(dataset)
        transformer = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=rgb_mean, std=rgb_std)])
        data_loader = DataLoader(dataset.transform_first(
            transformer), batch_size, shuffle=args.shuffle_dataset, num_workers=data_nthreads)

        # loading model
        symblock = gluon.SymbolBlock.imports(symbol_file, ['data'], param_file)

        num_inference_images = args.num_inference_batches * batch_size
        logger.info(f'Running model {symbol_file} for inference')
        score(symblock, data_loader, ctx, max_num_examples=num_inference_images,
              skip_num_batches=args.num_skipped_batches, logger=logger)
    else:
        # loading model
        symblock = gluon.SymbolBlock.imports(symbol_file, ['data'])
        initialize_block_params(symblock, mx.init.One())

        logger.info(f'Running model {symbol_file} for inference.')
        logger.info(f'Warmup batches: {args.num_warmup_batches}')
        logger.info(f'Inference batches: {args.num_inference_batches}')
        speed = benchmark_score(symblock, ctx, batch_size,
                                args.num_warmup_batches, args.num_inference_batches, data_layer_type)
        logger.info('batch size %2d, image/sec: %f', batch_size, speed)
