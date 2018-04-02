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
from data import get_iterator
import argparse
import train_model

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
    act1 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
    fc2  = mx.symbol.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
    act2 = mx.symbol.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
    fc3 = mx.symbol.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
    if use_caffe_loss:
        label = mx.symbol.Variable('softmax_label')
        mlp = mx.symbol.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
    else:
        mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return mlp

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')

    # first conv
    conv1 = mx.symbol.CaffeOp(data_0=data, num_weight=2, prototxt="layer{type:\"Convolution\" convolution_param { num_output: 20 kernel_size: 5 stride: 1} }")
    act1 = mx.symbol.CaffeOp(data_0=conv1, prototxt="layer{type:\"TanH\"}")
    pool1 = mx.symbol.CaffeOp(data_0=act1, prototxt="layer{type:\"Pooling\" pooling_param { pool: MAX kernel_size: 2 stride: 2}}")

    # second conv
    conv2 = mx.symbol.CaffeOp(data_0=pool1, num_weight=2, prototxt="layer{type:\"Convolution\" convolution_param { num_output: 50 kernel_size: 5 stride: 1} }")
    act2 = mx.symbol.CaffeOp(data_0=conv2, prototxt="layer{type:\"TanH\"}")
    pool2 = mx.symbol.CaffeOp(data_0=act2, prototxt="layer{type:\"Pooling\" pooling_param { pool: MAX kernel_size: 2 stride: 2}}")

    fc1 = mx.symbol.CaffeOp(data_0=pool2, num_weight=2, prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 500} }")
    act3 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")

    # second fullc
    fc2 = mx.symbol.CaffeOp(data_0=act3, num_weight=2, prototxt="layer{type:\"InnerProduct\"inner_product_param{num_output: 10} }")
    if use_caffe_loss:
        label = mx.symbol.Variable('softmax_label')
        lenet = mx.symbol.CaffeLoss(data=fc2, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
    else:
        lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_network_from_json_file(file_name):
    network = mx.sym.load(file_name)
    return network

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifier on mnist')
    parser.add_argument('--network', type=str, default='lenet',
                        help='the cnn to use (mlp | lenet | <path to network json file>')
    parser.add_argument('--caffe-loss', type=int, default=0,
                        help='Use CaffeLoss symbol')
    parser.add_argument('--caffe-data', action='store_true',
                        help='Use Caffe input-data layer only if specified')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_caffe_loss = args.caffe_loss
    use_caffe_data = args.caffe_data

    data_shape = ()
    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp()
    elif args.network == 'lenet':
        if not use_caffe_data:
            data_shape = (1, 28, 28)
        net = get_lenet()
    else:
        net = get_network_from_json_file(args.network)

    # train
    if use_caffe_loss:
        train_model.fit(args, net, get_iterator(data_shape, use_caffe_data), mx.metric.Caffe())
    else:
        train_model.fit(args, net, get_iterator(data_shape, use_caffe_data))
