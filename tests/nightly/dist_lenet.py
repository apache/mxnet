#!/usr/bin/env python

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

# distributed lenet
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../example/image-classification"))
sys.path.append(os.path.join(curr_path, "../../python"))
import mxnet as mx
import argparse
import train_mnist
import logging

if __name__ == '__main__':
    args = train_mnist.parse_args()
    args.batch_size = 100
    data_shape = (1, 28, 28)
    loader = train_mnist.get_iterator(data_shape)
    kv = mx.kvstore.create(args.kv_store)
    (train, val) = loader(args, kv)
    net = train_mnist.get_lenet()

    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mx.model.FeedForward.create(
        ctx           = mx.gpu(kv.rank),
        kvstore       = kv,
        symbol        = net,
        X             = train,
        eval_data     = val,
        num_epoch     = args.num_epochs,
        learning_rate = args.lr,
        momentum      = 0.9,
        wd            = 0.00001)
