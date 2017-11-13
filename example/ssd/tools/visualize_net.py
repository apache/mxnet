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

from __future__ import print_function
import find_mxnet
import mxnet as mx
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'symbol'))
import symbol_factory


parser = argparse.ArgumentParser(description='network visualization')
parser.add_argument('--network', type=str, default='vgg16_reduced',
                    help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=300,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

if not args.train:
    net = symbol_factory.get_symbol(args.network, args.data_shape, num_classes=args.num_classes)
    a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape)}, \
        node_attrs={"shape":'rect', "fixedsize":'false'})
    a.render("ssd_" + args.network + '_' + str(args.data_shape))
else:
    net = symbol_factory.get_symbol_train(args.network, args.data_shape, num_classes=args.num_classes)
    print(net.tojson())
