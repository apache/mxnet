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

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit

if __name__ == '__main__':
    train_fname = os.path.join("data", "caltech-256-60-val.rec")
    val_fname = os.path.join("data", "caltech-256-60-train.rec")
    # parse args
    parser = argparse.ArgumentParser(description="train caltech256",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'mlp',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 256,
        num_examples  = 15420,
        image_shape    = '3,256,256',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 300,
        lr             = .05,
        lr_step_epochs = '200,250',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
