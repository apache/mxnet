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

# pylint: skip-file
import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
import symbol_fcnxs
import init_fcnxs
from data import FileIter
from solver import Solver

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    ctx = mx.cpu() if not args.gpu else mx.gpu(args.gpu)
    fcnxs = symbol_fcnxs.get_fcn32s_symbol(numclass=21, workspace_default=1536)
    fcnxs_model_prefix = "model_pascal/FCN32s_VGG16"
    if args.model == "fcn16s":
        fcnxs = symbol_fcnxs.get_fcn16s_symbol(numclass=21, workspace_default=1536)
        fcnxs_model_prefix = "model_pascal/FCN16s_VGG16"
    elif args.model == "fcn8s":
        fcnxs = symbol_fcnxs.get_fcn8s_symbol(numclass=21, workspace_default=1536)
        fcnxs_model_prefix = "model_pascal/FCN8s_VGG16"
    arg_names = fcnxs.list_arguments()
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, args.epoch)
    if not args.retrain:
        if args.init_type == "vgg16":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
        elif args.init_type == "fcnxs":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    train_dataiter = FileIter(
        root_dir             = "./VOC2012",
        flist_name           = "train.lst",
        # cut_off_size         = 400,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = "./VOC2012",
        flist_name           = "val.lst",
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = 0,
        num_epoch           = 50,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs,
        learning_rate       = 1e-10,
        momentum            = 0.99,
        wd                  = 0.0005)
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcnxs',
        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
        help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=74,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--init-type', default="vgg16",
        help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    parser.add_argument("--gpu", type=int, help="0 to use GPU, not set to use CPU")
    args = parser.parse_args()
    logging.info(args)
    main()
