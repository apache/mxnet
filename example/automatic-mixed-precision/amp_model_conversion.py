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
import logging
import argparse
import mxnet as mx
from common import modelzoo
import gluoncv
from gluoncv.model_zoo import get_model
from mxnet.contrib.amp import amp
import numpy as np

def download_model(model_name, logger=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if logger is not None:
        logger.info('Downloading model {}... into path {}'.format(model_name, model_path))
    return modelzoo.download_model(args.model, os.path.join(dir_path, 'model'))


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at {}'.format(fname))
    sym.save(fname, remove_amp_cast=False)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at {}'.format(fname))
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    symbolic_models = ['imagenet1k-resnet-152',
                       'imagenet1k-resnet-18',
                       'imagenet1k-resnet-34',
                       'imagenet1k-resnet-50',
                       'imagenet1k-resnet-101',
                       'imagenet1k-resnext-50',
                       'imagenet1k-resnext-101',
                       'imagenet1k-resnext-101-64x4d',
                       'imagenet11k-place365ch-resnet-152',
                       'imagenet11k-place365ch-resnet-50']
    gluon_models = ['resnet18_v1',
                    'resnet50_v1',
                    'resnet101_v1',
                    'squeezenet1.0',
                    'mobilenet1.0',
                    'mobilenetv2_1.0',
                    'inceptionv3']
    models = symbolic_models + gluon_models

    parser = argparse.ArgumentParser(description='Convert a provided FP32 model to a mixed precision model')
    parser.add_argument('--model', type=str, choices=models)
    parser.add_argument('--run-dummy-inference', action='store_true', default=False,
                        help='Will generate random input of shape (1, 3, 224, 224) '
                             'and run a dummy inference forward pass')
    parser.add_argument('--use-gluon-model', action='store_true', default=False,
                        help='If enabled, will download pretrained model from Gluon-CV '
                             'and convert to mixed precision model ')
    parser.add_argument('--cast-optional-params', action='store_true', default=False,
                        help='If enabled, will try to cast params to target dtype wherever possible')
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    if not args.use_gluon_model:
        assert args.model in symbolic_models, "Please choose one of the available symbolic models: {} \
                                               If you want to use gluon use the script with --use-gluon-model".format(symbolic_models)

        prefix, epoch = download_model(model_name=args.model, logger=logger)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym, arg_params, aux_params,
                                                                             cast_optional_params=args.cast_optional_params)
        sym_name = "%s-amp-symbol.json" % (prefix)
        save_symbol(sym_name, result_sym, logger)
        param_name = '%s-%04d.params' % (prefix + '-amp', epoch)
        save_params(param_name, result_arg_params, result_aux_params, logger)
        if args.run_dummy_inference:
            logger.info("Running inference on the mixed precision model with dummy input, batch size: 1")
            mod = mx.mod.Module(result_sym, data_names=['data'], label_names=['softmax_label'], context=mx.gpu(0))
            mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])
            mod.set_params(arg_params, aux_params)
            mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                                        label=[mx.nd.ones((1,))]))
            result = mod.get_outputs()[0].asnumpy()
            logger.info("Inference run successfully")
    else:
        assert args.model in gluon_models, "Please choose one of the available gluon models: {} \
                                            If you want to use symbolic model instead, remove --use-gluon-model when running the script".format(gluon_models)
        net = gluoncv.model_zoo.get_model(args.model, pretrained=True)
        net.hybridize()
        result_before1 = net.forward(mx.nd.zeros((1, 3, 224, 224)))
        net.export("{}".format(args.model))
        net = amp.convert_hybrid_block(net, cast_optional_params=args.cast_optional_params)
        net.export("{}-amp".format(args.model), remove_amp_cast=False)
        if args.run_dummy_inference:
            logger.info("Running inference on the mixed precision model with dummy inputs, batch size: 1")
            result_after = net.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))
            result_after = net.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))
            logger.info("Inference run successfully")
