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

from __future__ import print_function
import argparse
from converter._mxnet_converter import convert
from converter.utils import load_model
import yaml
from ast import literal_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts an MXNet model to a CoreML model')

    parser.add_argument(
        '--model-prefix', required=True, type=str,
        help="Prefix of the existing model. The model is expected to be stored in the same directory from where "
             "this tool is being run. E.g. --model-prefix=squeezenet_v1.1. Note that this can include entire "
             "directory name too. E.g. --model-prefix=~/Downloads/squeezenet_v1.1."
    )
    parser.add_argument(
        '--epoch', required=True, type=int,
        help="The suffix of the MXNet model name which usually indicate the number of epochs. E.g. --epoch=0"
    )
    parser.add_argument(
        '--output-file', required=True, type=str,
        help="File where the resulting CoreML model will be saved. E.g. --output-file=\"squeezenet-v11.mlmodel\""
    )
    parser.add_argument(
        '--input-shape', required=True, type=str,
        help="Input shape information in a JSON string format. E.g. --input-shape='{\"data\":\"3,224,224\"}' where"
             " 'data' is the name of the input variable of the MXNet model and '3,244,244' is its shape "
             "(channel, height and weight) of the input image data."
    )
    parser.add_argument(
        '--label-names', required=False, type=str, default='softmax_label',
        help="label-names of the MXNet model's output variables. E.g. --label-names=softmax_label. "
             "(Usually this is the name of the last layer followed by suffix _label.)"
    )
    parser.add_argument(
        '--mode', required=False, type=str, default=None,
        help="When mode='classifier', a CoreML NeuralNetworkClassifier will be constructed. "
             "When mode='regressor', a CoreML NeuralNetworkRegressor will be constructed. "
             "When mode=None (default), a CoreML NeuralNetwork will be constructed."
    )
    parser.add_argument(
        '--class-labels', required=False, type=str, default=None,
        help="As a string it represents the name of the file which contains the classification labels (synset file)."
    )
    parser.add_argument(
        '--pre-processing-arguments', required=False, type=str, default=None,
        help="The parameters in the dictionary tell the converted coreml model how to pre-process any input "
             "before an inference is run on it. For the list of pre-processing arguments see https://goo.gl/GzFe86"
             "e.g. --pre-processing-arguments='{\"red_bias\": 127, \"blue_bias\":117, \"green_bias\": 103}'"
    )

    # TODO
    # We need to test how to use the order
    # parser.add_argument(
    #     '--order', required=True, type=str, default=None,
    #     help=""
    # )

    args, unknown = parser.parse_known_args()

    model_name = args.model_prefix
    epoch_num = args.epoch
    output_file = args.output_file
    mode = args.mode
    class_labels=args.class_labels

    # parse the input data name/shape and label name/shape
    input_shape = yaml.safe_load(args.input_shape)
    data_shapes = []
    for key in input_shape:
        # We prepend 1 because the coreml model only accept 1 input data at a time (=batch-size).
        shape = (1,)+literal_eval(input_shape[key])
        input_shape[key] = shape
        data_shapes.append((key, shape))

    # if label name is not in input then do not use the label
    label_names = [args.label_names,] if args.label_names in input_shape else None

    pre_processing_arguments = args.pre_processing_arguments

    mod = load_model(
        model_name=model_name,
        epoch_num=epoch_num,
        data_shapes=data_shapes,
        label_shapes=None,
        label_names=label_names
    )

    kwargs = {'input_shape': input_shape}
    if pre_processing_arguments is not None:
        kwargs['preprocessor_args'] = yaml.safe_load(pre_processing_arguments)

    coreml_model = convert(model=mod, mode=mode, class_labels=class_labels, **kwargs)
    coreml_model.save(output_file)
    print("\nSUCCESS\nModel %s has been converted and saved at %s\n" % (model_name, output_file))
