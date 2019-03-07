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
""" An example of predicting CAPTCHA image data with a LSTM network pre-trained with a CTC loss"""

from __future__ import print_function

import argparse

import numpy as np
from ctc_metrics import CtcMetrics
import cv2
from hyperparams import Hyperparams
import lstm
import mxnet as mx
from ocr_iter import SimpleBatch


def read_img(path):
    """ Reads image specified by path into numpy.ndarray"""
    img = cv2.resize(cv2.imread(path, 0), (80, 30)).astype(np.float32) / 255
    img = np.expand_dims(img.transpose(1, 0), 0)
    return img


def lstm_init_states(batch_size):
    """ Returns a tuple of names and zero arrays for LSTM init states"""
    hp = Hyperparams()
    init_shapes = lstm.init_states(batch_size=batch_size, num_lstm_layer=hp.num_lstm_layer, num_hidden=hp.num_hidden)
    init_names = [s[0] for s in init_shapes]
    init_arrays = [mx.nd.zeros(x[1]) for x in init_shapes]
    return init_names, init_arrays


def load_module(prefix, epoch, data_names, data_shapes):
    """Loads the model from checkpoint specified by prefix and epoch, binds it
    to an executor, and sets its parameters and returns a mx.mod.Module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    pred_fc = sym.get_internals()['pred_fc_output']
    sym = mx.sym.softmax(data=pred_fc)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod


def main():
    """Program entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the CAPTCHA image file")
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='ocr')
    parser.add_argument("--epoch", help="Checkpoint epoch [Default 100]", type=int, default=100)
    args = parser.parse_args()

    init_state_names, init_state_arrays = lstm_init_states(batch_size=1)
    img = read_img(args.path)

    sample = SimpleBatch(
        data_names=['data'] + init_state_names,
        data=[mx.nd.array(img)] + init_state_arrays)

    mod = load_module(args.prefix, args.epoch, sample.data_names, sample.provide_data)

    mod.forward(sample)
    prob = mod.get_outputs()[0].asnumpy()

    prediction = CtcMetrics.ctc_label(np.argmax(prob, axis=-1).tolist())
    # Predictions are 1 to 10 for digits 0 to 9 respectively (prediction 0 means no-digit)
    prediction = [p - 1 for p in prediction]
    print("Digits:", prediction)


if __name__ == '__main__':
    main()
