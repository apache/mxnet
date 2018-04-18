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

import sys
import cv2
import numpy as np
import mxnet as mx
from collections import namedtuple
from ocr_iter import SimpleBatch
from captcha_generator import DigitCaptcha
from ctc_metrics import CtcMetrics
import lstm
from hyperparams import Hyperparams


class lstm_ocr_model(object):
    # Keep Zero index for blank. (CTC request it)
    CONST_CHAR = '0123456789'

    def __init__(self, path_of_json, path_of_params):
        super(lstm_ocr_model, self).__init__()
        self.path_of_json = path_of_json
        self.path_of_params = path_of_params
        self.predictor = None
        self.__init_ocr()

    def __init_ocr(self):
        num_label = 4 # Set your max length of label, add one more for blank
        batch_size = 1

        num_hidden = 100
        num_lstm_layer = 2
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_states = init_c + init_h

        init_state_arrays = np.zeros((batch_size, num_hidden), dtype="float32")
        self.init_state_dict = {}
        for x in init_states:
            self.init_state_dict[x[0]] = init_state_arrays

        all_shapes = [('data', (batch_size, 80, 30))] + init_states + [('label', (batch_size, num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json, 'rb').read(),
                                   open(self.path_of_params, 'rb').read(),
                                   all_shapes_dict)

    def forward_ocr(self, img_):
        img_ = cv2.resize(img_, (80, 30))
        img_ = img_.transpose(1, 0)
        print(img_.shape)
        img_ = img_.reshape((1, 80, 30))
        print(img_.shape)
        # img_ = img_.reshape((80 * 30))
        img_ = np.multiply(img_, 1 / 255.0)
        self.predictor.forward(data=img_, **self.init_state_dict)
        prob = self.predictor.get_output(0)
        label_list = []
        for p in prob:
            print(np.argsort(p))
            max_index = np.argsort(p)[::-1][0]
            label_list.append(max_index)
        return self.__get_string(label_list)

    @staticmethod
    def __get_string(label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        # change to ascii
        s = ''
        for l in ret:
            if l > 0 and l < (len(lstm_ocr_model.CONST_CHAR)+1):
                c = lstm_ocr_model.CONST_CHAR[l-1]
            else:
                c = ''
            s += c
        return s


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="Path to the CAPTCHA image file")
    # parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='ocr')
    # parser.add_argument("--epoch", help="Checkpoint epoch [Default 100]", type=int, default=100)
    # args = parser.parse_args()
    #
    # # Create array of zeros for LSTM init states
    # hp = Hyperparams()
    # init_states = lstm.init_states(batch_size=1, num_lstm_layer=hp.num_lstm_layer, num_hidden=hp.num_hidden)
    # init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
    # # Read the image into an ndarray
    # img = cv2.resize(cv2.imread(args.path, 0), (80, 30)).astype(np.float32) / 255
    # img = np.expand_dims(img.transpose(1, 0), 0)
    #
    # data_names = ['data'] + [s[0] for s in init_states]
    # sample = SimpleBatch(data_names, data=[mx.nd.array(img)] + init_state_arrays)
    #
    # sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    #
    # # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    # pred_fc = sym.get_internals()['pred_fc_output']
    # sym = mx.sym.softmax(data=pred_fc)
    #
    # mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
    # mod.bind(for_training=False, data_shapes=sample.provide_data)
    # mod.set_params(arg_params, aux_params, allow_missing=False)
    #
    # mod.forward(sample)
    # prob = mod.get_outputs()[0].asnumpy()
    #
    # label_list = list()
    # prediction = CtcMetrics.ctc_label(np.argmax(prob, axis=-1).tolist())
    # # Predictions are 1 to 10 for digits 0 to 9 respectively (prediction 0 means no-digit)
    # prediction = [p - 1 for p in prediction]
    # print("Digits:", prediction)
    # exit(0)
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("predict_lib_path", help="Path to directory containing mxnet_predict.so")
    args = parser.parse_args()

    sys.path.append(args.predict_lib_path + "/python")
    from mxnet_predict import Predictor

    _lstm_ocr_model = lstm_ocr_model('ocr-symbol.json', 'ocr-0010.params')
    img = cv2.imread('sample0.png', 0)
    _str = _lstm_ocr_model.forward_ocr(img)
    print('Result: ', _str)
