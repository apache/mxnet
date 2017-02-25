#!/usr/bin/env python2.7
# coding=utf-8
from __future__ import print_function
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("../../amalgamation/python/")
sys.path.append("../../python/")

from mxnet_predict import Predictor
import mxnet as mx

import numpy as np
import cv2
import os

class lstm_ocr_model(object):
    # Keep Zero index for blank. (CTC request it)
    CONST_CHAR='0123456789'
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
        self.init_state_dict={}
        for x in init_states:
            self.init_state_dict[x[0]] = init_state_arrays

        all_shapes = [('data', (batch_size, 80 * 30))] + init_states + [('label', (batch_size, num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json).read(),
                                    open(self.path_of_params).read(),
                                    all_shapes_dict)

    def forward_ocr(self, img):
        img = cv2.resize(img, (80, 30))
        img = img.transpose(1, 0)
        img = img.reshape((80 * 30))
        img = np.multiply(img, 1/255.0)
        self.predictor.forward(data=img, **self.init_state_dict)
        prob = self.predictor.get_output(0)
        label_list = []
        for p in prob:
            max_index = np.argsort(p)[::-1][0]
            label_list.append(max_index)
        return self.__get_string(label_list)

    def __get_string(self, label_list):
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
    _lstm_ocr_model = lstm_ocr_model('ocr-symbol.json', 'ocr-0010.params')
    img = cv2.imread('sample.jpg', 0)
    _str = _lstm_ocr_model.forward_ocr(img)
    print('Result: ', _str)

