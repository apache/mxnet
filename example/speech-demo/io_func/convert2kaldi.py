# Copyright 2013    Yajie Miao    Carnegie Mellon University 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys

from StringIO import StringIO
import json
import utils.utils as utils
from model_io import string_2_array

# Various functions to convert models into Kaldi formats
def _nnet2kaldi(nnet_spec, set_layer_num = -1, filein='nnet.in',
               fileout='nnet.out', activation='sigmoid', withfinal=True):
    _nnet2kaldi_main(nnet_spec, set_layer_num=set_layer_num, filein=filein,
                    fileout=fileout, activation=activation, withfinal=withfinal, maxout=False)

def _nnet2kaldi_maxout(nnet_spec, pool_size = 1, set_layer_num = -1, 
                      filein='nnet.in', fileout='nnet.out', activation='sigmoid', withfinal=True):
    _nnet2kaldi_main(nnet_spec, set_layer_num=set_layer_num, filein=filein,
                    fileout=fileout, activation=activation, withfinal=withfinal,
                    pool_size = 1, maxout=True)

def _nnet2kaldi_main(nnet_spec, set_layer_num = -1, filein='nnet.in',
               fileout='nnet.out', activation='sigmoid', withfinal=True, maxout=False):
    elements = nnet_spec.split(':')
    layers = []
    for x in elements:
        layers.append(int(x))
    if set_layer_num == -1:
        layer_num = len(layers) - 1
    else:
        layer_num = set_layer_num + 1
    nnet_dict = {}
    nnet_dict = utils.pickle_load(filein)

    fout = open(fileout, 'wb')
    for i in xrange(layer_num - 1):
        input_size = int(layers[i])
        if maxout:
            output_size = int(layers[i + 1]) * pool_size
        else:
            output_size = int(layers[i + 1])
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = str(i) + ' ' + activation + ' W'
        matrix = string_2_array(nnet_dict[dict_key])

        for x in xrange(input_size):
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(matrix[x][t]) + ' '

        dict_key = str(i) + ' ' + activation + ' b'
        vector = string_2_array(nnet_dict[dict_key])
        for x in xrange(output_size):
            b_layer = b_layer + str(vector[x]) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        if maxout:
            fout.write('<maxout> ' + str(int(layers[i + 1])) + ' ' + str(output_size) + '\n')
        else:
            fout.write('<sigmoid> ' + str(output_size) + ' ' + str(output_size) + '\n')

    if withfinal:
        input_size = int(layers[-2])
        output_size = int(layers[-1])
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = 'logreg W'
        matrix = string_2_array(nnet_dict[dict_key])
        for x in xrange(input_size):
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(matrix[x][t]) + ' '


        dict_key = 'logreg b'
        vector = string_2_array(nnet_dict[dict_key])
        for x in xrange(output_size):
            b_layer = b_layer + str(vector[x]) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        fout.write('<softmax> ' + str(output_size) + ' ' + str(output_size) + '\n')

    fout.close();