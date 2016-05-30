#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet_predict
import numpy as np
import time

symbol_file = '../checkpoint/lstm-symbol.json'
param_file = '../checkpoint/lstm-0000.params'
vocab_map = {}
for line in open('vocab_map'):
    w, idx = line.strip().split('')
    vocab_map[w] = int(idx)

# lstm_segment input data format: (batch_size, context_size)
batch_size = 8
context_size = 7
num_hidden = 300
input_shapes = { 'data':(batch_size, context_size), 
       'l0_init_c': (batch_size, num_hidden), 'l0_init_h': (batch_size, num_hidden)}

lstm_predict_handle = mxnet_predict.Predictor(open(symbol_file).read(), open(param_file).read(), input_shapes, dev_type='cpu')

input_str = '至于计算机的使用'

def reshape_input(s, context_size):
    padding_num = int((context_size - 1) / 2)
    unicode_str = unicode(s, 'utf-8')
    idx_sen = []
    for char in unicode_str:
        schar = char.encode('utf-8')
        if schar in vocab_map:
            idx_sen.append(vocab_map[schar])
        else:
            # unknown symbol
            idx_sen.append(vocab_map['U'])

    for _ in range(padding_num):
        idx_sen.insert(0, vocab_map['P'])
        idx_sen.append(vocab_map['P'])

    x = []
    for i in range(len(unicode_str)):
        x.append(idx_sen[i:i+context_size])

    return np.array(x)

init_c = np.zeros((batch_size, num_hidden))
init_h = np.zeros((batch_size, num_hidden))

x_data = reshape_input(input_str, context_size)
num_of_char = x_data.shape[0]

Idx2Label = {0:'B', 1: 'M', 2: 'E', 3: 'S'}

print input_str
input_data_dict = {'l0_init_c': init_c, 'l0_init_h': init_h, 'data': x_data}
start = time.time()
lstm_predict_handle.forward(**input_data_dict)
output = lstm_predict_handle.get_output(0)
print [ Idx2Label[x] for x in np.argmax(output, axis=1) ]
print 'elapsed %.2fs' % (time.time() - start)

# for i in range(num_of_char):
#     input_data_dict['data'] = x_data[i]
#     lstm_predict_handle.forward(**input_data_dict)
#     output = lstm_predict_handle.get_output(0)
#     print Idx2Label[np.argmax(output, axis=1)[0]]
