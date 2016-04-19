# -*- coding:utf-8 -*-
# @author: Yuanqin Lu

import mxnet as mx
import numpy as np
from collections import namedtuple
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])




def vgg16_fc7_symbol(input_name):
    data = mx.sym.Variable(name=input_name)

    # conv1
    conv1_1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1   = mx.sym.Pooling(data=relu1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")

    # conv2
    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2   = mx.sym.Pooling(data=relu2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")

    # conv3
    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3   = mx.sym.Pooling(data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")

    # conv4
    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4   = mx.sym.Pooling(data=relu4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")

    # conv5
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5   = mx.sym.Pooling(data=relu5_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool5")

    # fc6
    flat6     = mx.sym.Flatten(data=pool5, name="flat6")
    fc6     = mx.sym.FullyConnected(data=flat6, num_hidden=4096, name="fc6")
    relu6   = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6   = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")

    # fc7
    fc7     = mx.sym.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    return fc7


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """

    :param num_hidden:
    :param indata:
    :param prev_state:
    :param param:
    :param seqidx:
    :param layeridx:
    :param dropout:
    :return:
    """
    if dropout > 0:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name ="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    return LSTMState(c=next_c, h=next_h)


def network_unroll(num_lstm_layer, seq_len, vocab_size, num_hidden,
                   num_embed, dropout=0.):
    """

    :param num_lstm_layer:
    :param seq_len:
    :param vocab_size:
    :param num_hidden:
    :param num_embed:
    :param dropout:
    :return:
    """
    # init lstm param variable and state variable
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" %i),
                                      i2h_bias=mx.sym.Variable("l%d_i2h_bias" %i),
                                      h2h_weight=mx.sym.Variable("l%d_h2h_weight" %i),
                                      h2h_bias=mx.sym.Variable("l%d_h2h_bias" %i)))
        last_states.append(LSTMState(c=mx.sym.Variable("l%d_init_c" %i),
                                     h=mx.sym.Variable("l%d_init_h" %i)))
    assert(len(last_states) == num_lstm_layer)

    # seq embedding layer
    seq = mx.sym.Variable('caption')
    label = mx.sym.Variable('label')
    embed = mx.sym.Embedding(data=seq, input_dim=vocab_size+2, output_dim=num_embed,
                             name='seq_embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len+1):
        if seqidx == 0:
            image_feature = mx.sym.Variable('image_feature')
            hidden = mx.sym.FullyConnected(data=image_feature, num_hidden=num_embed, name='img_embed')
        else:
            hidden = wordvec[seqidx-1]
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i,
                              dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if seqidx > 0:
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=vocab_size+2,
                                 name='pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm

