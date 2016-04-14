# -*- coding:utf-8 -*-
# @author: Yuanqin Lu

import mxnet as mx
import numpy as np
from collections import namedtuple
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])

#TODO: ndarray expander
class NDArrayExpander(mx.operator.NDArrayOp):
    def __init__(self, n):
        super(NDArrayExpander, self).__init__(True)
        self.fwd_kernel = None
        self.bwd_kernel = None
        self.n = n

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        assert(len(data_shape) == 2), 'feature shape should be 2D'
        output_shape = (self.n*data_shape[0], data_shape[1])
        return [data_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('expander', [('x', x), ('y', y)], """
int n = y_dims[0] / x_dims[0];
int i = blockIdx.x;
int j = threadIdx.x;
for (int k = 0; k < n; ++k) {
    y[(k+i*n)*x_dims[1]+j] = x[i*x_dims[1]+j];
}
            """)
        self.fwd_kernel.push([x], [y], (y.shape[0], 1, 1), (y.shape[1], 1, 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        dy = out_grad[0]
        dx = in_grad[0]
        if self.bwd_kernel is None:
            self.bwd_kernel = mx.rtc('expander_grad', [('dy', dy)], [('dx', dx)], """
int n = y_dims[0] / dx_dims[0];
int i = blockIdx.x;
int j = threadIdx.x;
for (int k = 0; k < n; ++k) {
    dx[i*dx_dims[1]+j] += dy[(k+i*n)*dx_dims[1]+j];
}
dx[i*dx_dims[1]+j] /= static_cast<float>(n);
            """)
        self.bwd_kernel.push([dy], [dx], (dy.shape[0], 1, 1), (dy.shape[0], 1, 1))


class NumpyExpander(mx.operator.NumpyOp):

    """Docstring for NumpyExpander. """

    def __init__(self, n):
        super(NumpyExpander, self).__init__(True)
        self.n = n

    def list_arguments(self):
        return ['data']

    def lsit_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        assert(len(data_shape) == 2), 'feature should be 2D'
        output_shape = (self.n * data_shape[0], data_shape[1])
        return [data_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        for i in range(x.shape[0]):
            y[self.n*i:self.n*(i+1)] = x[i]

    def backward(self, out_grad, in_data, out_data, in_grad):
        dy = out_grad[0]
        dx = in_grad[0]
        for i in range(dx.shape[0]):
            dx[i] = np.mean(dy[self.n*i:self.n*(i+1)], axis=0)





def vgg16_fc7_symbol(input_name, num_embed):

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
    relu7   = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    """
    drop7   = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")

    # fc8
    fc8     = mx.sym.FullyConnected(data=drop7, num_hidden=1000, name="fc8")
    softmax = mx.sym.Softmax(data=fc8, name="softmax")
    """

    # embedding
    embedding = mx.sym.FullyConnected(data=relu7, num_hidden=num_embed, name="t0_embed")
    embed_relu = mx.sym.Activation(data=embedding, act_type="relu", name="embed_relu")

    return embed_relu


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
                   num_embed, num_seq, dropout=0.):
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
    seq = mx.sym.Variable('seq')
    label = mx.sym.Variable('label')
    embed = mx.sym.Embedding(data=seq, input_dim=vocab_size, output_dim=num_embed,
                             name='seq_embed')
    print seq_len
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, axis=1)

    hidden_all = []
    for seqidx in range(seq_len+1):
        if seqidx == 0:
            data = vgg16_fc7_symbol('image', num_embed)
            expander = NumpyExpander(num_seq)
            hidden = expander(data=data, name="expand")
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
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=vocab_size, name='pred')

    label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    label = [label_slice[t] for t in range(seq_len)]
    label = mx.sym.Concat(*label, dim=0)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm




def lstm_inference_symbol(num_lstm_layer, vocab_size,
                          num_hidden, num_embed, dropout=0.):
    seqidx = 1
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data/%d" % seqidx)

    hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                              input_dim=vocab_size,
                              output_dim=num_embed,
                              name="t%d_embed" % seqidx)
    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp=0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias,
                               num_hidden=vocab_size)
    sm = mx.sym.SoftmaxOutput(data=fc, label=mx.sym.Variable('label/%d' % seqidx),
                              name='t%d_sm' % seqidx)
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)


def cnn_lstm_symbol(num_lstm_layer, num_hidden, num_embed, dropout=0.):
    # Init states
    param_cells = []
    init_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        init_states.append(state)

    # CNN and one timestep LSTM
    cnn = vgg16_fc7_symbol("data/0", num_embed)
    for i in range(num_lstm_layer):
        if i == 0:
            dp = 0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=cnn, prev_state=init_states[i],
                          param=param_cells[i], seqidx=0, layeridx=i, dropout=dp)
        init_states[i] = next_state

    output = []
    for state in init_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)



class InferenceModel(object):
    def __init__(self, num_lstm_layer, vocab_size, num_hidden, num_embed,
                 arg_params, ctx=mx.cpu(), dropout=0.):
        self.cnn_lstm_sym = cnn_lstm_symbol(num_lstm_layer, num_hidden, num_embed, dropout)
        self.lm_sym  = lstm_inference_symbol(num_lstm_layer, vocab_size, num_hidden,
                                             num_embed, dropout)
        batch_size = 1
        init_c = [('l%d_init_c' %l, (batch_size, num_hidden))
                        for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h' %l, (batch_size, num_hidden))
                        for l in range(num_lstm_layer)]
        img_shape = [('data/0', (batch_size, 3, 224, 224))]
        seq_shape = [('data/1', (batch_size,))]

        img_input_shape = dict(init_c + init_h + img_shape)
        seq_input_shape = dict(init_c + init_h + seq_shape)

        self.cnn_exec = self.cnn_lstm_sym.simple_bind(ctx=ctx, **img_input_shape)
        self.lm_exec  = self.lm_sym.simple_bind(ctx=ctx, **seq_input_shape)

        # init params
        for key in self.cnn_exec.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.cnn_exec.arg_dict[key])
        for key in self.lm_exec.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.lm_exec.arg_dict[key])

        state_name = []
        for i in range(num_lstm_layer):
            state_name.append('l%d_init_c' %i)
            state_name.append('l%d_init_h' %i)

        self.init_states_dict = dict(zip(state_name, self.cnn_exec.outputs))
        self.states_dict = dict(zip(state_name, self.lm_exec.outputs[1:]))
        self.word_array = mx.nd.zeros((batch_size, ), ctx=ctx)

    def generate(self, img, max_seq_length=15):
        """

        :param img: mx.nd.array
        :param max_seq_length: int
        """
        # initialize c, h and word
        self.word_array[:] = 0.
        for key in self.states_dict.keys():
            self.cnn_exec.arg_dict[key] = 0.
        img.copyto(self.cnn_exec.arg_dict['data/0'])
        self.cnn_exec.forward()
        probs = []
        index = []

        for i in range(max_seq_length):
            if i == 0:
                for key in self.init_states_dict.keys():
                    self.init_states_dict[key].copyto(self.lm_exec.arg_dict[key])
            self.word_array.copyto(self.lm_exec.arg_dict['data/1'])
            self.lm_exec.forward()
            for key in self.states_dict.keys():
                self.states_dict[key].copyto(self.lm_exec.arg_dict[key])
            prob = self.lm_exec.outputs[0].asnumpy()
            probs.append(prob)
            idx = np.argmax(prob, axis=1)
            assert(len(idx.shape) == 1)
            index.append(idx)
            if idx.any():
                self.word_array[:] = idx
            else:
                break
        return probs, index



