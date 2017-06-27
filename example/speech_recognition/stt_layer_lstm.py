# pylint:skip-file
from collections import namedtuple

import mxnet as mx

from stt_layer_batchnorm import batchnorm

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias",
                                     "ph2h_weight",
                                     "c2i_bias", "c2f_bias", "c2o_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


def vanilla_lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, is_batchnorm=False, gamma=None, beta=None):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    if is_batchnorm:
        i2h = batchnorm(net=i2h, gamma=gamma, beta=beta)
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0., num_hidden_proj=0, is_batchnorm=False,
         gamma=None, beta=None):
    """LSTM Cell symbol"""
    # dropout input
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    if is_batchnorm:
        i2h = batchnorm(net=i2h, gamma=gamma, beta=beta)

    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                # bias=param.h2h_bias,
                                no_bias=True,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))

    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))

    Wcidc = mx.sym.broadcast_mul(param.c2i_bias, prev_state.c) + slice_gates[0]
    in_gate = mx.sym.Activation(Wcidc, act_type="sigmoid")

    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")

    Wcfdc = mx.sym.broadcast_mul(param.c2f_bias, prev_state.c) + slice_gates[2]
    forget_gate = mx.sym.Activation(Wcfdc, act_type="sigmoid")

    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)

    Wcoct = mx.sym.broadcast_mul(param.c2o_bias, next_c) + slice_gates[3]
    out_gate = mx.sym.Activation(Wcoct, act_type="sigmoid")

    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    if num_hidden_proj > 0:
        proj_next_h = mx.sym.FullyConnected(data=next_h,
                                            weight=param.ph2h_weight,
                                            no_bias=True,
                                            num_hidden=num_hidden_proj,
                                            name="t%d_l%d_ph2h" % (seqidx, layeridx))

        return LSTMState(c=next_c, h=proj_next_h)
    else:
        return LSTMState(c=next_c, h=next_h)


def lstm_unroll(net, num_lstm_layer, seq_len, num_hidden_lstm_list, dropout=0., num_hidden_proj=0,
                lstm_type='fc_lstm', is_batchnorm=False, prefix="", direction="forward"):
    if num_lstm_layer > 0:
        param_cells = []
        last_states = []
        for i in range(num_lstm_layer):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix + "l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable(prefix + "l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable(prefix + "l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable(prefix + "l%d_h2h_bias" % i),
                                         ph2h_weight=mx.sym.Variable(prefix + "l%d_ph2h_weight" % i),
                                         c2i_bias=mx.sym.Variable(prefix + "l%d_c2i_bias" % i,
                                                                  shape=(1, num_hidden_lstm_list[i])),
                                         c2f_bias=mx.sym.Variable(prefix + "l%d_c2f_bias" % i,
                                                                  shape=(1, num_hidden_lstm_list[i])),
                                         c2o_bias=mx.sym.Variable(prefix + "l%d_c2o_bias" % i,
                                                                  shape=(1, num_hidden_lstm_list[i]))
                                         ))
            state = LSTMState(c=mx.sym.Variable(prefix + "l%d_init_c" % i),
                              h=mx.sym.Variable(prefix + "l%d_init_h" % i))
            last_states.append(state)
        assert (len(last_states) == num_lstm_layer)
        # declare batchnorm param(gamma,beta) in timestep wise
        if is_batchnorm:
            batchnorm_gamma = []
            batchnorm_beta = []
            for seqidx in range(seq_len):
                batchnorm_gamma.append(mx.sym.Variable(prefix + "t%d_i2h_gamma" % seqidx))
                batchnorm_beta.append(mx.sym.Variable(prefix + "t%d_i2h_beta" % seqidx))

        hidden_all = []
        for seqidx in range(seq_len):
            if direction == "forward":
                k = seqidx
                hidden = net[k]
            elif direction == "backward":
                k = seq_len - seqidx - 1
                hidden = net[k]
            else:
                raise Exception("direction should be whether forward or backward")

            # stack LSTM
            for i in range(num_lstm_layer):
                if i == 0:
                    dp = 0.
                else:
                    dp = dropout

                if lstm_type == 'fc_lstm':
                    if is_batchnorm:
                        next_state = lstm(num_hidden_lstm_list[i],
                                          indata=hidden,
                                          prev_state=last_states[i],
                                          param=param_cells[i],
                                          seqidx=k,
                                          layeridx=i,
                                          dropout=dp,
                                          num_hidden_proj=num_hidden_proj,
                                          is_batchnorm=is_batchnorm,
                                          gamma=batchnorm_gamma[k],
                                          beta=batchnorm_beta[k]
                                          )
                    else:
                        next_state = lstm(num_hidden_lstm_list[i],
                                          indata=hidden,
                                          prev_state=last_states[i],
                                          param=param_cells[i],
                                          seqidx=k,
                                          layeridx=i,
                                          dropout=dp,
                                          num_hidden_proj=num_hidden_proj,
                                          is_batchnorm=is_batchnorm
                                          )
                elif lstm_type == 'vanilla_lstm':
                    if is_batchnorm:
                        next_state = vanilla_lstm(num_hidden_lstm_list[i], indata=hidden,
                                                  prev_state=last_states[i],
                                                  param=param_cells[i],
                                                  seqidx=k, layeridx=i,
                                                  is_batchnorm=is_batchnorm,
                                                  gamma=batchnorm_gamma[k],
                                                  beta=batchnorm_beta[k]
                                                  )
                    else:
                        next_state = vanilla_lstm(num_hidden_lstm_list[i], indata=hidden,
                                                  prev_state=last_states[i],
                                                  param=param_cells[i],
                                                  seqidx=k, layeridx=i,
                                                  is_batchnorm=is_batchnorm
                                                  )
                else:
                    raise Exception("lstm type %s error" % lstm_type)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)

            if direction == "forward":
                hidden_all.append(hidden)
            elif direction == "backward":
                hidden_all.insert(0, hidden)
            else:
                raise Exception("direction should be whether forward or backward")
        net = hidden_all

    return net


def bi_lstm_unroll(net, num_lstm_layer, seq_len, num_hidden_lstm_list, dropout=0., num_hidden_proj=0,
                   lstm_type='fc_lstm', is_batchnorm=False):
    if num_lstm_layer > 0:
        net_forward = lstm_unroll(net=net,
                                  num_lstm_layer=num_lstm_layer,
                                  seq_len=seq_len,
                                  num_hidden_lstm_list=num_hidden_lstm_list,
                                  dropout=dropout,
                                  num_hidden_proj=num_hidden_proj,
                                  lstm_type=lstm_type,
                                  is_batchnorm=is_batchnorm,
                                  prefix="forward_",
                                  direction="forward")

        net_backward = lstm_unroll(net=net,
                                   num_lstm_layer=num_lstm_layer,
                                   seq_len=seq_len,
                                   num_hidden_lstm_list=num_hidden_lstm_list,
                                   dropout=dropout,
                                   num_hidden_proj=num_hidden_proj,
                                   lstm_type=lstm_type,
                                   is_batchnorm=is_batchnorm,
                                   prefix="backward_",
                                   direction="backward")
        hidden_all = []
        for i in range(seq_len):
            hidden_all.append(mx.sym.Concat(*[net_forward[i], net_backward[i]], dim=1))
        net = hidden_all
    return net


# bilistm_2to1
def bi_lstm_unroll_two_input_two_output(net1, net2, num_lstm_layer, seq_len, num_hidden_lstm_list, dropout=0.,
                                        num_hidden_proj=0,
                                        lstm_type='fc_lstm', is_batchnorm=False):
    if num_lstm_layer > 0:
        net_forward = lstm_unroll(net=net1,
                                  num_lstm_layer=num_lstm_layer,
                                  seq_len=seq_len,
                                  num_hidden_lstm_list=num_hidden_lstm_list,
                                  dropout=dropout,
                                  num_hidden_proj=num_hidden_proj,
                                  lstm_type=lstm_type,
                                  is_batchnorm=is_batchnorm,
                                  prefix="forward_",
                                  direction="forward")

        net_backward = lstm_unroll(net=net2,
                                   num_lstm_layer=num_lstm_layer,
                                   seq_len=seq_len,
                                   num_hidden_lstm_list=num_hidden_lstm_list,
                                   dropout=dropout,
                                   num_hidden_proj=num_hidden_proj,
                                   lstm_type=lstm_type,
                                   is_batchnorm=is_batchnorm,
                                   prefix="backward_",
                                   direction="backward")
        return net_forward, net_backward
    else:
        return net1, net2
