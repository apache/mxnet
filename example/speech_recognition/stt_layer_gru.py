from collections import namedtuple

import mxnet as mx

from stt_layer_batchnorm import batchnorm

GRUState = namedtuple("GRUState", ["h"])
GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
                                   "gates_h2h_weight", "gates_h2h_bias",
                                   "trans_i2h_weight", "trans_i2h_bias",
                                   "trans_h2h_weight", "trans_h2h_bias"])
GRUModel = namedtuple("GRUModel", ["rnn_exec", "symbol",
                                   "init_states", "last_states",
                                   "seq_data", "seq_labels", "seq_outputs",
                                   "param_blocks"])


def gru(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0., is_batchnorm=False, gamma=None, beta=None):
    """
    GRU Cell symbol
    Reference:
    * Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural
        networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).
    """
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden * 2,
                                name="t%d_l%d_gates_i2h" % (seqidx, layeridx))

    if is_batchnorm:
        i2h = batchnorm(net=i2h, gamma=gamma, beta=beta)
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.gates_h2h_weight,
                                bias=param.gates_h2h_bias,
                                num_hidden=num_hidden * 2,
                                name="t%d_l%d_gates_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=2,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    update_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    reset_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid")
    # The transform part of GRU is a little magic
    htrans_i2h = mx.sym.FullyConnected(data=indata,
                                       weight=param.trans_i2h_weight,
                                       bias=param.trans_i2h_bias,
                                       num_hidden=num_hidden,
                                       name="t%d_l%d_trans_i2h" % (seqidx, layeridx))
    h_after_reset = prev_state.h * reset_gate
    htrans_h2h = mx.sym.FullyConnected(data=h_after_reset,
                                       weight=param.trans_h2h_weight,
                                       bias=param.trans_h2h_bias,
                                       num_hidden=num_hidden,
                                       name="t%d_l%d_trans_i2h" % (seqidx, layeridx))
    h_trans = htrans_i2h + htrans_h2h
    h_trans_active = mx.sym.Activation(h_trans, act_type="tanh")
    next_h = prev_state.h + update_gate * (h_trans_active - prev_state.h)
    return GRUState(h=next_h)


def gru_unroll(net, num_gru_layer, seq_len, num_hidden_gru_list, dropout=0., is_batchnorm=False, prefix="",
               direction="forward"):
    if num_gru_layer > 0:
        param_cells = []
        last_states = []
        for i in range(num_gru_layer):
            param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable(prefix + "l%d_i2h_gates_weight" % i),
                                        gates_i2h_bias=mx.sym.Variable(prefix + "l%d_i2h_gates_bias" % i),
                                        gates_h2h_weight=mx.sym.Variable(prefix + "l%d_h2h_gates_weight" % i),
                                        gates_h2h_bias=mx.sym.Variable(prefix + "l%d_h2h_gates_bias" % i),
                                        trans_i2h_weight=mx.sym.Variable(prefix + "l%d_i2h_trans_weight" % i),
                                        trans_i2h_bias=mx.sym.Variable(prefix + "l%d_i2h_trans_bias" % i),
                                        trans_h2h_weight=mx.sym.Variable(prefix + "l%d_h2h_trans_weight" % i),
                                        trans_h2h_bias=mx.sym.Variable(prefix + "l%d_h2h_trans_bias" % i)))
            state = GRUState(h=mx.sym.Variable(prefix + "l%d_init_h" % i))
            last_states.append(state)
        assert (len(last_states) == num_gru_layer)
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

            # stack GRU
            for i in range(num_gru_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = dropout
                if is_batchnorm:
                    next_state = gru(num_hidden_gru_list[i], indata=hidden,
                                     prev_state=last_states[i],
                                     param=param_cells[i],
                                     seqidx=k, layeridx=i, dropout=dp_ratio,
                                     is_batchnorm=is_batchnorm,
                                     gamma=batchnorm_gamma[k],
                                     beta=batchnorm_beta[k])
                else:
                    next_state = gru(num_hidden_gru_list[i], indata=hidden,
                                     prev_state=last_states[i],
                                     param=param_cells[i],
                                     seqidx=k, layeridx=i, dropout=dp_ratio,
                                     is_batchnorm=is_batchnorm)
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


def bi_gru_unroll(net, num_gru_layer, seq_len, num_hidden_gru_list, dropout=0., is_batchnorm=False):
    if num_gru_layer > 0:
        net_forward = gru_unroll(net=net,
                                 num_gru_layer=num_gru_layer,
                                 seq_len=seq_len,
                                 num_hidden_gru_list=num_hidden_gru_list,
                                 dropout=dropout,
                                 is_batchnorm=is_batchnorm,
                                 prefix="forward_",
                                 direction="forward")
        net_backward = gru_unroll(net=net,
                                  num_gru_layer=num_gru_layer,
                                  seq_len=seq_len,
                                  num_hidden_gru_list=num_hidden_gru_list,
                                  dropout=dropout,
                                  is_batchnorm=is_batchnorm,
                                  prefix="backward_",
                                  direction="backward")
        hidden_all = []
        for i in range(seq_len):
            hidden_all.append(mx.sym.Concat(*[net_forward[i], net_backward[i]], dim=1))
        net = hidden_all
    return net


def bi_gru_unroll_two_input_two_output(net1, net2, num_gru_layer, seq_len, num_hidden_gru_list, dropout=0.,
                                       is_batchnorm=False):
    if num_gru_layer > 0:
        net_forward = gru_unroll(net=net1,
                                 num_gru_layer=num_gru_layer,
                                 seq_len=seq_len,
                                 num_hidden_gru_list=num_hidden_gru_list,
                                 dropout=dropout,
                                 is_batchnorm=is_batchnorm,
                                 prefix="forward_",
                                 direction="forward")
        net_backward = gru_unroll(net=net2,
                                  num_gru_layer=num_gru_layer,
                                  seq_len=seq_len,
                                  num_hidden_gru_list=num_hidden_gru_list,
                                  dropout=dropout,
                                  is_batchnorm=is_batchnorm,
                                  prefix="backward_",
                                  direction="backward")
        return net_forward, net_backward
    else:
        return net1, net2
