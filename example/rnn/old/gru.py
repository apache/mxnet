# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
from collections import namedtuple
GRUState = namedtuple("GRUState", ["h"])
GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
                                   "gates_h2h_weight", "gates_h2h_bias",
                                   "trans_i2h_weight", "trans_i2h_bias",
                                   "trans_h2h_weight", "trans_h2h_bias"])
GRUModel = namedtuple("GRUModel", ["rnn_exec", "symbol",
                                   "init_states", "last_states",
                                   "seq_data", "seq_labels", "seq_outputs",
                                   "param_blocks"])

def gru(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
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

def gru_unroll(num_gru_layer, seq_len, input_size,
               num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_gru_layer):
        param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable("l%d_i2h_gates_weight" % i),
                                    gates_i2h_bias=mx.sym.Variable("l%d_i2h_gates_bias" % i),
                                    gates_h2h_weight=mx.sym.Variable("l%d_h2h_gates_weight" % i),
                                    gates_h2h_bias=mx.sym.Variable("l%d_h2h_gates_bias" % i),
                                    trans_i2h_weight=mx.sym.Variable("l%d_i2h_trans_weight" % i),
                                    trans_i2h_bias=mx.sym.Variable("l%d_i2h_trans_bias" % i),
                                    trans_h2h_weight=mx.sym.Variable("l%d_h2h_trans_weight" % i),
                                    trans_h2h_bias=mx.sym.Variable("l%d_h2h_trans_bias" % i)))
        state = GRUState(h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_gru_layer)
    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

        # stack GRU
        for i in range(num_gru_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = gru(num_hidden, indata=hidden,
                             prev_state=last_states[i],
                             param=param_cells[i],
                             seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')
    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))
    return mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

