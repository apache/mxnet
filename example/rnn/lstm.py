# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        in_data = mx.sym.Dropout(data=in_data, p=dropout)
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
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_embed, num_label, dropout=0.):
    """unrolled lstm network"""
    # initialize the parameter symbols
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

    out_prob = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("t%d_data" % seqidx)
        label = mx.sym.Variable("t%d_label" % seqidx)
        hidden = mx.sym.FullyConnected(data=data, weight=embed_weight,
                                      num_hidden = num_embed, no_bias=True,
                                      name="t%d_embed" % seqidx)
        # stack LSTM
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=0.)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        fc = mx.sym.FullyConnected(data=hidden,
                                   weight=cls_weight,
                                   bias=cls_bias,
                                   num_hidden=num_label,
                                   name="t%d_cls" % seqidx)
        sm = mx.sym.Softmax(data=fc, label=label, name="t%d_sm" % seqidx)
        out_prob.append(sm)

    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                          h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")


def setup_rnn_model(ctx,
                    num_lstm_layer, seq_len,
                    num_hidden, num_embed, num_label,
                    batch_size, input_size,
                    initializer, dropout=0.):
    """set up rnn model with lstm cells"""
    rnn_sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                          num_hidden=num_hidden,
                          seq_len=seq_len,
                          num_embed=num_embed,
                          num_label=num_label,
                          dropout)
    print(rnn_sym.list_outputs())
    arg_names = rnn_sym.list_arguments()
    print sorted(arg_names)

    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, input_size)
        else:
            print("ignore %s " % name)

    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if is_param_name(name):
            args_grad[name] = mx.nd.zeros(shape, ctx)
        # else:
            # print("Do not need gradient for %s" % name)

    rnn_exec = rnn_sym.bind(ctx=ctx, args=arg_arrays,
                            args_grad=args_grad,
                            grad_req="add")
    param_blocks = []
    arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])

            param_blocks.append((i, arg_dict[name], args_grad[name], name))
        else:
            assert name not in args_grad
    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))

    init_states = [LSTMState(c=arg_dict["l%d_init_c" % i],
                             h=arg_dict["l%d_init_h" % i]) for i in range(num_lstm_layer)]
    seq_labels = [rnn_exec.arg_dict["t%d_label" % i] for i in range(seq_len)]
    seq_data = [rnn_exec.arg_dict["t%d_data" % i] for i in range(seq_len)]
    last_states = [LSTMState(c=out_dict["l%d_last_c_output" % i],
                             h=out_dict["l%d_last_h_output" % i]) for i in range(num_lstm_layer)]
    seq_outputs = [out_dict["t%d_sm_output" % i] for i in range(seq_len)]

    return LSTMModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                     init_states=init_states, last_states=last_states,
                     seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                     param_blocks=param_blocks)


def set_onehot_input(onehot, xidx):
    """setup onehot input"""
    onehot[:] = 0.
    onehot[np.arange(onehot.shape[0]), xidx.astype("int32")] = 1.

def logloss(y, prob):
    eps = 1e-10
    assert prob.shape[0] == len(y)
    py = prob[np.arange(len(y)), y.astype("int32")]
    return -np.sum(np.log(np.maximum(py, eps))) / len(y)

def set_rnn_inputs(m, X, onehot, begin):
    seq_len = len(m.seq_data)
    batch_size, vocab = onehot.shape
    for seqidx in range(seq_len):
        idx = (begin + seqidx) % X.shape[0]
        next_idx = (begin + seqidx + 1) % X.shape[0]
        x = X[idx, :]
        y = X[next_idx, :]
        onehot[:] = 0.
        onehot[np.arange(batch_size), x.astype("int32")] = 1.
        m.seq_data[seqidx][:] = onehot
        m.seq_labels[seqidx][:] = y

def calc_nll(seq_out, X, begin):
    nll = 0.
    for seqidx in range(len(seq_out)):
        next_idx = (begin + seqidx + 1) % X.shape[0]
        y = X[next_idx, :]
        nll += logloss(y, seq_out[seqidx].asnumpy())
    return nll

def train_lstm(model, X_train_batch, X_val_batch,
               num_round, update_period,
               optimizer='sgd', half_life=2, **kwargs):
    print("Training swith train.shape=%s" % str(X_train_batch.shape))
    print("Training swith val.shape=%s" % str(X_val_batch.shape))
    m = model
    onehot = np.zeros(m.seq_data[0].shape, dtype='float32')
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    print("batch_size=%d" % batch_size)
    print("seq_len=%d" % seq_len)
    rescale_grad = 1.0 / (seq_len * batch_size * update_period)
    opt = mx.optimizer.create(optimizer,
                              rescale_grad=rescale_grad,
                              **kwargs)
    updater = mx.optimizer.get_updater(opt)
    epoch_counter = 0
    watch_weight = False
    log_period = max(1000 / seq_len, 1)

    for iteration in range(num_round):
        nbatch = 0
        train_nll = 0
        # reset states
        for state in m.init_states:
            state.c[:] = 0.0
            state.h[:] = 0.0
        tic = time.time()
        assert X_train_batch.shape[0] % seq_len == 0
        assert X_val_batch.shape[0] % seq_len == 0
        for begin in range(0, X_train_batch.shape[0], seq_len):
            set_rnn_inputs(m, X_train_batch, onehot, begin=begin)
            m.rnn_exec.forward(is_train=True)
            seq_outs = [out.copyto(mx.cpu()) for out in m.seq_outputs]
            m.rnn_exec.backward()
            # transfer the states
            for init, last in zip(m.init_states, m.last_states):
                last.c.copyto(init.c)
                last.h.copyto(init.h)
            # update epoch counter
            epoch_counter += 1
            if epoch_counter % update_period == 0:
                # TODO add gradient clip here
                # updare parameters
                for idx, weight, grad, name in m.param_blocks:
                    if epoch_counter % log_period == 0 and watch_weight:
                        dw = grad.asnumpy()
                        w = weight.asnumpy()
                        dwnorm = np.linalg.norm(dw, 2) * rescale_grad
                        wnorm = np.linalg.norm(w, 2)
                        print("dw:norm(%s): %.3f" % (name, dwnorm))
                        print("w:norm(%s): %.3f" % (name, wnorm))
                        if name == "cls_bias":
                            print len(dw[dw<0])
                    updater(idx, grad, weight)
                    # reset gradient to zero
                    grad[:] = 0.0
            train_nll += calc_nll(seq_outs, X_train_batch, begin=begin)

            nbatch = begin + seq_len
            if epoch_counter % log_period == 0:
                print("Epoch [%d] Train: NLL=%.3f, Prep=%.3f" % (
                    epoch_counter, train_nll / nbatch, np.exp(train_nll / nbatch)))
        # end of training loop
        toc = time.time()
        print("Iter [%d] Train: Time: %.3f sec, NLL=%.3f, Prep=%.3f" % (
            iteration, toc - tic, train_nll / nbatch, np.exp(train_nll / nbatch)))

        val_nll = 0.0
        # validation set, reset states
        for state in m.init_states:
            state.c[:] = 0.0
            state.h[:] = 0.0
        for begin in range(0, X_val_batch.shape[0], seq_len):
            set_rnn_inputs(m, X_val_batch, onehot, begin=begin)
            m.rnn_exec.forward(is_train=False)
            seq_outs = [out.copyto(mx.cpu()) for out in m.seq_outputs]
            # transfer the states
            for init, last in zip(m.init_states, m.last_states):
                last.c.copyto(init.c)
                last.h.copyto(init.h)
            val_nll += calc_nll(seq_outs, X_val_batch, begin=begin)
        nbatch = X_val_batch.shape[0]
        print("Iter [%d] Val: NLL=%.3f, Prep=%.3f" % (
            iteration, val_nll / nbatch, np.exp(val_nll / nbatch)))
        if (iteration + 1) % half_life == 0:
            opt.lr *= 0.9
            print("Reset learning rate to %g" % opt.lr)
