#!/usr/bin/env python

import sys
import mxnet as mx
import numpy as np
import time
import math
from collections import namedtuple

logs = sys.stderr

LSTMState = namedtuple("LSTMState", ['c', 'h'])
LSTMParam = namedtuple('LSTMParam', ['i2h_weight', 'i2h_bias', 'h2h_weight', 'h2h_bias'])
LSTMModel = namedtuple('LSTMModel', ['lstm_exec', 'symbol', 'init_states', 'last_states', 'seq_data', 'seq_labels', 'seq_outputs', 'param_blocks'])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout):
    """LSTM Memory Unit"""
    i2h = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight, bias=param.i2h_bias,
                                num_hidden=num_hidden * 4, name='t%d_l%d_i2h' % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight, bias=param.h2h_bias,
                                num_hidden=num_hidden * 4, name='t%d_l%d_h2h' % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, name='t%d_l%d_slice' % (seqidx, layeridx))

    # input gate
    input_gate = mx.sym.Activation(slice_gates[0], act_type='sigmoid')
    input_transform = mx.sym.Activation(slice_gates[1], act_type='tanh')
    # forget gate
    forget_gate = mx.sym.Activation(slice_gates[2], act_type='sigmoid')
    # output gate
    output_gate = mx.sym.Activation(slice_gates[3], act_type='sigmoid')
    next_c = (forget_gate * prev_state.c) + (input_gate * input_transform)
    next_h = output_gate * mx.sym.Activation(next_c, act_type='tanh')

    return LSTMState(c=next_c, h=next_h)


def unroll_lstm(num_lstm_layer, num_hidden, step_size, context_size, vocab_size, num_embed, num_label, dropout=0.):
    # initialize the parameter sysmbols
    embed_weight = mx.sym.Variable('embed_weight')
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
        state = LSTMState(c=mx.sym.Variable('l%d_init_c' % i), h=mx.sym.Variable('l%d_init_h' % i))
        last_states.append(state)

    # embedding layer
    # data = mx.sym.Variable('data')
    # label = mx.sym.Variable('label')
    # embed = mx.sym.Embedding(data=data, weight=embed_weight,
    #         input_dim=vocab_size, output_dim=num_embed, name='embed')
    # wordvec = mx.sym.SliceChannel(data=embed, num_outputs=context_size, squeeze_axis=1)
    last_hidden = []
    for seqidx in range(step_size):
        # embedding layer
        data = mx.sym.Variable("t%d_data" % seqidx)
        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                input_dim=vocab_size, output_dim=num_embed, name='t%d_embed' % seqidx)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp = 0.
            else:
                dp = dropout
            next_state = lstm(num_hidden, indata=hidden, prev_state=last_states[i],
                              param=param_cells[i], seqidx=seqidx, layeridx=i, dropout=dropout)
            hidden = next_state.h
            last_states[i] = next_state

        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)

    out_prob = []
    for seqidx in range(step_size):
        fc = mx.sym.FullyConnected(data=last_hidden[seqidx], weight=cls_weight,
                bias=cls_bias, num_hidden=num_label, name='t%d_cls' % seqidx)
        label = mx.sym.Variable('t%d_label' % seqidx)
        sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='t%d_sm' % seqidx)
        out_prob.append(sm)

    # concat = mx.sym.Concat(*last_hidden, dim=0)
    # fc = mx.sym.FullyConnected(data=concat, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    # label = mx.sym.Variable("label")
    # sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='sm')

    # hidden_concat = mx.sym.Concat(*last_hidden, dim=0)
    # use last hidden h as feature
    # fc = mx.sym.FullyConnected(data=last_hidden[-1], weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    # sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='sm')

    # out_prob = [sm]

    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name='l%d_last_c' % i),
                          h=mx.sym.BlockGrad(state.h, name='l%d_last_h' % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith('weight') or name.endswith('bias') or \
        name.endswith('gamma') or name.endswith('beta')

def setup_lstm_model(ctx, num_lstm_layer, step_size, context_size, num_hidden, num_embed, num_label,
        batch_size, vocab_size, initializer, dropout=0.):

    lstm_sym = unroll_lstm(num_lstm_layer=num_lstm_layer, num_hidden=num_hidden, step_size=step_size,
                           context_size=context_size, vocab_size=vocab_size,
                           num_embed=num_embed, num_label=num_label, dropout=dropout)

    arg_names = lstm_sym.list_arguments()

    input_shapes = {}
    for name in arg_names:
        if name.endswith('init_c') or name.endswith('init_h'):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith('data'):
            input_shapes[name] = (batch_size, context_size)
        elif name == 'label':
            input_shapes[name] = (batch_size * step_size, )
        elif name.endswith('label'):
            input_shapes[name] = (batch_size,)
        else:
            pass

    arg_shape, out_shape, aux_shape = lstm_sym.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name, in zip(arg_shape, arg_names):
        if is_param_name(name):
            print >> logs, 'parameter argument', name, shape
            args_grad[name] = mx.nd.zeros(shape, ctx)
        else:
            print >> logs, 'input argument', name, shape

    lstm_exec = lstm_sym.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, lstm_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])
            param_blocks.append( (i, arg_dict[name], args_grad[name], name) )
        else:
            assert name not in args_grad

    out_dict = dict(zip(lstm_sym.list_outputs(), lstm_exec.outputs))

    init_states = [LSTMState(c=arg_dict['l%d_init_c' % i],
                             h=arg_dict['l%d_init_h' % i]) for i in range(num_lstm_layer)]
    seq_data = [arg_dict['t%d_data' % i] for i in range(step_size)]
    last_states = [LSTMState(c=out_dict['l%d_last_c_output' % i],
                             h=out_dict['l%d_last_h_output' % i]) for i in range(num_lstm_layer)]
    seq_outputs = [out_dict['t%d_sm_output' % i] for i in range(step_size)]
    seq_labels = [arg_dict['t%d_label' % i] for i in range(step_size)]

    return LSTMModel(lstm_exec=lstm_exec, symbol=lstm_sym, init_states=init_states,
                     last_states=last_states, seq_data=seq_data, seq_labels=seq_labels,
                     seq_outputs=seq_outputs, param_blocks=param_blocks)


def set_lstm_inputs(m, x_batch, y_batch):
    step_size = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    # print 'x batch shape %s' % str(x_batch[:, 0, :].shape)
    # print 'y batch shape %s' % str(y_batch.shape)
    for seqidx in range(step_size):
        m.seq_data[seqidx][:] = x_batch[:, seqidx, :]
        m.seq_labels[seqidx][:] = y_batch[:, seqidx]


# shape : num-instance * context-size
def train_lstm(model, X_train_batch, y_train_batch, X_val_batch, y_val_batch,
        num_epoch, optimizer='RMSProp', max_grad_norm=5.0, learning_rate=0.001, **kwargs):
    print >> logs, 'Training with train shape=%s' % str(X_train_batch.shape)
    print >> logs, 'Training with dev shape=%s' % str(X_val_batch.shape)

    m = model
    batch_size = m.seq_data[0].shape[0]
    step_size = len(m.seq_data)
    print >> logs, 'batch_size=%d' % batch_size
    print >> logs, 'step_size=%d' % step_size
    eta = 1e-4

    opt = mx.optimizer.create(optimizer, **kwargs)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)

    for iteration in range(num_epoch):
        # reset states
        for state in m.init_states:
            state.c[:] = 0.0
            state.h[:] = 0.0

        tic = time.time()
        num_correct = 0.
        num_total = 0.
        for begin in range(0, X_train_batch.shape[0], batch_size):
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            # m.seq_data[:] = batchX
            # m.seq_labels[:] = batchY
            set_lstm_inputs(m, batchX, batchY)

            m.lstm_exec.forward(is_train=True)

            m.lstm_exec.backward()
            # transfer the states
            for init, last in zip(m.init_states, m.last_states):
                last.c.copyto(init.c)
                last.h.copyto(init.h)

            # update parameters
            norm = 0.
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm;
            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)
                updater(idx, grad, weight)
                # reset gradient to zero
                grad[:] = 0.0

            pred = np.array([np.argmax(ypred.asnumpy(), axis=1) for ypred in m.seq_outputs])
            pred = pred.transpose()
            num_correct += sum((batchY == pred).flatten())
            num_total += batch_size * step_size

        # end of training epoch
        toc = time.time()
        train_acc = num_correct * 100.0 / num_total

        # saving checkpoint
        prefix = 'lstm'
        m.symbol.save('checkpoint/%s-symbol.json' % prefix)
        save_dict = { ('arg:%s' % k) :v  for k, v in m.lstm_exec.arg_dict.items() if is_param_name(k) }
        save_dict.update({('aux:%s' % k) : v for k, v in m.lstm_exec.aux_dict.items()})
        param_name = 'checkpoint/%s-%04d.params' % (prefix, iteration)
        mx.nd.save(param_name, save_dict)
        print >> logs, 'Saved checkpoint to %s' % param_name

        # evaluate on dev data
        num_correct = 0.
        num_total = 0.
        for begin in range(0, X_val_batch.shape[0], batch_size):
            batchX = X_val_batch[begin:begin+batch_size]
            batchY = y_val_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            # m.seq_data[:] = batchX
            # m.seq_labels[:] = batchY
            set_lstm_inputs(m, batchX, batchY)

            m.lstm_exec.forward(is_train=False)
            pred = np.array([np.argmax(ypred.asnumpy(), axis=1) for ypred in m.seq_outputs])
            pred = pred.transpose()
            num_correct += sum((batchY == pred).flatten())
            num_total += batch_size * step_size
    
        dev_acc = num_correct * 100 / float(num_total)
        print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy:%.3f---Dev Accuracy thus far: %.3f' \
            % (iteration, toc - tic, train_acc, dev_acc)
    

if __name__ == '__main__':
    lstm_model = setup_lstm_model(ctx=mx.cpu(0), num_lstm_layer=1,
                                  context_size = 7,
                                  num_hidden=100, num_embed=300,
                                  num_label=4, batch_size=50,
                                  vocab_size=1000,
                                  initializer=mx.initializer.Uniform(0.1),
                                  dropout=0.5)
