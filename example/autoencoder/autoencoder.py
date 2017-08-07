# pylint: skip-file
import mxnet as mx
from mxnet import misc
import numpy as np
import model
import logging
from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle

class AutoEncoderModel(model.MXModel):
    def setup(self, dims, sparseness_penalty=None, pt_dropout=None, ft_dropout=None, input_act=None, internal_act='relu', output_act=None):
        self.N = len(dims) - 1
        self.dims = dims
        self.stacks = []
        self.pt_dropout = pt_dropout
        self.ft_dropout = ft_dropout
        self.input_act = input_act
        self.internal_act = internal_act
        self.output_act = output_act

        self.data = mx.symbol.Variable('data')
        for i in range(self.N):
            if i == 0:
                decoder_act = input_act
                idropout = None
            else:
                decoder_act = internal_act
                idropout = pt_dropout
            if i == self.N-1:
                encoder_act = output_act
                odropout = None
            else:
                encoder_act = internal_act
                odropout = pt_dropout
            istack, iargs, iargs_grad, iargs_mult, iauxs = self.make_stack(i, self.data, dims[i], dims[i+1],
                                                sparseness_penalty, idropout, odropout, encoder_act, decoder_act)
            self.stacks.append(istack)
            self.args.update(iargs)
            self.args_grad.update(iargs_grad)
            self.args_mult.update(iargs_mult)
            self.auxs.update(iauxs)
        self.encoder, self.internals = self.make_encoder(self.data, dims, sparseness_penalty, ft_dropout, internal_act, output_act)
        self.decoder = self.make_decoder(self.encoder, dims, sparseness_penalty, ft_dropout, internal_act, input_act)
        if input_act == 'softmax':
            self.loss = self.decoder
        else:
            self.loss = mx.symbol.LinearRegressionOutput(data=self.decoder, label=self.data)

    def make_stack(self, istack, data, num_input, num_hidden, sparseness_penalty=None, idropout=None,
                   odropout=None, encoder_act='relu', decoder_act='relu'):
        x = data
        if idropout:
            x = mx.symbol.Dropout(data=x, p=idropout)
        x = mx.symbol.FullyConnected(name='encoder_%d'%istack, data=x, num_hidden=num_hidden)
        if encoder_act:
            x = mx.symbol.Activation(data=x, act_type=encoder_act)
            if encoder_act == 'sigmoid' and sparseness_penalty:
                x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_encoder_%d' % istack, penalty=sparseness_penalty)
        if odropout:
            x = mx.symbol.Dropout(data=x, p=odropout)
        x = mx.symbol.FullyConnected(name='decoder_%d'%istack, data=x, num_hidden=num_input)
        if decoder_act == 'softmax':
            x = mx.symbol.Softmax(data=x, label=data, prob_label=True, act_type=decoder_act)
        elif decoder_act:
            x = mx.symbol.Activation(data=x, act_type=decoder_act)
            if decoder_act == 'sigmoid' and sparseness_penalty:
                x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_decoder_%d' % istack, penalty=sparseness_penalty)
            x = mx.symbol.LinearRegressionOutput(data=x, label=data)
        else:
            x = mx.symbol.LinearRegressionOutput(data=x, label=data)

        args = {'encoder_%d_weight'%istack: mx.nd.empty((num_hidden, num_input), self.xpu),
                'encoder_%d_bias'%istack: mx.nd.empty((num_hidden,), self.xpu),
                'decoder_%d_weight'%istack: mx.nd.empty((num_input, num_hidden), self.xpu),
                'decoder_%d_bias'%istack: mx.nd.empty((num_input,), self.xpu),}
        args_grad = {'encoder_%d_weight'%istack: mx.nd.empty((num_hidden, num_input), self.xpu),
                     'encoder_%d_bias'%istack: mx.nd.empty((num_hidden,), self.xpu),
                     'decoder_%d_weight'%istack: mx.nd.empty((num_input, num_hidden), self.xpu),
                     'decoder_%d_bias'%istack: mx.nd.empty((num_input,), self.xpu),}
        args_mult = {'encoder_%d_weight'%istack: 1.0,
                     'encoder_%d_bias'%istack: 2.0,
                     'decoder_%d_weight'%istack: 1.0,
                     'decoder_%d_bias'%istack: 2.0,}
        auxs = {}
        if encoder_act == 'sigmoid' and sparseness_penalty:
            auxs['sparse_encoder_%d_moving_avg' % istack] = mx.nd.ones((num_hidden), self.xpu) * 0.5
        if decoder_act == 'sigmoid' and sparseness_penalty:
            auxs['sparse_decoder_%d_moving_avg' % istack] = mx.nd.ones((num_input), self.xpu) * 0.5
        init = mx.initializer.Uniform(0.07)
        for k,v in args.items():
            init(k,v)

        return x, args, args_grad, args_mult, auxs

    def make_encoder(self, data, dims, sparseness_penalty=None, dropout=None, internal_act='relu', output_act=None):
        x = data
        internals = []
        N = len(dims) - 1
        for i in range(N):
            x = mx.symbol.FullyConnected(name='encoder_%d'%i, data=x, num_hidden=dims[i+1])
            if internal_act and i < N-1:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
                if internal_act=='sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_encoder_%d' % i, penalty=sparseness_penalty)
            elif output_act and i == N-1:
                x = mx.symbol.Activation(data=x, act_type=output_act)
                if output_act=='sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_encoder_%d' % i, penalty=sparseness_penalty)
            if dropout:
                x = mx.symbol.Dropout(data=x, p=dropout)
            internals.append(x)
        return x, internals

    def make_decoder(self, feature, dims, sparseness_penalty=None, dropout=None, internal_act='relu', input_act=None):
        x = feature
        N = len(dims) - 1
        for i in reversed(range(N)):
            x = mx.symbol.FullyConnected(name='decoder_%d'%i, data=x, num_hidden=dims[i])
            if internal_act and i > 0:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
                if internal_act=='sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_decoder_%d' % i, penalty=sparseness_penalty)
            elif input_act and i == 0:
                x = mx.symbol.Activation(data=x, act_type=input_act)
                if input_act=='sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(data=x, name='sparse_decoder_%d' % i, penalty=sparseness_penalty)
            if dropout and i > 0:
                x = mx.symbol.Dropout(data=x, p=dropout)
        return x

    def layerwise_pretrain(self, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):
        def l2_norm(label, pred):
            return np.mean(np.square(label-pred))/2.0
        solver = Solver(optimizer, momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=True,
                                      last_batch_handle='roll_over')
        for i in range(self.N):
            if i == 0:
                data_iter_i = data_iter
            else:
                X_i = list(model.extract_feature(self.internals[i-1], self.args, self.auxs,
                                            data_iter, X.shape[0], self.xpu).values())[0]
                data_iter_i = mx.io.NDArrayIter({'data': X_i}, batch_size=batch_size,
                                                last_batch_handle='roll_over')
            logging.info('Pre-training layer %d...'%i)
            solver.solve(self.xpu, self.stacks[i], self.args, self.args_grad, self.auxs, data_iter_i,
                         0, n_iter, {}, False)

    def finetune(self, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):
        def l2_norm(label, pred):
           return np.mean(np.square(label-pred))/2.0
        solver = Solver(optimizer, momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=True,
                                      last_batch_handle='roll_over')
        logging.info('Fine tuning...')
        solver.solve(self.xpu, self.loss, self.args, self.args_grad, self.auxs, data_iter,
                     0, n_iter, {}, False)

    def eval(self, X):
        batch_size = 100
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        Y = list(model.extract_feature(self.loss, self.args, self.auxs, data_iter,
                                 X.shape[0], self.xpu).values())[0]
        return np.mean(np.square(Y-X))/2.0
