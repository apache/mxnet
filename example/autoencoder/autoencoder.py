# pylint: skip-file
import mxnet as mx
import numpy as np
import data
import model
import logging
from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle

class AutoEncoderModel(model.MXModel):
    def setup(self, dims, pt_dropout=None, ft_dropout=None, input_act=None, internal_act='relu', output_act=None):
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
            istack, iargs, iargs_grad, iargs_mult = self.make_stack(i, self.data, dims[i], dims[i+1],
                                                        idropout, odropout, encoder_act, decoder_act)
            self.stacks.append(istack)
            self.args.update(iargs)
            self.args_grad.update(iargs_grad)
            self.args_mult.update(iargs_mult)

        self.encoder, self.internals = self.make_encoder(self.data, dims, ft_dropout, internal_act, output_act)
        self.decoder = self.make_decoder(self.encoder, dims, ft_dropout, internal_act, input_act)
        if input_act == 'softmax':
            self.loss = self.decoder
        else:
            self.loss = mx.symbol.LinearRegressionOutput(data=self.decoder, label=self.data)

    def make_stack(self, istack, data, num_input, num_hidden, idropout=None,
                   odropout=None, encoder_act='relu', decoder_act='relu'):
        x = data
        if idropout:
            x = mx.symbol.Dropout(data=x, p=idropout)
        x = mx.symbol.FullyConnected(name='encoder_%d'%istack, data=x, num_hidden=num_hidden)
        if encoder_act:
            x = mx.symbol.Activation(data=x, act_type=encoder_act)
        if odropout:
            x = mx.symbol.Dropout(data=x, p=odropout)
        x = mx.symbol.FullyConnected(name='decoder_%d'%istack, data=x, num_hidden=num_input)
        if decoder_act == 'softmax':
            x = mx.symbol.Softmax(data=x, label=data, prob_label=True, act_type=decoder_act)
        elif decoder_act:
            x = mx.symbol.Activation(data=x, act_type=decoder_act)
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
        init = mx.initializer.Normal(0.01)
        for k,v in args.items():
            init(k,v)

        return x, args, args_grad, args_mult

    def make_encoder(self, data, dims, dropout=None, internal_act='relu', output_act=None):
        x = data
        internals = []
        N = len(dims) - 1
        for i in range(N):
            x = mx.symbol.FullyConnected(name='encoder_%d'%i, data=x, num_hidden=dims[i+1])
            if internal_act and i < N-1:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
            elif output_act and i == N-1:
                x = mx.symbol.Activation(data=x, act_type=output_act)
            if dropout:
                x = mx.symbol.Dropout(data=x, p=dropout)
            internals.append(x)
        return x, internals

    def make_decoder(self, feature, dims, dropout=None, internal_act='relu', input_act=None):
        x = feature
        N = len(dims) - 1
        for i in reversed(range(N)):
            x = mx.symbol.FullyConnected(name='decoder_%d'%i, data=x, num_hidden=dims[i])
            if internal_act and i > 0:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
            elif input_act and i == 0:
                x = mx.symbol.Activation(data=x, act_type=input_act)
            if dropout and i > 0:
                x = mx.symbol.Dropout(data=x, p = dropout)
        return x

    def layerwise_pretrain(self, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):
        def l2_norm(label, pred):
            return np.mean(np.square(label-pred))/2.0
        solver = Solver('sgd', momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='roll_over')
        for i in range(self.N):
            if i == 0:
                data_iter_i = data_iter
            else:
                X_i = model.extract_feature(self.internals[i-1], self.args,
                                            data_iter, X.shape[0], self.xpu).values()[0]
                data_iter_i = mx.io.NDArrayIter({'data': X_i}, batch_size=batch_size,
                                                last_batch_handle='roll_over')
            logging.info('Pre-training layer %d...'%i)
            solver.solve(self.xpu, self.stacks[i], self.args, self.args_grad, data_iter_i,
                         0, n_iter, {}, False)

    def finetune(self, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):
        def l2_norm(label, pred):
            return np.mean(np.square(label-pred))/2.0
        solver = Solver('sgd', momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='roll_over')
        logging.info('Fine tuning...')
        solver.solve(self.xpu, self.loss, self.args, self.args_grad, data_iter,
                     0, n_iter, {}, False)

    def eval(self, X):
        batch_size = 100
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        Y = model.extract_feature(self.loss, self.args, data_iter,
                                 X.shape[0], self.xpu).values()[0]
        return np.mean(np.square(Y-X))/2.0



if __name__ == '__main__':
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG) 
    ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2)

    X, _ = data.get_mnist()
    train_X = X[:60000]
    val_X = X[60000:]

    ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0,
                             lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    ae_model.finetune(train_X, 256, 100000, 'sgd', l_rate=0.1, decay=0.0,
                   lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    ae_model.save('mnist_pt.arg')
    ae_model.load('mnist_pt.arg')
    print "Training error:", ae_model.eval(train_X)
    print "Validation error:", ae_model.eval(val_X)









