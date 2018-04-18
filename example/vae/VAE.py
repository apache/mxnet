# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import mxnet as mx
import numpy as np
import os
import logging

class VAE:
    """This class implements the Variational Auto Encoder"""
    
    def Bernoulli(x_hat,loss_label):
        return(-mx.symbol.sum(mx.symbol.broadcast_mul(loss_label,mx.symbol.log(x_hat))
                              + mx.symbol.broadcast_mul(1-loss_label,mx.symbol.log(1-x_hat)), axis=1))

    def __init__(self, n_latent=5, num_hidden_ecoder=400, num_hidden_decoder=400, x_train=None, x_valid=None,
                 batch_size=100, learning_rate=0.001, weight_decay=0.01, num_epoch=100, optimizer='sgd',
                 model_prefix=None, initializer=mx.init.Normal(0.01), likelihood=Bernoulli):
        self.n_latent = n_latent                      # dimension of the latent space Z
        self.num_hidden_ecoder = num_hidden_ecoder    # number of hidden units in the encoder
        self.num_hidden_decoder = num_hidden_decoder  # number of hidden units in the decoder
        self.batch_size = batch_size                  # mini batch size
        self.learning_rate = learning_rate            # learning rate during training
        self.weight_decay = weight_decay              # weight decay during training, for regularization of parameters
        self.num_epoch = num_epoch                    # total number of training epoch
        self.optimizer = optimizer                    # 'sgd' optimizer by default

        # train the model
        self.model, self.training_loss = VAE.train_vae(x_train, x_valid, batch_size, n_latent, num_hidden_ecoder,
                                                       num_hidden_decoder, learning_rate, weight_decay,
                                                       num_epoch,optimizer, model_prefix, likelihood, initializer)

        # save model parameters (i.e. weights and biases)
        self.arg_params = self.model.get_params()[0]

        # save loss(ELBO) for the training set
        nd_iter = mx.io.NDArrayIter(data={'data':x_train}, label={'loss_label':x_train}, batch_size=batch_size)

        # if saved parameters, can access them at specific iteration e.g. last epoch using
        #   sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, self.num_epoch)
        #   assert sym.tojson() == output.tojson()
        #   self.arg_params = arg_params

    @staticmethod
    def train_vae(x_train, x_valid, batch_size, n_latent, num_hidden_ecoder, num_hidden_decoder, learning_rate,
                  weight_decay, num_epoch, optimizer, model_prefix, likelihood, initializer):
        [N,features] = np.shape(x_train)          # number of examples and features

        # create data iterator to feed into NN
        nd_iter = mx.io.NDArrayIter(data={'data':x_train}, label={'loss_label':x_train}, batch_size=batch_size)

        if x_valid is not None:
            nd_iter_val = mx.io.NDArrayIter(data={'data':x_valid}, label={'loss_label':x_valid}, batch_size=batch_size)
        else:
            nd_iter_val = None

        data = mx.sym.var('data')
        loss_label = mx.sym.var('loss_label')

        # build network architecture
        encoder_h = mx.sym.FullyConnected(data=data, name="encoder_h", num_hidden=num_hidden_ecoder)
        act_h = mx.sym.Activation(data=encoder_h, act_type="tanh", name="activation_h")

        mu = mx.sym.FullyConnected(data=act_h, name="mu", num_hidden=n_latent)
        logvar = mx.sym.FullyConnected(data=act_h, name="logvar", num_hidden=n_latent)

        # latent manifold
        z = mu + mx.symbol.broadcast_mul(mx.symbol.exp(0.5*logvar),
                                         mx.symbol.random_normal(loc=0, scale=1, shape=(batch_size, n_latent)))
        decoder_z = mx.sym.FullyConnected(data=z, name="decoder_z", num_hidden=num_hidden_decoder)
        act_z = mx.sym.Activation(data=decoder_z, act_type="tanh", name="actication_z")

        decoder_x = mx.sym.FullyConnected(data=act_z, name="decoder_x", num_hidden=features)
        act_x = mx.sym.Activation(data=decoder_x, act_type="sigmoid", name='activation_x')

        KL = -0.5 * mx.symbol.sum(1+logvar-pow(mu,2)-mx.symbol.exp(logvar), axis=1)

        # compute minus ELBO to minimize
        loss = likelihood(act_x, loss_label)+KL
        output = mx.symbol.MakeLoss(sum(loss), name='loss')

        # train the model
        nd_iter.reset()
        logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

        model = mx.mod.Module(
            symbol=output ,
            data_names=['data'],
            label_names=['loss_label'])

        training_loss = list()

        def log_to_list(period, lst):
                def _callback(param):
                        """The checkpoint function."""
                        if param.nbatch % period == 0:
                                name, value = param.eval_metric.get()
                                lst.append(value)
                return _callback

        model.fit(nd_iter,  # train data
                  initializer=initializer, # initialize the weights and bias
                  eval_data=nd_iter_val,
                  optimizer=optimizer,  # use SGD to train
                  optimizer_params={'learning_rate':learning_rate, 'wd':weight_decay},
                  # save parameters for each epoch if model_prefix is supplied
                  epoch_end_callback=None if model_prefix==None else mx.callback.do_checkpoint(model_prefix, 1),
                  batch_end_callback=log_to_list(int(N/batch_size), training_loss),  # this can save the training loss
                  num_epoch=num_epoch,
                  eval_metric='Loss')

        return model,training_loss

    @staticmethod
    def encoder(model, x):
        params = model.arg_params
        encoder_n = np.shape(params['encoder_h_bias'].asnumpy())[0]
        encoder_h = np.dot(params['encoder_h_weight'].asnumpy(), np.transpose(x)) \
                    + np.reshape(params['encoder_h_bias'].asnumpy(), (encoder_n,1))
        act_h = np.tanh(encoder_h)
        mu = np.transpose(np.dot(params['mu_weight'].asnumpy(),act_h)) + params['mu_bias'].asnumpy()
        logvar = np.transpose(np.dot(params['logvar_weight'].asnumpy(),act_h)) + params['logvar_bias'].asnumpy()
        return mu,logvar

    @staticmethod
    def sampler(mu, logvar):
        z = mu + np.multiply(np.exp(0.5*logvar), np.random.normal(loc=0, scale=1,size=np.shape(logvar)))
        return z

    @staticmethod
    def decoder(model, z):
        params = model.arg_params
        decoder_n = np.shape(params['decoder_z_bias'].asnumpy())[0]
        decoder_z = np.dot(params['decoder_z_weight'].asnumpy(),np.transpose(z)) \
                    + np.reshape(params['decoder_z_bias'].asnumpy(),(decoder_n,1))
        act_z = np.tanh(decoder_z)
        decoder_x = np.transpose(np.dot(params['decoder_x_weight'].asnumpy(),act_z)) + params['decoder_x_bias'].asnumpy()
        reconstructed_x = 1/(1+np.exp(-decoder_x))
        return reconstructed_x
