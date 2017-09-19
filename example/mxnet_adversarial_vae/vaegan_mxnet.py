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

'''
Created on Jun 15, 2017

@author: shujon
'''

from __future__ import print_function
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
import logging
import cv2
from datetime import datetime
from PIL import Image
import os
import argparse
from scipy.io import savemat
#from layer import GaussianSampleLayer

######################################################################
#An adversarial variational autoencoder implementation in mxnet
# following the implementation at https://github.com/JeremyCCHsu/tf-vaegan
# of paper `Larsen, Anders Boesen Lindbo, et al. "Autoencoding beyond pixels using a 
# learned similarity metric." arXiv preprint arXiv:1512.09300 (2015).`
######################################################################

#constant operator in mxnet, not used in this code
@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)
        
#######################################################################
#The encoder is a CNN which takes 32x32 image as input
# generates the 100 dimensional shape embedding as a sample from normal distribution
# using predicted meand and variance 
#######################################################################        

def encoder(nef, z_dim, batch_size, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    
    data = mx.sym.Variable('data')
    #label = mx.sym.Variable('label')    

    e1 = mx.sym.Convolution(data, name='enc1', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef, no_bias=no_bias)
    ebn1 = BatchNorm(e1, name='encbn1', fix_gamma=fix_gamma, eps=eps)
    eact1 = mx.sym.LeakyReLU(ebn1, name='encact1', act_type='leaky', slope=0.2)

    e2 = mx.sym.Convolution(eact1, name='enc2', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*2, no_bias=no_bias)
    ebn2 = BatchNorm(e2, name='encbn2', fix_gamma=fix_gamma, eps=eps)
    eact2 = mx.sym.LeakyReLU(ebn2, name='encact2', act_type='leaky', slope=0.2)

    e3 = mx.sym.Convolution(eact2, name='enc3', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*4, no_bias=no_bias)
    ebn3 = BatchNorm(e3, name='encbn3', fix_gamma=fix_gamma, eps=eps)
    eact3 = mx.sym.LeakyReLU(ebn3, name='encact3', act_type='leaky', slope=0.2)

    e4 = mx.sym.Convolution(eact3, name='enc4', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*8, no_bias=no_bias)
    ebn4 = BatchNorm(e4, name='encbn4', fix_gamma=fix_gamma, eps=eps)
    eact4 = mx.sym.LeakyReLU(ebn4, name='encact4', act_type='leaky', slope=0.2)

    eact4 = mx.sym.Flatten(eact4)

    z_mu = mx.sym.FullyConnected(eact4, num_hidden=z_dim, name="enc_mu")
    z_lv = mx.sym.FullyConnected(eact4, num_hidden=z_dim, name="enc_lv")
    

    #eps = mx.symbol.random_normal(loc=0, scale=1, shape=(batch_size,z_dim) )
    #std = mx.symbol.sqrt(mx.symbol.exp(z_lv))
    #z = mx.symbol.elemwise_add(z_mu, mx.symbol.broadcast_mul(eps, std))    
    
    z = z_mu + mx.symbol.broadcast_mul(mx.symbol.exp(0.5*z_lv),mx.symbol.random_normal(loc=0, scale=1,shape=(batch_size,z_dim))) 
    
    return z_mu, z_lv, z    
    
    
#######################################################################
#The genrator is a CNN which takes 100 dimensional embedding as input
# and reconstructs the input image given to the encoder
####################################################################### 
    
def generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, z_dim=100, activation='sigmoid'):
    
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')
    
    rand = mx.sym.Reshape(rand, shape=(-1, z_dim, 1, 1))

    #g1 = mx.sym.FullyConnected(rand, name="g1", num_hidden=2*2*ngf*8, no_bias=True)
    g1 = mx.sym.Deconvolution(rand, name='gen1', kernel=(5,5), stride=(2,2),target_shape=(2,2), num_filter=ngf*8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='genbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name="genact1", act_type="relu")
    # 4 x 4
    #gact1 = mx.sym.Reshape(gact1, shape=(-1, ngf * 8, 2, 2))

    #g1 = mx.sym.Deconvolution(g0, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
    #gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    #gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='gen2', kernel=(5,5), stride=(2,2),target_shape=(4,4), num_filter=ngf*4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='genbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='genact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='gen3', kernel=(5,5), stride=(2,2), target_shape=(8,8), num_filter=ngf*2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='genbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='genact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='gen4', kernel=(5,5), stride=(2,2), target_shape=(16,16), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='genbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='genact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='gen5', kernel=(5,5), stride=(2,2), target_shape=(32,32), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='genact5', act_type=activation)    

    return gout


#######################################################################
# First part of the discriminator which takes a 32x32 image as input
# and output a convolutional feature map, this is required to calculate
# the layer loss
####################################################################### 

def discriminator1(ndf, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):

    BatchNorm = mx.sym.BatchNorm
    
    data = mx.sym.Variable('data')
    
    #label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=ndf*2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=ndf*4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    return dact3 

#######################################################################
# Second part of the discriminator which takes a 256x8x8 feature map as input
# and generates the loss based on whether the input image was a real one or fake one
####################################################################### 
def discriminator2(ndf, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
        
    BatchNorm = mx.sym.BatchNorm
            
    data = mx.sym.Variable('data')
    
    label = mx.sym.Variable('label')
    
    d4 = mx.sym.Convolution(data, name='d4', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=ndf*8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    #d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
    #d5 = mx.sym.Flatten(d5)
    h = mx.sym.Flatten(dact4)    
    
    d5 = mx.sym.FullyConnected(h, num_hidden=1, name="d5")
            
    #dloss = (0.5 * (label == 0) + (label != 0) ) * mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss') 
    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss') 
    
    return dloss

#######################################################################
# GaussianLogDensity loss calculation for layer wise loss
####################################################################### 
def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity', EPSILON = 1e-6):
    c = mx.sym.ones_like(log_var)*2.0 * 3.1416
    c = mx.symbol.log(c)
    var = mx.sym.exp(log_var)
    x_mu2 = mx.symbol.square(x - mu)   # [Issue] not sure the dim works or not?
    x_mu2_over_var = mx.symbol.broadcast_div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    #log_prob = (x_mu2)
    log_prob = mx.symbol.sum(log_prob, axis=1, name=name)   # keep_dims=True,
    return log_prob

#######################################################################
# Calculate the discriminator layer loss
####################################################################### 
def DiscriminatorLayerLoss():
        
    data = mx.sym.Variable('data')
    
    label = mx.sym.Variable('label')    
    
    data = mx.sym.Flatten(data)
    label = mx.sym.Flatten(label)        
    
    label = mx.sym.BlockGrad(label)
    
    zeros = mx.sym.zeros_like(data)
    
    output = -GaussianLogDensity(label, data, zeros)
    
    dloss = mx.symbol.MakeLoss(mx.symbol.mean(output),name='lloss')
            
    #dloss = mx.sym.MAERegressionOutput(data=data, label=label, name='lloss')
    
    return dloss    

#######################################################################
# KLDivergence loss
####################################################################### 
def KLDivergenceLoss():
    
    data = mx.sym.Variable('data')
    mu1, lv1 = mx.sym.split(data,  num_outputs=2, axis=0)
    mu2 = mx.sym.zeros_like(mu1)
    lv2 = mx.sym.zeros_like(lv1)
    
    v1 = mx.sym.exp(lv1)
    v2 = mx.sym.exp(lv2)
    mu_diff_sq = mx.sym.square(mu1 - mu2)
    dimwise_kld = .5 * (
    (lv2 - lv1) + mx.symbol.broadcast_div(v1, v2) + mx.symbol.broadcast_div(mu_diff_sq, v2) - 1.)
    KL = mx.symbol.sum(dimwise_kld, axis=1)
        
    KLloss = mx.symbol.MakeLoss(mx.symbol.mean(KL),name='KLloss')
    return KLloss

#######################################################################
# Get the dataset
####################################################################### 
def get_data(path, activation):
    #mnist = fetch_mldata('MNIST original')
    #import ipdb; ipdb.set_trace()
    data = []
    image_names = []
    #set the path to the 32x32 images of caltech101 dataset created using the convert_data_inverted.py script
    #path = '/home/ubuntu/datasets/caltech101/data/images32x32/'
    #path_wo_ext = '/home/ubuntu/datasets/caltech101/data/images/'
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)
        image_names.append(filename)
        if img is not None:
            data.append(img)
    
    data = np.asarray(data)
    
    if activation == 'sigmoid':
        #converting image values from 0 to 1 as the generator activation is sigmoid
        data = data.astype(np.float32)/(255.0)
    elif activation == 'tanh':
        #converting image values from -1 to 1 as the generator activation is tanh
        data = data.astype(np.float32)/(255.0/2) - 1.0

    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2])) 
    
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(data.shape[0])
    X = data[p]

    return X, image_names

#######################################################################
# Create a random iterator for generator
####################################################################### 
class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

#######################################################################
# fill the ith grid of the buffer matrix with the values from the img
# buf : buffer matrix
# i : serial of the image in the 2D grid  
# img : image data
# shape : ( height width depth ) of image 
####################################################################### 
def fill_buf(buf, i, img, shape):
    
    #n = buf.shape[0]/shape[1]
    
    # grid height is a multiple of individual image height
    m = buf.shape[0]/shape[0]

    sx = (i%m)*shape[1]
    sy = (i/m)*shape[0]
    buf[sy:sy+shape[0], sx:sx+shape[1], :] = img


#######################################################################
# create a grid of images and save it as a final image
# title : grid image name
# X : array of images
####################################################################### 
def visual(title, X, activation):
    assert len(X.shape) == 4
            
    X = X.transpose((0, 2, 3, 1))
    if activation == 'sigmoid':
        X = np.clip((X)*(255.0), 0, 255).astype(np.uint8)
    elif activation == 'tanh':
        X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    #buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    #local_out = 1
    #num = 1
    cv2.imwrite('%s.jpg' % (title), buff)


#######################################################################
# adverial training of the VAE
####################################################################### 
def train(dataset, nef, ndf, ngf, nc, batch_size, Z, lr, beta1, epsilon, ctx, check_point, g_dl_weight, output_path, checkpoint_path, data_path, activation,num_epoch, save_after_every, visualize_after_every, show_after_every): 
    
    #encoder
    z_mu, z_lv, z = encoder(nef, Z, batch_size)
    symE = mx.sym.Group([z_mu, z_lv, z])        
    
    #generator
    symG = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, z_dim = Z, activation=activation )
    
    #discriminator
    h  = discriminator1(ndf)
    dloss  = discriminator2(ndf)
    #symD = mx.sym.Group([dloss, h])
    symD1 = h
    symD2 = dloss    

    
    #symG, symD = make_dcgan_sym(nef, ngf, ndf, nc)
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    #if dataset == 'caltech':
    X_train, _ = get_data(data_path, activation)
        #import ipdb; ipdb.set_trace()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size, shuffle=True)
    #elif dataset == 'imagenet':
    #    train_iter = ImagenetIter(imgnet_path, batch_size, (3, 32, 32))
    
    #print('=============================================', str(batch_size), str(Z))
    rand_iter = RandIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    # =============module E=============
    modE = mx.mod.Module(symbol=symE, data_names=('data',), label_names=None, context=ctx)
    modE.bind(data_shapes=train_iter.provide_data)
    modE.init_params(initializer=mx.init.Normal(0.02))
    modE.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 1e-6,
            'beta1': beta1,
            'epsilon': epsilon,
            'rescale_grad': (1.0/batch_size)
        })
    mods = [modE]    

    # =============module G=============
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data, inputs_need_grad=True)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 1e-6,
            'beta1': beta1,
            'epsilon': epsilon,
            #'rescale_grad': (1.0/batch_size)
        })
    mods.append(modG)

    # =============module D=============
    modD1 = mx.mod.Module(symD1, label_names=[], context=ctx)
    modD2 = mx.mod.Module(symD2, label_names=('label',), context=ctx)
    modD = mx.mod.SequentialModule()
    modD.add(modD1).add(modD2, take_labels=True, auto_wiring=True)    
    #modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 1e-3,
            'beta1': beta1,
            'epsilon': epsilon,
            'rescale_grad': (1.0/batch_size)
        })
    mods.append(modD)
    
    
    # =============module DL=============    
    symDL = DiscriminatorLayerLoss()
    modDL = mx.mod.Module(symbol=symDL, data_names=('data',), label_names=('label',), context=ctx)
    modDL.bind(data_shapes=[('data', (batch_size,nef * 4,4,4))], ################################################################################################################################ fix 512 here
              label_shapes=[('label', (batch_size,nef * 4,4,4))],
              inputs_need_grad=True)
    modDL.init_params(initializer=mx.init.Normal(0.02))
    modDL.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
            'epsilon': epsilon,
            'rescale_grad': (1.0/batch_size)
        })
    
    # =============module KL=============    
    symKL = KLDivergenceLoss()
    modKL = mx.mod.Module(symbol=symKL, data_names=('data',), label_names=None, context=ctx)
    modKL.bind(data_shapes=[('data', (batch_size*2,Z))],
               inputs_need_grad=True)
    modKL.init_params(initializer=mx.init.Normal(0.02))
    modKL.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
            'epsilon': epsilon,
            'rescale_grad': (1.0/batch_size)
        })    
    mods.append(modKL)
       
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    # ============calculating prediction accuracy==============
    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    # ============calculating binary cross-entropy loss==============
    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()
    
    # ============calculating KL divergence loss==============
    def kldivergence(label, pred):
        #pred = pred.ravel()
        #label = label.ravel()
        mean, log_var = np.split(pred, 2, axis=0)
        var = np.exp(log_var)
        KLLoss = -0.5 * np.sum(1 + log_var - np.power(mean, 2) - var)
        KLLoss = KLLoss / nElements
        return KLLoss   

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mE = mx.metric.CustomMetric(kldivergence)
    mACC = mx.metric.CustomMetric(facc)

    print('Training...')
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            #print('======================================================================')
            #print(outG)

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            #modD.update()
            gradD11 = [[grad.copyto(grad.context) for grad in grads] for grads in modD1._exec_group.grad_arrays]
            gradD12 = [[grad.copyto(grad.context) for grad in grads] for grads in modD2._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])


            #update discriminator on decoded
            modE.forward(batch, is_train=True)
            mu, lv, z = modE.get_outputs()
            #z = GaussianSampleLayer(mu, lv)          
            z = z.reshape((batch_size, Z, 1, 1))
            sample = mx.io.DataBatch([z], label=None, provide_data = [('rand', (batch_size, Z, 1, 1))])                          
            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()      
            label[:] = 0    
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()
                        
            #modD.update()
            gradD21 = [[grad.copyto(grad.context) for grad in grads] for grads in modD1._exec_group.grad_arrays]
            gradD22 = [[grad.copyto(grad.context) for grad in grads] for grads in modD2._exec_group.grad_arrays]    
            
            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            lx = [out.copyto(out.context) for out in modD1.get_outputs()]            
            modD.backward()
            for gradsr, gradsf, gradsd in zip(modD1._exec_group.grad_arrays, gradD11, gradD21):
                for gradr, gradf, gradd in zip(gradsr, gradsf, gradsd):
                    gradr += 0.5 * (gradf + gradd)
            for gradsr, gradsf, gradsd in zip(modD2._exec_group.grad_arrays, gradD12, gradD22):
                for gradr, gradf, gradd in zip(gradsr, gradsf, gradsd):
                    gradr += 0.5 * (gradf + gradd)       
                                        
            modD.update()
            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator twice as the discriminator is too strong           
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()            
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            #modG.update()
            gradG1 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]                        
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()      
            label[:] = 1    
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()            
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG2 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            #modG.update() 
            mG.update([label], modD.get_outputs())
            
            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()            
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()            
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            modDL.backward()
            dlGrad = modDL.get_input_grads()                        
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)          
           
            for grads, gradsG1, gradsG2 in zip(modG._exec_group.grad_arrays, gradG1, gradG2):
                for grad, gradg1, gradg2 in zip(grads, gradsG1, gradsG2):
                    grad = g_dl_weight * grad + 0.5 * (gradg1 + gradg2)            
            
            modG.update()            
            mG.update([label], modD.get_outputs())

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()            
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            #modG.update()
            gradG1 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]                        
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()      
            label[:] = 1    
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()            
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG2 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            #modG.update() 
            mG.update([label], modD.get_outputs())
            
            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()            
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()            
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            modDL.backward()
            dlGrad = modDL.get_input_grads()                        
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)          
           
            for grads, gradsG1, gradsG2 in zip(modG._exec_group.grad_arrays, gradG1, gradG2):
                for grad, gradg1, gradg2 in zip(grads, gradsG1, gradsG2):
                    grad = g_dl_weight * grad + 0.5 * (gradg1 + gradg2)            
            
            modG.update()            
            mG.update([label], modD.get_outputs())
            
                        
            ##update encoder--------------------------------------------------
            
            #modE.forward(batch, is_train=True)
            #mu, lv, z = modE.get_outputs()          
            #z = z.reshape((batch_size, Z, 1, 1))
            #sample = mx.io.DataBatch([z], label=None, provide_data = [('rand', (batch_size, Z, 1, 1))])                          
            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            
            #update generator
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()            
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            DLloss = modDL.get_outputs()
            modDL.backward()
            dlGrad = modDL.get_input_grads()                        
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)          
            #modG.update()
            
            #print('updating encoder=====================================')            
            
            #update encoder
            nElements = batch_size
            #var = mx.ndarray.exp(lv)
            
            modKL.forward(mx.io.DataBatch([mx.ndarray.concat(mu,lv, dim=0)]), is_train=True)
            KLloss = modKL.get_outputs()
            modKL.backward()
            gradKLLoss = modKL.get_input_grads()
                        
            diffG = modG.get_input_grads()
            #print('======================================================================')
            #print(np.sum(diffG[0].asnumpy()))
            diffG = diffG[0].reshape((batch_size, Z))
            modE.backward(mx.ndarray.split(gradKLLoss[0], num_outputs=2, axis=0) + [diffG])
            modE.update()
            #print('mu type : ')
            #print(type(mu))
            pred = mx.ndarray.concat(mu,lv, dim=0)
            #print(pred)
            mE.update([pred], [pred])            
            


            if mon is not None:
                mon.toc_print()
                
            t += 1
            if t % show_after_every == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get(), mE.get(), KLloss[0].asnumpy(), DLloss[0].asnumpy())
                mACC.reset()
                mG.reset()
                mD.reset()
                mE.reset()
                
            if epoch % visualize_after_every == 0:
                visual(output_path +'gout'+str(epoch), outG[0].asnumpy(), activation)
                #diff = diffD[0].asnumpy()
                #diff = (diff - diff.mean())/diff.std()
                #visual('diff', diff)
                visual(output_path + 'data'+str(epoch), batch.data[0].asnumpy(), activation)

        if check_point and epoch % save_after_every == 0:
            print('Saving...')
            modG.save_params(checkpoint_path + '/%s_G-%04d.params'%(dataset, epoch))
            modD.save_params(checkpoint_path + '/%s_D-%04d.params'%(dataset, epoch))
            modE.save_params(checkpoint_path + '/%s_E-%04d.params'%(dataset, epoch))

#######################################################################
# Test the VAE with a pretrained encoder and generator. 
# Keep the batch size 1 
####################################################################### 
def test(nef, ngf, nc, batch_size, Z, ctx, pretrained_encoder_path, pretrained_generator_path, output_path, data_path, activation, save_embedding, embedding_path = ''):
    
    #encoder
    z_mu, z_lv, z = encoder(nef, Z, batch_size)
    symE = mx.sym.Group([z_mu, z_lv, z])    
    
    #generator
    symG = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, z_dim = Z, activation=activation )    
    
    #symG, symD = make_dcgan_sym(nef, ngf, ndf, nc)
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    X_test, image_names = get_data(data_path, activation)
    #import ipdb; ipdb.set_trace()
    test_iter = mx.io.NDArrayIter(X_test, batch_size=batch_size, shuffle=False)

    # =============module E=============
    modE = mx.mod.Module(symbol=symE, data_names=('data',), label_names=None, context=ctx)
    modE.bind(data_shapes=test_iter.provide_data)
    #modE.init_params(initializer=mx.init.Normal(0.02))
    modE.load_params(pretrained_encoder_path)

    # =============module G============= 
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=[('rand', (1, Z, 1, 1))])
    #modG.init_params(initializer=mx.init.Normal(0.02))        
    modG.load_params(pretrained_generator_path)

    print('Testing...')

    # =============test===============
    test_iter.reset()
    for t, batch in enumerate(test_iter):

        #update discriminator on decoded
        modE.forward(batch, is_train=False)
        mu, lv, z = modE.get_outputs()
        #z = GaussianSampleLayer(mu, lv)          
        mu = mu.reshape((batch_size, Z, 1, 1))
        sample = mx.io.DataBatch([mu], label=None, provide_data = [('rand', (batch_size, Z, 1, 1))])         
        modG.forward(sample, is_train=False)
        outG = modG.get_outputs()                              
            
        visual(output_path + '/' + 'gout'+str(t), outG[0].asnumpy(), activation)
        visual(output_path +  '/' + 'data'+str(t), batch.data[0].asnumpy(), activation)
        image_name = image_names[t].split('.')[0]
        
        if save_embedding:            
            savemat(embedding_path+'/'+image_name+'.mat', {'embedding':mu.asnumpy()})    

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test an Adversarial Variatiional Encoder')

    parser.add_argument('--train', help='train the network', action='store_true')
    parser.add_argument('--test', help='test the network', action='store_true')
    parser.add_argument('--save_embedding', help='saves the shape embedding of each input image', action='store_true')                   
    parser.add_argument('--dataset', help='dataset name', default='caltech', type=str)
    parser.add_argument('--activation', help='activation i.e. sigmoid or tanh', default='sigmoid', type=str)    
    parser.add_argument('--training_data_path', help='training data path', default='/home/ubuntu/datasets/caltech101/data/images32x32/', type=str)    
    parser.add_argument('--testing_data_path', help='testing data path', default='/home/ubuntu/datasets/MPEG7dataset/images/', type=str)
    parser.add_argument('--pretrained_encoder_path', help='pretrained encoder model path', default='checkpoints32x32_sigmoid/caltech_E-0045.params', type=str)
    parser.add_argument('--pretrained_generator_path', help='pretrained generator model path', default='checkpoints32x32_sigmoid/caltech_G-0045.params', type=str)    
    parser.add_argument('--output_path', help='output path for the generated images', default='outputs32x32_sigmoid/', type=str)
    parser.add_argument('--embedding_path', help='output path for the generated embeddings', default='outputs32x32_sigmoid/', type=str)    
    parser.add_argument('--checkpoint_path', help='checkpoint saving path ', default='checkpoints32x32_sigmoid/', type=str)     
    parser.add_argument('--nef', help='encoder filter count in the first layer', default=64, type=int)
    parser.add_argument('--ndf', help='discriminator filter count in the first layer', default=64, type=int)
    parser.add_argument('--ngf', help='generator filter count in the second last layer', default=64, type=int)    
    parser.add_argument('--nc', help='generator filter count in the last layer i.e. 1 for grayscale image, 3 for RGB image', default=1, type=int)    
    parser.add_argument('--batch_size', help='batch size, keep it 1 during testing', default=64, type=int)    
    parser.add_argument('--Z', help='embedding size', default=100, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)         
    parser.add_argument('--beta1', help='beta1 for adam optimizer', default=0.5, type=float)    
    parser.add_argument('--epsilon', help='epsilon for adam optimizer', default=1e-5, type=float)
    parser.add_argument('--g_dl_weight', help='discriminator layer loss weight', default=1e-1, type=float)                     
    parser.add_argument('--gpu', help='gpu index', default=0, type=int)
    parser.add_argument('--num_epoch', help='number of maximum epochs ', default=45, type=int)
    parser.add_argument('--save_after_every', help='save checkpoint after every this number of epochs ', default=5, type=int)
    parser.add_argument('--visualize_after_every', help='save output images after every this number of epochs', default=5, type=int)
    parser.add_argument('--show_after_every', help='show metrics after this number of iterations', default=10, type=int)    
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # gpu context 
    ctx = mx.gpu(args.gpu)
    
    # checkpoint saving flags
    check_point = True
    
    if args.train:        
        train(args.dataset, args.nef, args.ndf, args.ngf, args.nc, args.batch_size, args.Z, args.lr, args.beta1, args.epsilon, ctx, check_point, args.g_dl_weight, args.output_path, args.checkpoint_path, args.training_data_path, args.activation, args.num_epoch, args.save_after_every, args.visualize_after_every, args.show_after_every)
    
    if args.test:
        test(args.nef, args.ngf, args.nc, 1, args.Z, ctx, args.pretrained_encoder_path, args.pretrained_generator_path, args.output_path, args.testing_data_path, args.activation, args.save_embedding, args.embedding_path)    

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    
    main()
        
  
    
