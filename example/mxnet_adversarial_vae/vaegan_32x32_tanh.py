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
#from layer import GaussianSampleLayer

@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)

def encoder(nef, z_dim, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    
    data = mx.sym.Variable('data')
    #label = mx.sym.Variable('label')    

    e1 = mx.sym.Convolution(data, name='e1', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef, no_bias=no_bias)
    ebn1 = BatchNorm(e1, name='ebn1', fix_gamma=fix_gamma, eps=eps)
    eact1 = mx.sym.LeakyReLU(ebn1, name='eact1', act_type='leaky', slope=0.2)

    e2 = mx.sym.Convolution(eact1, name='e2', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*2, no_bias=no_bias)
    ebn2 = BatchNorm(e2, name='ebn2', fix_gamma=fix_gamma, eps=eps)
    eact2 = mx.sym.LeakyReLU(ebn2, name='eact2', act_type='leaky', slope=0.2)

    e3 = mx.sym.Convolution(eact2, name='e3', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*4, no_bias=no_bias)
    ebn3 = BatchNorm(e3, name='ebn3', fix_gamma=fix_gamma, eps=eps)
    eact3 = mx.sym.LeakyReLU(ebn3, name='eact3', act_type='leaky', slope=0.2)

    e4 = mx.sym.Convolution(eact3, name='e4', kernel=(5,5), stride=(2,2), pad=(2,2), num_filter=nef*8, no_bias=no_bias)
    ebn4 = BatchNorm(e4, name='ebn4', fix_gamma=fix_gamma, eps=eps)
    eact4 = mx.sym.LeakyReLU(ebn4, name='eact4', act_type='leaky', slope=0.2)

    eact4 = mx.sym.Flatten(eact4)

    z_mu = mx.sym.FullyConnected(eact4, num_hidden=z_dim, name="z_mu")
    z_lv = mx.sym.FullyConnected(eact4, num_hidden=z_dim, name="z_lv")
    

    #eps = mx.symbol.random_normal(loc=0, scale=1, shape=(batch_size,z_dim) )
    #std = mx.symbol.sqrt(mx.symbol.exp(z_lv))
    #z = mx.symbol.elemwise_add(z_mu, mx.symbol.broadcast_mul(eps, std))    
    
    z = z_mu + mx.symbol.broadcast_mul(mx.symbol.exp(0.5*z_lv),mx.symbol.random_normal(loc=0, scale=1,shape=(batch_size,z_dim))) 
    
    return z_mu, z_lv, z    
    
def generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.FullyConnected(rand, name="g1", num_hidden=2*2*ngf*8, no_bias=True)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name="gact1", act_type="relu")
    # 4 x 4
    gact1 = mx.sym.Reshape(gact1, shape=(-1, ngf * 8, 2, 2))

    #g1 = mx.sym.Deconvolution(g0, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
    #gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    #gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(5,5), stride=(2,2),target_shape=(4,4), num_filter=ngf*4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(5,5), stride=(2,2), target_shape=(8,8), num_filter=ngf*2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(5,5), stride=(2,2), target_shape=(16,16), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(5,5), stride=(2,2), target_shape=(32,32), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')    

    return gout

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

def get_caltech_101():
    #mnist = fetch_mldata('MNIST original')
    #import ipdb; ipdb.set_trace()
    caltech = []
    path = '/home/ubuntu/datasets/caltech101/data/images32x32/'
    path_wo_ext = '/home/ubuntu/datasets/caltech101/data/images/'
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            caltech.append(img)
    
    caltech = np.asarray(caltech)
    
    caltech = caltech.astype(np.float32)/(255.0/2) - 1.0
    caltech = caltech.reshape((caltech.shape[0], 1, 32, 32)) 
    
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(caltech.shape[0])
    X_train = caltech[p]

    return X_train

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

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    #buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    local_out = 1
    num = 1
    cv2.imwrite('%s.jpg' % (title), buff)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # =============setting============
    dataset = 'caltech'
    imgnet_path = './train.rec'
    nef = 64
    ndf = 64
    ngf = 64
    nc = 1
    batch_size = 64
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    epsilon = 1e-5
    ctx = mx.gpu(1)
    check_point = True
    g_dl_weight = 1e-1
    
    #encoder
    z_mu, z_lv, z = encoder(nef, Z)
    symE = mx.sym.Group([z_mu, z_lv, z])    
    
    #sampler
    #batch_size = z_mu.shape[0]


    #z = GaussianSampleLayer(z_mu, z_lv)    
    
    #generator
    symG = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12)
    
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
    if dataset == 'caltech':
        X_train = get_caltech_101()
        #import ipdb; ipdb.set_trace()
        train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size, shuffle=True)
    elif dataset == 'imagenet':
        train_iter = ImagenetIter(imgnet_path, batch_size, (3, 32, 32))
    
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
    
    # =============module DL=============    
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
       
    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()
    
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
    for epoch in range(200):
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

            # update generator twice            
            #1
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

            #2
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
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get(), mE.get(), KLloss[0].asnumpy(), DLloss[0].asnumpy())
                mACC.reset()
                mG.reset()
                mD.reset()
                mE.reset()
                
                if epoch % 5 == 0:
                    visual('outputs32x32/gout'+str(epoch), outG[0].asnumpy())
                    #diff = diffD[0].asnumpy()
                    #diff = (diff - diff.mean())/diff.std()
                    #visual('diff', diff)
                    visual('outputs32x32/data'+str(epoch), batch.data[0].asnumpy())

        if check_point and epoch % 5 == 0:
            print('Saving...')
            modG.save_params('checkpoints32x32/%s_G-%04d.params'%(dataset, epoch))
            modD.save_params('checkpoints32x32/%s_D-%04d.params'%(dataset, epoch))
            modE.save_params('checkpoints32x32/%s_E-%04d.params'%(dataset, epoch))
