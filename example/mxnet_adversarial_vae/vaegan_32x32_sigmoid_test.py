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
from scipy.io import savemat
#from layer import GaussianSampleLayer

###########################################################################
# test the VAE and saves the output images and embeddings  

###########################################################################

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
def encoder(nef, z_dim, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
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
def generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, z_dim=100):
    
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
    gout = mx.sym.Activation(g5, name='genact5', act_type='sigmoid')    

    return gout

#######################################################################
# Get test dataset
####################################################################### 
def get_data():
    #mnist = fetch_mldata('MNIST original')
    #import ipdb; ipdb.set_trace()  
    
    dataset = []
    image_names = []
    path = '/home/ubuntu/datasets/MPEG7dataset/images/'
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)
        image_names.append(filename)      
        if img is not None:
            dataset.append(img)
    
    #import ipdb; ipdb.set_trace()      
    dataset = np.asarray(dataset)
    print('dataset.shape = ', dataset.shape)
        
    dataset = dataset.astype(np.float32)/(255.0)
    dataset = dataset.reshape((dataset.shape[0], 1, 32, 32)) 
    
    #np.random.seed(1234) # set seed for deterministic ordering
    #p = np.random.permutation(caltech.shape[0])
    #X_train = caltech[p]
    X_train = dataset

    return X_train, image_names

#######################################################################
# fill the buffer with the values from the img
####################################################################### 
def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

#######################################################################
# visualize or store an image
####################################################################### 
def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X)*(255.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    #buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    local_out = 1
    num = 1
    cv2.imwrite('%s.jpg' % (title), buff)


#######################################################################
# testing the VAE
####################################################################### 
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # =============setting============
    # encoder filter count in the first layer
    nef = 64
    # discriminator filter count in the first layer
    ndf = 64
    # generator filter count 
    ngf = 64    
    # generator filter count in the last layer i.e. 1 for grayscale image, 3 for RGB image
    nc = 1
    
    batch_size = 1
    
    #embedding size
    Z = 100
    
    #learning rate
    lr = 0.0002
    
    # optimizer params
    beta1 = 0.5
    epsilon = 1e-5
    
    # gpu context 
    ctx = mx.gpu(15)
    
    # checkpoint saving flags
    check_point = True
    
    # discriminator layer loss weight
    g_dl_weight = 1e-1
    
    embedding_path = 'embeddings_test/'
    
    #encoder
    z_mu, z_lv, z = encoder(nef, Z)
    symE = mx.sym.Group([z_mu, z_lv, z])    
    
    #sampler
    #batch_size = z_mu.shape[0]


    #z = GaussianSampleLayer(z_mu, z_lv)    
    
    #generator
    symG = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12)    
    
    #symG, symD = make_dcgan_sym(nef, ngf, ndf, nc)
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    X_train, image_names = get_data()
    #import ipdb; ipdb.set_trace()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size, shuffle=False)

    # =============module E=============
    modE = mx.mod.Module(symbol=symE, data_names=('data',), label_names=None, context=ctx)
    modE.bind(data_shapes=train_iter.provide_data)
    #modE.init_params(initializer=mx.init.Normal(0.02))
    modE.load_params('checkpoints32x32_sigmoid/caltech_E-0045.params')

    # =============module G============= 
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=[('rand', (1, Z, 1, 1))])
    #modG.init_params(initializer=mx.init.Normal(0.02))        
    modG.load_params('checkpoints32x32_sigmoid/caltech_G-0045.params')

    print('Testing...')
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(1):
        train_iter.reset()
        for t, batch in enumerate(train_iter):

            #update discriminator on decoded
            modE.forward(batch, is_train=False)
            mu, lv, z = modE.get_outputs()
            #z = GaussianSampleLayer(mu, lv)          
            mu = mu.reshape((batch_size, Z, 1, 1))
            sample = mx.io.DataBatch([mu], label=None, provide_data = [('rand', (batch_size, Z, 1, 1))])         
            modG.forward(sample, is_train=False)
            outG = modG.get_outputs()                              
                
            visual('outputs_test/gout'+str(epoch), outG[0].asnumpy())
            visual('outputs_test/data'+str(epoch), batch.data[0].asnumpy())
            image_name = image_names[t].split('.')[0]
            savemat(embedding_path+image_name+'.mat', {'embedding':mu.asnumpy()})
