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

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
from mxnet.gluon import nn, Block, HybridBlock, Parameter
from mxnet.base import numeric_types
import mxnet.ndarray as F

class InstanceNorm(HybridBlock):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon}
        if in_channels != 0:
            self.in_channels = in_channels
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, x, gamma, beta):
        return F.InstanceNorm(x, gamma, beta,
                           name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        if hasattr(self, 'in_channels'):
            s += ', in_channels={0}'.format(self.in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class ReflectancePadding(HybridBlock):
    def __init__(self, pad_width=None, **kwargs):
        super(ReflectancePadding, self).__init__(**kwargs)
        self.pad_width = pad_width
        
    def forward(self, x):
        return F.pad(x, mode='reflect', pad_width=self.pad_width)

    
class Bottleneck(Block):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=InstanceNorm):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2D(in_channels=inplanes, 
                                            channels=planes * self.expansion,
                                            kernel_size=1, strides=(stride, stride))
        self.conv_block = nn.Sequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes, 
                                 kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(ConvLayer(planes, planes, kernel_size=3, 
                stride=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, 
                                 channels=planes * self.expansion, 
                                 kernel_size=1))
        
    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(Block):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=InstanceNorm):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                      kernel_size=1, stride=1, upsample=stride)
        self.conv_block = nn.Sequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes, 
                                kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, 
                                channels=planes * self.expansion, 
                                kernel_size=1))

    def forward(self, x):
        return  self.residual_layer(x) + self.conv_block(x)


class ConvLayer(Block):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = int(np.floor(kernel_size / 2))
        self.pad = ReflectancePadding(pad_width=(0,0,0,0,padding,padding,padding,padding))
        self.conv2d = nn.Conv2D(in_channels=in_channels, channels=out_channels, 
                                kernel_size=kernel_size, strides=(stride,stride),
                                padding=0)

    def forward(self, x):
        x = self.pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(Block):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
            stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2D(in_channels=in_channels, 
                                channels=out_channels, 
                                kernel_size=kernel_size, strides=(stride,stride),
                                padding=self.reflection_padding)

    def forward(self, x):
        if self.upsample:
            x = F.UpSampling(x, scale=self.upsample, sample_type='nearest')
        out = self.conv2d(x)
        return out


def gram_matrix(y):
    (b, ch, h, w) = y.shape
    features = y.reshape((b, ch, w * h))
    #features_t = F.SwapAxis(features,1, 2)
    gram = F.batch_dot(features, features, transpose_b=True) / (ch * h * w)
    return gram


class GramMatrix(Block):
    def forward(self, x):
        gram = gram_matrix(x)
        return gram

class Net(Block):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, 
                 norm_layer=InstanceNorm, n_blocks=6, gpu_ids=[]):
        super(Net, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        with self.name_scope():
            self.model1 = nn.Sequential()
            self.ins = Inspiration(ngf*expansion)
            self.model = nn.Sequential()

            self.model1.add(ConvLayer(input_nc, 64, kernel_size=7, stride=1))
            self.model1.add(norm_layer(in_channels=64))
            self.model1.add(nn.Activation('relu'))
            self.model1.add(block(64, 32, 2, 1, norm_layer))
            self.model1.add(block(32*expansion, ngf, 2, 1, norm_layer))


            self.model.add(self.model1)
            self.model.add(self.ins)

            for i in range(n_blocks):
                self.model.add(block(ngf*expansion, ngf, 1, None, norm_layer))
        
            self.model.add(upblock(ngf*expansion, 32, 2, norm_layer))
            self.model.add(upblock(32*expansion, 16, 2, norm_layer))
            self.model.add(norm_layer(in_channels=16*expansion))
            self.model.add(nn.Activation('relu'))
            self.model.add(ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1))


    def set_target(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.set_target(G)

    def forward(self, input):
        return self.model(input)


class Inspiration(Block):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.C = C
        self.weight = self.params.get('weight', shape=(1,C,C),
                                      init=mx.initializer.Uniform(),
                                      allow_deferred_init=True)
        self.gram = F.random.uniform(shape=(B, C, C))

    def set_target(self, target):
        self.gram = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = F.batch_dot(F.broadcast_to(self.weight.data(), shape=(self.gram.shape)), self.gram)
        return F.batch_dot(F.SwapAxis(self.P,1,2).broadcast_to((X.shape[0], self.C, self.C)), X.reshape((0,0,X.shape[2]*X.shape[3]))).reshape(X.shape)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'


class Vgg16(Block):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=1, padding=1)
        self.conv1_2 = nn.Conv2D(in_channels=64, channels=64, kernel_size=3, strides=1, padding=1)

        self.conv2_1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=128, channels=128, kernel_size=3, strides=1, padding=1)

        self.conv3_1 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_2 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_3 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)

        self.conv4_1 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

        self.conv5_1 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

    def forward(self, X):
        h = F.Activation(self.conv1_1(X), act_type='relu')
        h = F.Activation(self.conv1_2(h), act_type='relu')
        relu1_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv2_1(h), act_type='relu')
        h = F.Activation(self.conv2_2(h), act_type='relu')
        relu2_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv3_1(h), act_type='relu')
        h = F.Activation(self.conv3_2(h), act_type='relu')
        h = F.Activation(self.conv3_3(h), act_type='relu')
        relu3_3 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv4_1(h), act_type='relu')
        h = F.Activation(self.conv4_2(h), act_type='relu')
        h = F.Activation(self.conv4_3(h), act_type='relu')
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]
