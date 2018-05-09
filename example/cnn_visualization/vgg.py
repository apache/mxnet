import mxnet as mx
from mxnet import gluon

import os
from mxnet.gluon.model_zoo import model_store

from mxnet.initializer import Xavier
from mxnet.gluon.nn import MaxPool2D, Flatten, Dense, Dropout, BatchNorm
from gradcam import Activation, Conv2D

class VGG(mx.gluon.HybridBlock):
    def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)
            self.features.add(Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(Dropout(rate=0.5))
            self.features.add(Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(Dropout(rate=0.5))
            self.output = Dense(classes,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')

    def _make_features(self, layers, filters, batch_norm):
        featurizer = mx.gluon.nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(BatchNorm())
                featurizer.add(Activation('relu'))
            featurizer.add(MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

def get_vgg(num_layers, pretrained=False, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    if pretrained:
        from mxnet.gluon.model_zoo.model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.load_params(get_model_file('vgg%d%s'%(num_layers, batch_norm_suffix),
                                       root=root), ctx=ctx)
    return net

def vgg16(**kwargs):
    return get_vgg(16, **kwargs)

