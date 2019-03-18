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

# Test gluon estimator on CPU using CNN models

import numpy as np
import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.gluon.estimator import estimator, event_handler
from mxnet.gluon.model_zoo import vision

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    Bilinear interpolation using transposed convolution
    https://github.com/d2l-ai/d2l-en/blob/master/chapter_computer-vision/fcn.md
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

def FCN(num_classes=21, ctx=None):
    '''
    FCN model for semantic segmentation
    '''
    pretrained_net = vision.resnet18_v2(pretrained=True, ctx=ctx)

    net = gluon.nn.HybridSequential()
    for layer in pretrained_net.features[:-2]:
        net.add(layer)

    net.add(gluon.nn.Conv2D(num_classes, kernel_size=1),
            gluon.nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
    return net

def test_estimator():
    '''
    Test estimator by doing one pass over each model with synthetic data
    '''
    models = ['resnet18_v1',
              'alexnet',
              'FCN'
              ]
    context = mx.cpu()
    for model_name in models:
        batch_size = 1
        num_epochs = 1
        lr = 0.001
        # Get model and initialize, define loss
        if model_name is 'FCN':
            num_classes = 21
            net = FCN(num_classes=num_classes, ctx=context)
            dataset = gluon.data.dataset.ArrayDataset(mx.nd.random.uniform(shape=(batch_size, 3, 320, 480)),
                                                      mx.nd.zeros(shape=(batch_size, 320, 480)))
            loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
            net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)), ctx=context)
            net[-2].initialize(init=init.Xavier(), ctx=context)
        else:
            net = vision.get_model(model_name, classes=10)
            dataset = gluon.data.dataset.ArrayDataset(mx.nd.random.uniform(shape=(batch_size, 1, 224, 224)),
                                                      mx.nd.zeros(batch_size))
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
            net.initialize(mx.init.Xavier(), ctx=context)

        train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
        # Define evaluation metrics
        acc = mx.metric.Accuracy()
        # Hybridize net
        net.hybridize()
        # Define trainer
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
        # Define estimator
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=acc,
                                  trainers=trainer,
                                  context=context)
        # Call fit() to begin training
        est.fit(train_data=train_data,
                # val_data=test_data,
                epochs=num_epochs,
                batch_size=batch_size)

if __name__ == '__main__':
    test_estimator()
