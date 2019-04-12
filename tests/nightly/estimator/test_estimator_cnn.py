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

# Test gluon estimator on CNN models

import argparse
import numpy as np
import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.gluon import data
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.model_zoo import vision

def load_data_mnist(batch_size, resize=None, num_workers=4):
    '''
    Load MNIST dataset
    '''
    transformer = []
    if resize:
        transformer += [data.vision.transforms.Resize(resize)]
    transformer += [data.vision.transforms.ToTensor()]
    transformer = data.vision.transforms.Compose(transformer)
    mnist_train = data.vision.MNIST(train=True)
    mnist_test = data.vision.MNIST(train=False)
    train_iter = data.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = data.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

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

def get_net(model_name, context):
    if model_name == 'FCN':
        num_classes = 21
        pretrained_net = vision.resnet18_v2(pretrained=True, ctx=context)
        net = gluon.nn.HybridSequential()
        for layer in pretrained_net.features[:-2]:
            net.add(layer)
        net.add(gluon.nn.Conv2D(num_classes, kernel_size=1),
                gluon.nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
        net[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 64)), ctx=context)
        net[-2].initialize(init=init.Xavier(), ctx=context)
        input_shape = (1, 3, 320, 480)
        label_shape = (1, 320, 480)
        loss_axis = 1
    else:
        net = vision.get_model(model_name, classes=10)
        net.initialize(mx.init.Xavier(), ctx=context)
        input_shape = (1, 1, 224, 224)
        label_shape = 1
        loss_axis = -1
    return net, input_shape, label_shape, loss_axis

def test_estimator_cpu():
    '''
    Test estimator by doing one pass over each model with synthetic data
    '''
    models = ['resnet18_v1',
              'FCN'
              ]
    context = mx.cpu()
    for model_name in models:
        net, input_shape, label_shape, loss_axis = get_net(model_name, context)
        train_dataset = gluon.data.dataset.ArrayDataset(mx.nd.random.uniform(shape=input_shape),
                                                        mx.nd.zeros(shape=label_shape))
        val_dataset = gluon.data.dataset.ArrayDataset(mx.nd.random.uniform(shape=input_shape),
                                                      mx.nd.zeros(shape=label_shape))
        loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=loss_axis)
        train_data = gluon.data.DataLoader(train_dataset, batch_size=1)
        val_data = gluon.data.DataLoader(val_dataset, batch_size=1)
        net.hybridize()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
        # Define estimator
        est = estimator.Estimator(net=net,
                                  loss=loss,
                                  metrics=mx.metric.Accuracy(),
                                  trainer=trainer,
                                  context=context)
        # Call fit()
        est.fit(train_data=train_data,
                val_data=val_data,
                epochs=1)

def test_estimator_gpu():
    '''
    Test estimator by training resnet18_v1 for 5 epochs on MNIST and verify accuracy
    '''
    model_name = 'resnet18_v1'
    batch_size = 128
    num_epochs = 5
    context = mx.gpu(0)
    net, _, _, _ = get_net(model_name, context)
    train_data, test_data = load_data_mnist(batch_size, resize=224)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    net.hybridize()
    acc = mx.metric.Accuracy()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # Define estimator
    est = estimator.Estimator(net=net,
                              loss=loss,
                              metrics=acc,
                              trainer=trainer,
                              context=context)
    # Call fit()
    est.fit(train_data=train_data,
            val_data=test_data,
            epochs=num_epochs)

    assert acc.get()[1] > 0.80

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test gluon estimator')
    parser.add_argument('--type', type=str, default='cpu')
    opt = parser.parse_args()
    if opt.type == 'cpu':
        test_estimator_cpu()
    elif opt.type == 'gpu':
        test_estimator_gpu()
    else:
        raise RuntimeError("Unknown test type")
