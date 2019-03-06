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

# This example is inspired from
# https://github.com/d2l-ai/d2l-en/blob/master/chapter_convolutional-neural-networks/alexnet.md
# Model definition is from
# https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/alexnet.py


import os
import sys
import argparse
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, data
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.estimator import estimator, event_handler

def parse_args():
    '''
    Command Line Interface
    '''
    parser = argparse.ArgumentParser(description='Train ResNet18 on Fashion-MNIST')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of training epochs.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate. default is 0.001')
    parser.add_argument('-j', '--num-workers', default=None, type=int,
                        help='number of preprocessing workers')
    opt = parser.parse_args()
    return opt

class AlexNet(HybridBlock):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

def load_data_mnist(batch_size, resize=None, num_workers=None,
                    root=os.path.join('~', '.mxnet', 'datasets', 'mnist')):
    '''
    Load MNIST dataset
    '''
    root = os.path.expanduser(root)  # Expand the user path '~'.
    transformer = []
    if resize:
        transformer += [data.vision.transforms.Resize(resize)]
    transformer += [data.vision.transforms.ToTensor()]
    transformer = data.vision.transforms.Compose(transformer)
    mnist_train = data.vision.MNIST(root=root, train=True)
    mnist_test = data.vision.MNIST(root=root, train=False)

    if num_workers is None:
        num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = data.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = data.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter


def main():
    # Parse CLI arguments
    opt = parse_args()
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    input_size = opt.input_size
    lr = opt.lr
    num_workers = opt.num_workers
    # Set context
    if mx.context.num_gpus() > 0:
        context = mx.gpu(0)
    else:
        context = mx.cpu()
    # Get AlexNet model
    net = AlexNet(classes=10)
    # Load train and validation data
    train_data, test_data = load_data_mnist(batch_size, resize=input_size,
                                            num_workers=num_workers)
    # Define loss and evaluation metrics
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    # Hybridize and initialize net
    net.hybridize()
    net.initialize(mx.init.MSRAPrelu(), ctx=context)
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
            val_data=test_data,
            epochs=num_epochs,
            batch_size=batch_size,
            event_handlers=[event_handler.LoggingHandler(est, 'alexnet_log', 'alexnet_log')])


if __name__ == '__main__':
    main()
