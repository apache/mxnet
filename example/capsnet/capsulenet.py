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

import mxnet as mx
import numpy as np
import os
import re
import urllib
import gzip
import struct
from capsulelayers import primary_caps, CapsuleLayer


def margin_loss(y_true, y_pred):
    loss = y_true * mx.sym.square(mx.sym.maximum(0., 0.9 - y_pred)) +\
        0.5 * (1 - y_true) * mx.sym.square(mx.sym.maximum(0., y_pred - 0.1))
    return mx.sym.mean(data=mx.sym.sum(loss, 1))


def capsnet(batch_size, n_class, num_routing):
    # data.shape = [batch_size, 1, 28, 28]
    data = mx.sym.Variable('data')

    input_shape = (1, 28, 28)
    # Conv2D layer
    # net.shape = [batch_size, 256, 20, 20]
    conv1 = mx.sym.Convolution(data=data,
                               num_filter=256,
                               kernel=(9, 9),
                               layout='NCHW',
                               name='conv1')
    conv1 = mx.sym.Activation(data=conv1, act_type='relu', name='conv1_act')
    # net.shape = [batch_size, 256, 6, 6]

    primarycaps = primary_caps(data=conv1,
                               dim_vector=8,
                               n_channels=32,
                               kernel=(9, 9),
                               strides=[2, 2],
                               name='primarycaps')
    primarycaps.infer_shape(data=(batch_size, 1, 28, 28))
    # CapsuleLayer
    kernel_initializer = mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3)
    bias_initializer = mx.init.Zero()
    digitcaps = CapsuleLayer(num_capsule=10,
                             dim_vector=16,
                             batch_size=batch_size,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             num_routing=num_routing)(primarycaps)

    # out_caps : (batch_size, 10)
    out_caps = mx.sym.sqrt(data=mx.sym.sum(mx.sym.square(digitcaps), 2))
    out_caps.infer_shape(data=(batch_size, 1, 28, 28))

    y = mx.sym.Variable('softmax_label', shape=(batch_size,))
    y_onehot = mx.sym.one_hot(y, n_class)
    y_reshaped = mx.sym.Reshape(data=y_onehot, shape=(batch_size, -4, n_class, -1))
    y_reshaped.infer_shape(softmax_label=(batch_size,))

    # inputs_masked : (batch_size, 16)
    inputs_masked = mx.sym.linalg_gemm2(y_reshaped, digitcaps, transpose_a=True)
    inputs_masked = mx.sym.Reshape(data=inputs_masked, shape=(-3, 0))
    x_recon = mx.sym.FullyConnected(data=inputs_masked, num_hidden=512, name='x_recon')
    x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act')
    x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=1024, name='x_recon2')
    x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act2')
    x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=np.prod(input_shape), name='x_recon3')
    x_recon = mx.sym.Activation(data=x_recon, act_type='sigmoid', name='x_recon_act3')

    data_flatten = mx.sym.flatten(data=data)
    squared_error = mx.sym.square(x_recon-data_flatten)
    recon_error = mx.sym.mean(squared_error)
    loss = mx.symbol.MakeLoss((1-0.392)*margin_loss(y_onehot, out_caps)+0.392*recon_error)

    out_caps_blocked = out_caps
    out_caps_blocked = mx.sym.BlockGrad(out_caps_blocked)
    return mx.sym.Group([out_caps_blocked, loss])


def download_data(url, force_download=False):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname


def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return label, image


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


class LossMetric(mx.metric.EvalMetric):
    def __init__(self, batch_size, num_gpu):
        super(LossMetric, self).__init__('LossMetric')
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.sum_metric = 0
        self.num_inst = 0
        self.loss = 0.0
        self.batch_sum_metric = 0
        self.batch_num_inst = 0
        self.batch_loss = 0.0

    def update(self, labels, preds):
        batch_sum_metric = 0
        batch_num_inst = 0
        for label, pred_outcaps in zip(labels[0], preds[0]):
            label_np = int(label.asnumpy())
            pred_label = int(np.argmax(pred_outcaps.asnumpy()))
            batch_sum_metric += int(label_np == pred_label)
            batch_num_inst += 1
        batch_loss = preds[1].asnumpy()
        self.sum_metric += batch_sum_metric
        self.num_inst += batch_num_inst
        self.loss += batch_loss
        self.batch_sum_metric = batch_sum_metric
        self.batch_num_inst = batch_num_inst
        self.batch_loss = batch_loss

    def get_name_value(self):
        acc = float(self.sum_metric)/float(self.num_inst)
        mean_loss = self.loss / float(self.num_inst)
        return acc, mean_loss

    def get_batch_log(self, n_batch):
        print("n_batch :"+str(n_batch)+" batch_acc:" +
              str(float(self.batch_sum_metric) / float(self.batch_num_inst)) +
              ' batch_loss:' + str(float(self.batch_loss)/float(self.batch_num_inst)))
        self.batch_sum_metric = 0
        self.batch_num_inst = 0
        self.batch_loss = 0.0

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0
        self.loss = 0.0


class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """

    def __init__(self, learning_rate=0.001):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate


def do_training(num_epoch, optimizer, kvstore, learning_rate):
    lr_scheduler = SimpleLRScheduler(learning_rate)
    optimizer_params = {'lr_scheduler': lr_scheduler}
    module.init_params()
    module.init_optimizer(kvstore=kvstore,
                          optimizer=optimizer,
                          optimizer_params=optimizer_params)
    n_epoch = 0
    while True:
        if n_epoch >= num_epoch:
            break
        train_iter.reset()
        val_iter.reset()
        loss_metric.reset()
        for n_batch, data_batch in enumerate(train_iter):
            module.forward_backward(data_batch)
            module.update()
            module.update_metric(loss_metric, data_batch.label)
            loss_metric.get_batch_log(n_batch)
        train_acc, train_loss = loss_metric.get_name_value()
        loss_metric.reset()
        for n_batch, data_batch in enumerate(val_iter):
            module.forward(data_batch)
            module.update_metric(loss_metric, data_batch.label)
            loss_metric.get_batch_log(n_batch)
        val_acc, val_loss = loss_metric.get_name_value()
        print('Epoch[' + str(n_epoch) + '] train acc:' + str(train_acc) + ' loss:' + str(train_loss))
        print('Epoch[' + str(n_epoch) + '] val acc:' + str(val_acc) + ' loss:' + str(val_loss))
        print('SAVE CHECKPOINT')

        module.save_checkpoint(prefix='capsnet', epoch=n_epoch)
        n_epoch += 1
        lr_scheduler.learning_rate = learning_rate * (0.9 ** n_epoch)

if __name__ == "__main__":
    # Read mnist data set
    path = 'http://yann.lecun.com/exdb/mnist/'
    (train_lbl, train_img) = read_data(
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')
    # set batch size
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--devices', default='gpu0', type=str)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_routing', default=3, type=int)
    args = parser.parse_args()
    contexts = re.split(r'\W+', args.devices)
    for i, ctx in enumerate(contexts):
        if ctx[:3] == 'gpu':
            contexts[i] = mx.context.gpu(int(ctx[3:]))
        else:
            contexts[i] = mx.context.cpu()
    num_gpu = len(contexts)

    if args.batch_size % num_gpu != 0:
        raise Exception('num_gpu should be positive divisor of batch_size')

    # generate train_iter, val_iter
    train_iter = mx.io.NDArrayIter(data=to4d(train_img), label=train_lbl, batch_size=args.batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(data=to4d(val_img), label=val_lbl, batch_size=args.batch_size,)

    # define capsnet
    final_net = capsnet(batch_size=args.batch_size/num_gpu, n_class=10, num_routing=args.num_routing)

    # set metric
    loss_metric = LossMetric(args.batch_size/num_gpu, 1)

    # run model
    module = mx.mod.Module(symbol=final_net, context=contexts, data_names=('data',), label_names=('softmax_label',))
    module.bind(data_shapes=train_iter.provide_data,
                label_shapes=val_iter.provide_label,
                for_training=True)
    do_training(num_epoch=args.num_epoch, optimizer='adam', kvstore='device', learning_rate=args.lr)
