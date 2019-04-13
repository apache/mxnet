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
"""Generate MXNet implementation of CapsNet
Reference 1: https://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
Reference 2: https://arxiv.org/pdf/1710.09829.pdf
"""
import os
import re
import gzip
import struct
import numpy as np
import scipy.ndimage as ndi
import mxnet as mx
from capsulelayers import primary_caps, CapsuleLayer

from mxboard import SummaryWriter


def margin_loss(y_true, y_pred):
    loss = y_true * mx.sym.square(mx.sym.maximum(0., 0.9 - y_pred)) +\
        0.5 * (1 - y_true) * mx.sym.square(mx.sym.maximum(0., y_pred - 0.1))
    return mx.sym.mean(data=mx.sym.sum(loss, 1))


def capsnet(batch_size, n_class, num_routing, recon_loss_weight):
    """Create CapsNet"""
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
    recon_error_stopped = recon_error
    recon_error_stopped = mx.sym.BlockGrad(recon_error_stopped)
    loss = mx.symbol.MakeLoss((1-recon_loss_weight)*margin_loss(y_onehot, out_caps)+recon_loss_weight*recon_error)

    out_caps_blocked = out_caps
    out_caps_blocked = mx.sym.BlockGrad(out_caps_blocked)
    return mx.sym.Group([out_caps_blocked, loss, recon_error_stopped])


def download_data(url, force_download=False):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        mx.test_utils.download(url, fname)
    return fname


def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8)
        np.reshape(image, len(label), (rows, cols))
    return label, image


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


class LossMetric(mx.metric.EvalMetric):
    """Evaluate the loss function"""
    def __init__(self, batch_size, num_gpus):
        super(LossMetric, self).__init__('LossMetric')
        self.batch_size = batch_size
        self.num_gpu = num_gpus
        self.sum_metric = 0
        self.num_inst = 0
        self.loss = 0.0
        self.batch_sum_metric = 0
        self.batch_num_inst = 0
        self.batch_loss = 0.0
        self.recon_loss = 0.0
        self.n_batch = 0

    def update(self, labels, preds):
        """Update the hyper-parameters and loss of CapsNet"""
        batch_sum_metric = 0
        batch_num_inst = 0
        for label, pred_outcaps in zip(labels[0], preds[0]):
            label_np = int(label.asnumpy())
            pred_label = int(np.argmax(pred_outcaps.asnumpy()))
            batch_sum_metric += int(label_np == pred_label)
            batch_num_inst += 1
        batch_loss = preds[1].asnumpy()
        recon_loss = preds[2].asnumpy()
        self.sum_metric += batch_sum_metric
        self.num_inst += batch_num_inst
        self.loss += batch_loss
        self.recon_loss += recon_loss
        self.batch_sum_metric = batch_sum_metric
        self.batch_num_inst = batch_num_inst
        self.batch_loss = batch_loss
        self.n_batch += 1

    def get_name_value(self):
        acc = float(self.sum_metric)/float(self.num_inst)
        mean_loss = self.loss / float(self.n_batch)
        mean_recon_loss = self.recon_loss / float(self.n_batch)
        return acc, mean_loss, mean_recon_loss

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
        self.recon_loss = 0.0
        self.n_batch = 0


class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """

    def __init__(self, learning_rate=0.001):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate


def do_training(num_epoch, optimizer, kvstore, learning_rate, model_prefix, decay):
    """Perform CapsNet training"""
    summary_writer = SummaryWriter(args.tblog_dir)
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
        train_acc, train_loss, train_recon_err = loss_metric.get_name_value()
        loss_metric.reset()
        for n_batch, data_batch in enumerate(val_iter):
            module.forward(data_batch)
            module.update_metric(loss_metric, data_batch.label)
            loss_metric.get_batch_log(n_batch)
        val_acc, val_loss, val_recon_err = loss_metric.get_name_value()

        summary_writer.add_scalar('train_acc', train_acc, n_epoch)
        summary_writer.add_scalar('train_loss', train_loss, n_epoch)
        summary_writer.add_scalar('train_recon_err', train_recon_err, n_epoch)
        summary_writer.add_scalar('val_acc', val_acc, n_epoch)
        summary_writer.add_scalar('val_loss', val_loss, n_epoch)
        summary_writer.add_scalar('val_recon_err', val_recon_err, n_epoch)

        print('Epoch[%d] train acc: %.4f loss: %.6f recon_err: %.6f' % (n_epoch, train_acc, train_loss,
                                                                        train_recon_err))
        print('Epoch[%d] val acc: %.4f loss: %.6f recon_err: %.6f' % (n_epoch, val_acc, val_loss, val_recon_err))
        print('SAVE CHECKPOINT')

        module.save_checkpoint(prefix=model_prefix, epoch=n_epoch)
        n_epoch += 1
        lr_scheduler.learning_rate = learning_rate * (decay ** n_epoch)


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.):
    """Apply transform on nd.array"""
    x = np.rollaxis(x, 0, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, 0 + 1)
    return x


def random_shift(x, width_shift_fraction, height_shift_fraction):
    tx = np.random.uniform(-height_shift_fraction, height_shift_fraction) * x.shape[2]
    ty = np.random.uniform(-width_shift_fraction, width_shift_fraction) * x.shape[1]
    shift_matrix = np.array([[1, 0, tx],
                             [0, 1, ty],
                             [0, 0, 1]])
    x = apply_transform(x, shift_matrix, 'nearest')
    return x


def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for idx_k, idx_v in data:
        shuffle_data.append((idx_k, mx.ndarray.array(idx_v.asnumpy()[idx], idx_v.context)))

    return shuffle_data


class MNISTCustomIter(mx.io.NDArrayIter):
    """Create custom iterator of mnist dataset"""
    def __init__(self, data, label, batch_size, shuffle):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = None

    def reset(self):
        """Reset class MNISTCustomIter(mx.io.NDArrayIter):"""
        # shuffle data
        if self.is_train:
            np.random.shuffle(self.idx)
            self.data = _shuffle(self.data, self.idx)
            self.label = _shuffle(self.label, self.idx)

        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor % self.num_data) % self.batch_size
        else:
            self.cursor = -self.batch_size

    def set_is_train(self, is_train):
        """Set training flag"""
        self.is_train = is_train

    def next(self):
        """Generate next of iterator"""
        if self.iter_next():
            if self.is_train:
                data_raw_list = self.getdata()
                data_shifted = []
                for data_raw in data_raw_list[0]:
                    data_shifted.append(random_shift(data_raw.asnumpy(), 0.1, 0.1))
                return mx.io.DataBatch(data=[mx.nd.array(data_shifted)], label=self.getlabel(),
                                       pad=self.getpad(), index=None)
            else:
                return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(), pad=self.getpad(), index=None)

        else:
            raise StopIteration


if __name__ == "__main__":
    # Read mnist data set
    path = 'http://yann.lecun.com/exdb/mnist/'
    (train_lbl, train_img) = read_data(path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

    # set batch size
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--devices', default='gpu0', type=str)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_routing', default=3, type=int)
    parser.add_argument('--model_prefix', default='capsnet', type=str)
    parser.add_argument('--decay', default=0.9, type=float)
    parser.add_argument('--tblog_dir', default='tblog', type=str)
    parser.add_argument('--recon_loss_weight', default=0.392, type=float)
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print("{0}: {1}".format(k, v))
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
    train_iter = MNISTCustomIter(data=to4d(train_img), label=train_lbl, batch_size=int(args.batch_size), shuffle=True)
    train_iter.set_is_train(True)
    val_iter = MNISTCustomIter(data=to4d(val_img), label=val_lbl, batch_size=int(args.batch_size), shuffle=True)
    val_iter.set_is_train(False)
    # define capsnet
    final_net = capsnet(batch_size=int(args.batch_size/num_gpu),
                        n_class=10,
                        num_routing=args.num_routing,
                        recon_loss_weight=args.recon_loss_weight)
    # set metric
    loss_metric = LossMetric(args.batch_size/num_gpu, 1)

    # run model
    module = mx.mod.Module(symbol=final_net, context=contexts, data_names=('data',), label_names=('softmax_label',))
    module.bind(data_shapes=train_iter.provide_data,
                label_shapes=val_iter.provide_label,
                for_training=True)

    do_training(num_epoch=args.num_epoch, optimizer='adam', kvstore='device', learning_rate=args.lr,
                model_prefix=args.model_prefix, decay=args.decay)
