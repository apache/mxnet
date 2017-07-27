from __future__ import print_function

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection
from data import det_mnist_iterator


# CLI
parser = argparse.ArgumentParser(description='Train a toy object detector.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus to use, empty string to use CPU.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning Rate. default is 0.1.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='Number of batches to wait before logging.')
parser.add_argument('--demo', default=False, action="store_true",
                    help='Skip training, load pretrained for demo.')
args = parser.parse_args()
print(args)
try:
    ctx = [mx.gpu(int(x)) for x in args.gpus.strip().split(',')]
except ValueError:
    ctx = [mx.cpu()]
print('Using context: ', ctx)

# get toy dataset
train_iter, val_iter, data_shape, cls_names = det_mnist_iterator(args.batch_size)
num_class = len(cls_names)
print('Classes: ', num_class, cls_names)

# define detection toy network
class ToySSD(gluon.Block):
    """Build a toy network for simple object detection task."""
    def __init__(self, num_class, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # sizes control the scale of anchor boxes, with decreasing feature map size,
        # the anchor boxes are expected to be larger in design
        self.sizes = [[.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # ratios control the aspect ratio of anchor boxes, here we use 1, 2, 0.5
        self.ratios = [[1,2,.5], [1,2,.5], [1,2,.5], [1,2,.5]]
        num_anchors = [len(x) + len(y) - 1 for x, y in zip(self.sizes, self.ratios)]
        self.num_anchors = num_anchors
        self.num_class = num_class
        with self.name_scope():
            # first build a body as feature
            self.body = nn.HybridSequential()
            # 64 x 64
            # make basic block is a stack of sequential conv layers, followed by
            # a pooling layer to reduce feature map size
            self.body.add(self._make_basic_block(16))
            # 32 x 32
            self.body.add(self._make_basic_block(32))
            # 16 x 16
            self.body.add(self._make_basic_block(64))
            # 8 x 8
            # use cls1 conv layer to get the class predictions on 8x8 feature map
            # use loc1 conv layer to get location offsets on 8x8 feature map
            # use blk1 conv block to reduce the feature map size again
            self.cls1 = nn.Conv2D(num_anchors[0] * (num_class + 1), 3, padding=1)
            self.loc1 = nn.Conv2D(num_anchors[0] * 4, 3, padding=1)
            self.blk1 = self._make_basic_block(64)
            # 4 x 4
            self.cls2 = nn.Conv2D(num_anchors[1] * (num_class + 1), 3, padding=1)
            self.loc2 = nn.Conv2D(num_anchors[1] * 4, 3, padding=1)
            self.blk2 = self._make_basic_block(64)
            # 2 x 2
            self.cls3 = nn.Conv2D(num_anchors[2] * (num_class + 1), 3, padding=1)
            self.loc3 = nn.Conv2D(num_anchors[2] * 4, 3, padding=1)
            # 1 x 1
            self.cls4 = nn.Conv2D(num_anchors[3] * (num_class + 1), 3, padding=1)
            self.loc4 = nn.Conv2D(num_anchors[3] * 4, 3, padding=1)

    def _make_basic_block(self, num_filter):
        """Basic block is a stack of sequential convolution layers, followed by
        a pooling layer to reduce feature map. """
        out = nn.HybridSequential()
        out.add(nn.Conv2D(num_filter, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filter))
        out.add(nn.Activation('relu'))
        out.add(nn.Conv2D(num_filter, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filter))
        out.add(nn.Activation('relu'))
        out.add(nn.MaxPool2D())
        return out

    def forward(self, x):
        anchors = []
        loc_preds = []
        cls_preds = []
        x = self.body(x)
        # 8 x 8, generate anchors, predict class and location offsets with conv layer
        # transpose, reshape and append to list for further concatenation
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[0], ratios=self.ratios[0]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc1(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls1(x), axes=(0, 2, 3, 1))))
        x = self.blk1(x)
        # 4 x 4
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[1], ratios=self.ratios[1]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc2(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls2(x), axes=(0, 2, 3, 1))))
        x = self.blk2(x)
        # 2 x 2
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[2], ratios=self.ratios[2]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc3(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls3(x), axes=(0, 2, 3, 1))))
        # we use pooling directly here without convolution layers
        x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(2, 2))
        # 1 x 1
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[3], ratios=self.ratios[3]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc4(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls4(x), axes=(0, 2, 3, 1))))
        # concat multiple layers
        anchors = nd.reshape(nd.concat(*anchors, dim=1), shape=(0, -1, 4))
        loc_preds = nd.concat(*loc_preds, dim=1)
        cls_preds = nd.reshape(nd.concat(*cls_preds, dim=1), (0, -1, self.num_class+1))
        cls_preds = nd.transpose(cls_preds, axes=(0, 2, 1))
        return [anchors, cls_preds, loc_preds]

# create network
net = ToySSD(num_class)

def test(val_iter):
    """Evaluation workflow"""
    import sys
    sys.path.append('../../ssd/evaluate')
    from eval_metric import MApMetric
    metric = MApMetric(class_names=cls_names)
    val_iter.reset()
    for batch in val_iter:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = batch.label[0]
        outputs = []
        for x in data:
            # network taks image as input
            # network return anchors, class predictions, loc predictions
            anchors, cls_preds, loc_preds = net(x)
            # pass raw predictions to softmax
            cls_probs = nd.SoftmaxActivation(cls_preds, mode='channel')
            outputs.append(MultiBoxDetection(*[cls_probs, loc_preds, anchors]))
        output = mx.nd.concat(*outputs, dim=0)
        metric.update([label], [output])
    return metric.get()

def train(epochs):
    """Training workflow."""
    # initialize all parameters
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
    # use gluon.Trainer to ease the training process
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr})
    # use accuracy as class prediction metric
    cls_metric = mx.metric.Accuracy()
    # use mean absolute error as metric for location predictions
    loc_metric = mx.metric.MAE()
    # use softmax entropy loss for class predictions
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # use SmoothL1Loss for location predictions which is smoother than L2 loss
    loc_loss = gluon.loss.SmoothL1Loss()

    for epoch in range(epochs):
        # reset iterators and tick
        tic = time.time()
        train_iter.reset()
        val_iter.reset()
        cls_metric.reset()
        loc_metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_iter):
            # split data for multiple devices
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            cls_pred_outputs = []
            cls_target_outputs = []
            loc_pred_outputs = []
            loc_target_outputs = []
            Ls = []
            # record the gradients flow
            with ag.record():
                for x, y in zip(data, label):
                    # the detection network generate 3 parts
                    # anchors, class preds, loc preds
                    anchors, cls_preds, loc_preds = net(x)
                    # use ground-truths to generate the training targets
                    z = MultiBoxTarget(*[anchors, y, cls_preds])
                    # loc offset target
                    loc_target = z[0]
                    # mask is used to mask out predictions we don't want to penalize
                    loc_mask = z[1]
                    # cls_target are class labels for all anchors boxes
                    cls_target = z[2]
                    # loss1 is the loss for class predictions
                    loss1 = cls_loss(nd.transpose(cls_preds, (0, 2, 1)), cls_target)
                    # loss2 is the loss for location predictions
                    loss2 = loc_loss(loc_preds * loc_mask, loc_target)
                    # here use loss1 and loss2 evenly, without weight
                    loss = loss1 + loss2
                    # combine results on multiple devices
                    Ls.append(loss)
                    cls_pred_outputs.append(cls_preds)
                    cls_target_outputs.append(cls_target)
                    loc_pred_outputs.append(loc_preds * loc_mask)
                    loc_target_outputs.append(loc_target)
                for L in Ls:
                    L.backward()
            # apply gradients according to learning_rate and other optimize params
            trainer.step(batch.data[0].shape[0])
            # update metrics
            cls_metric.update(cls_target_outputs, cls_pred_outputs)
            loc_metric.update(loc_target_outputs, loc_pred_outputs)
            if args.log_interval and i % args.log_interval == 0:
                name1, acc = cls_metric.get()
                name2, l1 = loc_metric.get()
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
		              %(epoch ,i, args.batch_size/(time.time()-btic), name1, acc, name2, l1))
            btic = time.time()

        # end of epoch logging
        name1, acc = cls_metric.get()
        name2, l1 = loc_metric.get()
        print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, acc, name2, l1))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

        # evaluation
        name, val_score = test(val_iter)
        if not isinstance(name, list):
            name = [name]
            val_score = [val_score]
        for k, v in zip(name, val_score):
            print('[Epoch %d] validation: %s=%f'%(epoch, k, v))

        # save parameters to file
        net.collect_params().save('ssd_%d.params' % (epoch + 1))

def visualize_detection(img, dets, classes=[], thresh=0.6):
    """
    visualize detections in one image

    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Require matplotlib.pyplot for ploting, skip...')
        return
    import random
    plt.clf()
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = dets[i, 1]
            if score > thresh:
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3.5)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                plt.gca().text(xmin, ymin - 2,
                                '{:s} {:.3f}'.format(class_name, score),
                                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                fontsize=12, color='white')
    plt.show()

def demo(epoch, test_iter, thresh=0.1, limit=10):
    """Run in inference mode and display the results."""
    # load pre-trained parameters
    net.collect_params().load('ssd_%d.params' % epoch, ctx)
    test_iter.reset()
    count = 0
    for batch in test_iter:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            anchors, cls_preds, loc_preds = net(x)
            # pass raw predictions to softmax
            cls_probs = nd.SoftmaxActivation(cls_preds, mode='channel')
            # combine results, apply non-maximum-suppression, etc...
            outputs.append(MultiBoxDetection(*[cls_probs, loc_preds, anchors],
                                             force_suppress=True))
        # combine results on multiple devices
        output = mx.nd.concat(*outputs, dim=0)
        for k, out in enumerate(output.asnumpy()):
            img = batch.data[0][k].asnumpy().transpose((1, 2, 0)) * 255
            img = img[:, :, (0, 0, 0)]
            img = img.astype(np.uint8)
            if count > limit:
                return
            # display
            visualize_detection(img, out, cls_names, thresh=thresh)
            count += 1

if __name__ == '__main__':
    if not args.demo:
        train(args.epochs)
    demo(args.epochs, val_iter, thresh=0.1, limit=10)
