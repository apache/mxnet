import argparse
import os
import logging
import time
import random
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd as ag
from dataset.dataloader import DataLoader
from dataset import VOCDetection
from dataset import transform
from config import config as cfg
import model_zoo
from block.loss import *
from block.target import *
from block.coder import MultiClassDecoder, NormalizedBoxCenterDecoder
from trainer.metric import Accuracy, SmoothL1, LossRecorder, MultiBoxMetric
from trainer.debugger import super_print, find_abnormal
from evaluation.eval_metric import VOC07MApMetric, MApMetric

def preprocess(filename, data_shape):
    im = mx.image.imread(filename)
    im = mx.image.imresize(im, data_shape[1], data_shape[0])
    im = im.astype('float32')
    im -= mx.nd.array([123, 117, 104])
    im /= mx.nd.array([58, 57, 57])
    im = im.transpose((2, 0, 1))
    im = im.expand_dims(axis=0)
    return im

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
    import matplotlib.pyplot as plt
    import random
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

def predict_net(im_path, model, data_shape, num_class,
              pretrained=0, seed=None, log_file=None, dev=False, ctx=mx.cpu(), **kwargs):
    """Wrapper function for entire training phase.




    """
    data_shape = [int(x) for x in data_shape.split(',')]
    if len(data_shape) == 1:
        data_shape = data_shape * 2

    model = '_'.join(['ssd', str(data_shape[0]), model])

    class_names = 'aeroplane, bicycle, bird, boat, bottle, bus, \
    car, cat, chair, cow, diningtable, dog, horse, motorbike, \
    person, pottedplant, sheep, sofa, train, tvmonitor'.split(',')

    net = model_zoo.get_detection_model(model, pretrained=pretrained, classes=num_class, ctx=ctx)
    net.collect_params().load(os.path.join(os.path.dirname(__file__), '..', 'model', 'ssd.params'), ctx=ctx)

    def ctx_as_list(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        return ctx

    box_decoder = NormalizedBoxCenterDecoder()
    cls_decoder = MultiClassDecoder()

    x = preprocess(im_path, data_shape)
    z = net(x)
    cls_preds, box_preds, anchors = z
    # out1 = mx.nd.contrib.MultiBoxDetection(nd.softmax(cls_preds).transpose((0, 2, 1)), box_preds.reshape((0, -1)), anchors, nms_topk=400)
    # print(out)
    # visualize_detection(mx.image.imread(im_path).asnumpy(), out[0].asnumpy(), class_names, thresh=0.1)
    # raise
    boxes = box_decoder(box_preds, anchors)
    boxes = nd.clip(boxes, 0.0, 1.0)
    cls_ids, scores = cls_decoder(nd.softmax(cls_preds))
    # print(mx.nd.sum(cls_ids > -0.5))
    result = nd.concat(cls_ids.reshape((0, 0, 1)), scores.reshape((0, 0, 1)), boxes, dim=2)
    out = nd.contrib.box_nms(result, topk=400)
    # np.testing.assert_allclose(out1.asnumpy(), out.asnumpy())
    # print(out)
    visualize_detection(mx.image.imread(im_path).asnumpy(), out[0].asnumpy(), class_names, thresh=0.1)
