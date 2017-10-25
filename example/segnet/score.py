import argparse
import mxnet as mx
import time
import os
import logging
import pdb
import numpy as np
from PIL import Image
from data_iter import FileIter
from common import contrib_metrics

def getpallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    """
    pallete = [0] * (num_cls * 3)
    for j in xrange(0, num_cls):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def score(args):
    """
    score segnet
    """
    pallete = getpallete(256)
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.log_file:
        file_path = logging.FileHandler(args.log_file)
        logger.addHandler(file_path)

    data = FileIter(
        batch_size=args.batch_size,
        root_dir=args.data_path,
        flist_name=args.score_file,
        rgb_mean=[float(i) for i in args.rgb_mean.split(',')],
        shuffle=True)
    # load model
    from importlib import import_module
    network = import_module('symbols.'+args.network).get_symbol(args.num_classes)

    try:
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
    except ():
        logging.info('model load error!')

    # create module
    if args.gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    mod = mx.mod.Module(symbol=network, context=devs, label_names=['softmax_label',])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    # evaluation metrices
    metric = contrib_metrics.Accuracy(ignore_label=11)
    tic = time.time()
    num = 0
    data.reset()
    metric.reset()
    for batch in data:
        mod.forward(batch, is_train=False)
        out_img = np.uint8(np.squeeze(mod.get_outputs()[0].asnumpy().argmax(axis=1)))
        out_img = Image.fromarray(out_img)
        out_img.putpalette(pallete)
        out_img.save('res_pic/' + str(num) + 'res.png')
        mod.update_metric(metric, batch.label)
        logging.info(metric.get())
        num += args.batch_size
    logging.info(metric.get())
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rgb_mean', type=str, default='123.68, 116.779, 103.939')
    parser.add_argument('--image_shape', type=str, default='3,360,480')
    parser.add_argument('--data_nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    parser.add_argument('--data_path', type=str, default="/data/CamVid",
                        help='data path')
    parser.add_argument('--score_file', type=str, default="test.txt",
                        help='test data list file name in data path')
    parser.add_argument('--model_prefix', type=str, default='models/segnet',
                        help='model prefix')
    parser.add_argument('--load_epoch', type=int, default=200,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--log_file', type=str, default="log.txt",
                        help='the name of log file')
    parser.add_argument('--network', type=str, required=True,
                        help='the neural network to use')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='num classes to clissufy')

    args = parser.parse_args()

    (speed,) = score(args)
    logging.info('Finished with %f images per second', speed)
