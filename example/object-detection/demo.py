"""Train Gluon Object-Detection models."""
import os
import argparse
import mxnet as mx
from predict import predict_ssd

def parse_args():
    parser = argparse.ArgumentParser(description='Train a gluon detection network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', dest='algorithm', type=str, default='ssd',
                        help='which network to use')
    parser.add_argument('--data-shape', dest='data_shape', type=str, default='512',
                        help='image data shape, can be int or tuple')
    parser.add_argument('--model', dest='model', type=str, default='resnet50_v1',
                        help='base network to use, choices are models from gluon model_zoo')
    parser.add_argument('--dataset', dest='dataset', type=str, default='voc',
                        help='which dataset to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='Whether use pretrained models. '
                        ' 0: from scratch, 1: use base model, 2: use pretrained detection model')
    parser.add_argument('--prefix', dest='prefix', type=str, help='new model prefix',
                        default=os.path.join(os.path.dirname(__file__), 'model', 'default'))
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # choose algorithm
    if args.algorithm.lower() == 'ssd':
        model = '_'.join([args.algorithm, args.data_shape, args.model])
        predict_ssd.predict_net(args.images, args.model, args.data_shape, num_class=20)
    else:
        raise NotImplementedError("Training algorithm {} not supported.".format(args.algorithm))
