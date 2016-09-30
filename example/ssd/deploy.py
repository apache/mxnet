import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a trained model to deploy model')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced'], help='which network to use')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_300'), type=str)
    parser.add_argument('--num-class', dest='num_classes', help='number of classes',
                        default=20, type=int)
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    net = importlib.import_module("symbol_" + args.network) \
        .get_symbol(args.num_classes, args.nms_thresh, args.force_nms)
    _, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    # new name
    tmp = args.prefix.rsplit('/', 1)
    save_prefix = '/deploy_'.join(tmp)
    mx.model.save_checkpoint(save_prefix, args.epoch, net, arg_params, aux_params)
    print "Saved model: {}-{:04d}.param".format(save_prefix, args.epoch)
    print "Saved symbol: {}-symbol.json".format(save_prefix)
