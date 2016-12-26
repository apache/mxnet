from __future__ import print_function
import find_mxnet
import mxnet as mx
import importlib
import argparse
import sys

parser = argparse.ArgumentParser(description='network visualization')
parser.add_argument('--network', type=str, default='vgg16_reduced',
                    choices = ['vgg16_reduced'],
                    help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=300,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

sys.path.append('../symbol')

if not args.train:
    net = importlib.import_module("symbol_" + args.network).get_symbol(args.num_classes)
    a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape)}, \
        node_attrs={"shape":'rect', "fixedsize":'false'})
    a.render("ssd_" + args.network)
else:
    net = importlib.import_module("symbol_" + args.network).get_symbol_train(args.num_classes)
    print(net.tojson())
