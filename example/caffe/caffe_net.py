import os, sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../image-classification/"))
import mxnet as mx
from data import get_iterator 
import argparse
import train_model

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.CaffeOperator(data_0 = data, name='fc1', prototxt="layer{ inner_product_param{num_output: 128}}", op_type_string="InnerProduct")
    act1 = mx.symbol.CaffeOperator(data_0 = fc1, op_type_string="Tanh")
    fc2  = mx.symbol.CaffeOperator(data_0 = act1, name='fc2', prototxt="layer{ inner_product_param{num_output: 64}}", op_type_string="InnerProduct")
    act2 = mx.symbol.CaffeOperator(data_0 = fc2, op_type_string="Tanh")
    fc3 = mx.symbol.CaffeOperator(data_0 = act2, name='fc3', prototxt="layer{ inner_product_param{num_output: 10}}", op_type_string="InnerProduct")
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')

    # first conv
    conv1 = mx.symbol.CaffeOperator(data_0=data, prototxt="layer { convolution_param { num_output: 20 kernel_size: 5 stride: 1} }", op_type_string="Conv")
    act1 = mx.symbol.CaffeOperator(data_0=conv1, op_type_string="Tanh")
    pool1 = mx.symbol.CaffeOperator(data_0=act1, prototxt="layer { pooling_param { pool: MAX kernel_size: 2 stride: 2}}", op_type_string="Pool")

    # second conv
    conv2 = mx.symbol.CaffeOperator(data_0=pool1, prototxt="layer { convolution_param { num_output: 50 kernel_size: 5 stride: 1} }", op_type_string="Conv")
    act2 = mx.symbol.CaffeOperator(data_0=conv2, op_type_string="Tanh")
    pool2 = mx.symbol.CaffeOperator(data_0=act2, prototxt="layer { pooling_param { pool: MAX kernel_size: 2 stride: 2}}", op_type_string="Pool")

    fc1 = mx.symbol.CaffeOperator(data_0=pool2, prototxt="layer { inner_product_param{num_output: 500} }", op_type_string="InnerProduct")
    act3 = mx.symbol.CaffeOperator(data_0=fc1, op_type_string="Tanh")

    # second fullc
    fc2 = mx.symbol.CaffeOperator(data_0=act3, prototxt="layer { inner_product_param{num_output: 10} }", op_type_string="InnerProduct")
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet',
                        choices = ['mlp', 'lenet'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp()
    else:
        data_shape = (1, 28, 28)
        net = get_lenet()

    # train
    train_model.fit(args, net, get_iterator(data_shape))
