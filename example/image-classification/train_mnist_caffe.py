import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model

def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip")
        os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.CaffeOperator(arg0 = data, name='fc1', para="layer{ inner_product_param{num_output: 128}}", op_type_name="fullyconnected")
    act1 = mx.symbol.CaffeOperator(arg0 = fc1, para="layer{}", op_type_name="tanh")
    fc2  = mx.symbol.CaffeOperator(arg0 = act1, name='fc2', para="layer{ inner_product_param{num_output: 64}}", op_type_name="fullyconnected")
    act2 = mx.symbol.CaffeOperator(arg0 = fc2, para="layer{}", op_type_name="tanh")
    fc3 = mx.symbol.CaffeOperator(arg0 = act2, name='fc3', para="layer{ inner_product_param{num_output: 10}}", op_type_name="fullyconnected")
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
    conv1 = mx.symbol.CaffeOperator(arg0=data, para="layer { convolution_param { num_output: 20 kernel_size: 5 stride: 1} }", op_type_name="conv")
    # TODO(Haoran): Tanh does not work!!
    #tanh1 = mx.symbol.CaffeOperator(data = conv1, para="layer{}", op_type_name="tanh")
    act1 = mx.symbol.CaffeOperator(arg0=conv1, para="layer{}", op_type_name="tanh")
    pool1 = mx.symbol.Pooling(data=act1, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    conv2 = mx.symbol.CaffeOperator(arg0=pool1, para="layer { convolution_param { num_output: 50 kernel_size: 5 stride: 1} }", op_type_name="conv")
    act2 = mx.symbol.CaffeOperator(arg0=conv2, para="layer{}", op_type_name="tanh")
    pool2 = mx.symbol.Pooling(data=act2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.CaffeOperator(arg0=flatten, para="layer{ inner_product_param{num_output: 500} }", op_type_name="fullyconnected")
    act3 = mx.symbol.CaffeOperator(arg0=fc1, para="layer{}", op_type_name="tanh")

    # second fullc
    fc2 = mx.symbol.CaffeOperator(arg0=act3, para="layer{ inner_product_param{num_output: 10} }", op_type_name="fullyconnected")
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir
        if '://' not in args.data_dir:
            _download(args.data_dir)
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = data_dir + "train-images-idx3-ubyte",
            label       = data_dir + "train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        val = mx.io.MNISTIter(
            image       = data_dir + "t10k-images-idx3-ubyte",
            label       = data_dir + "t10k-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        return (train, val)
    return get_iterator_impl

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
