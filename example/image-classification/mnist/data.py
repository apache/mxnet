import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../../python"))
import mxnet as mx

def _download(data_dir='data/'):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip")
        os.system("unzip -u mnist.zip")
    os.chdir("..")

def get_iterator(batch_size, input_shape, data_dir='data/'):
    """return train and val iterators for mnist"""
    _download()
    flat = False if len(input_shape) == 3 else True
    train = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

    val = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return (train, val)
