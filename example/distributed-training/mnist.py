import sys
sys.path.insert(0, "../../python/")
import mxnet as mx


def data(data_dir, batch_size, num_parts=1, part_index=0):
    if data_dir == 'data':
        sys.path.insert(0, "../../tests/python/common")
        import get_data
        get_data.GetMNIST_ubyte()

    train_dataiter = mx.io.MNISTIter(
        image = data_dir + "/train-images-idx3-ubyte",
        label = data_dir + "/train-labels-idx1-ubyte",
        data_shape=(1, 28, 28),
        batch_size=batch_size, shuffle=True, flat=False, silent=False,
        num_parts   = num_parts,
        part_index  = part_index)
    val_dataiter = mx.io.MNISTIter(
        image = data_dir + "/t10k-images-idx3-ubyte",
        label = data_dir + "/t10k-labels-idx1-ubyte",
        data_shape=(1, 28, 28),
        batch_size=batch_size, shuffle=True, flat=False, silent=False)

    return (train_dataiter, val_dataiter)

def lenet():
    data = mx.symbol.Variable('data')
    conv1= mx.symbol.Convolution(data = data, num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1)
    act1 = mx.symbol.Activation(data = bn1, act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, kernel=(2,2), stride=(2,2), pool_type='max')

    conv2= mx.symbol.Convolution(data = mp1, num_filter=32, kernel=(3,3), stride=(2,2))
    bn2 = mx.symbol.BatchNorm(data = conv2)
    act2 = mx.symbol.Activation(data = bn2, act_type="relu")
    mp2 = mx.symbol.Pooling(data = act2, kernel=(2,2), stride=(2,2), pool_type='max')

    fl = mx.symbol.Flatten(data = mp2)
    fc2 = mx.symbol.FullyConnected(data = fl, num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(data = fc2)
    return softmax
