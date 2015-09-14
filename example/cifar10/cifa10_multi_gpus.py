# pylint: skip-file
import numpy as np
import mxnet as mx
import copy
import sys
sys.path.append("../../tests/python/common")
import get_data
import time

# use multiple devices
num_devs = 2
devs = [mx.cpu(i) for i in range(num_devs)]
mx.kvstore.start()

# define the network
conv_cnt = 1
concat_cnt = 1
pool_cnt = 1

def ConvFactory(**kwargs):
    global conv_cnt
    param = copy.copy(kwargs)
    act = param["act_type"]
    del param["act_type"]
    param["workspace"] = 256
    param["name"] = "conv%d" % conv_cnt
    conv = mx.symbol.Convolution(**param)
    bn = mx.symbol.BatchNorm(data = conv, name="bn%d" % conv_cnt)
    relu = mx.symbol.Activation(data = bn, name = "%s%d" % (act, conv_cnt), act_type=act)
    conv_cnt += 1
    return relu

def DownsampleFactory(data, ch_3x3, stride = 2):
    global pool_cnt
    global concat_cnt
    param = {}
    # conv 3x3
    param["kernel"] = (3, 3)
    param["stride"] = (stride, stride)
    param["num_filter"] = ch_3x3
    param["act_type"] = "relu"
    param["data"] = data
    param["pad"] = (1, 1)
    conv3x3 = ConvFactory(**param)
    # pool
    del param["num_filter"]
    del param["act_type"]
    del param["pad"]
    param["pool_type"] = "max"
    param["name"] = "pool%d" % pool_cnt
    pool = mx.symbol.Pooling(**param)
    pool_cnt += 1
    # concat
    concat = mx.symbol.Concat(*[conv3x3, pool], name="concat%d" % concat_cnt)
    concat_cnt += 1
    return concat

def SimpleFactory(data, ch_1x1, ch_3x3):
    global concat_cnt
    param = {}
    # 1x1
    param["kernel"] = (1, 1)
    param["num_filter"] = ch_1x1
    param["pad"] = (0, 0)
    param["stride"] = (1, 1)
    param["act_type"] = "relu"
    param["data"] = data
    conv1x1 = ConvFactory(**param)

    # 3x3
    param["kernel"] = (3, 3)
    param["num_filter"] = ch_3x3
    param["pad"] = (1, 1)
    conv3x3 = ConvFactory(**param)

    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3], name="concat%d" % concat_cnt)
    concat_cnt += 1
    return concat

def RandomInit(narray):
    in_num = narray.shape[1]
    out_num = narray.shape[0]
    a = np.sqrt(3.0 / (in_num + out_num))
    tmp = mx.nd.array(np.random.uniform(-a, a, narray.shape))
    narray[:] = tmp

data = mx.symbol.Variable(name="data")
conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
in3a = SimpleFactory(conv1, 32, 32)
in3b = SimpleFactory(in3a, 32, 48)
in3c = DownsampleFactory(in3b, 80)
in4a = SimpleFactory(in3c, 112, 48)
in4b = SimpleFactory(in4a, 96, 64)
in4c = SimpleFactory(in4b, 80, 80)
in4d = SimpleFactory(in4c, 48, 96)
in4e = DownsampleFactory(in4d, 96)
in5a = SimpleFactory(in4e, 176, 160)
in5b = SimpleFactory(in5a, 176, 160)
pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="pool%d" % pool_cnt)
flatten = mx.symbol.Flatten(data=pool, name="flatten1")
fc = mx.symbol.FullyConnected(data=flatten, num_hidden=10, name="fc1")
loss = mx.symbol.Softmax(data=fc, name="sm")

# define model updater
updater = mx.updater.momentum(
    learning_rate = .05, weight_decay = .0001, momentum = 0.9)
# mx.kvstore.set_updater(updater)

# infer shape
batch_size = 128
batch_size -= (batch_size % num_devs)
data_shape = (batch_size, 3, 28, 28)

# create executors for devices
d = devs[0]
loss.simple_bind(d, data = mx.nd.empty(data_shape, d))
loss.simple_bind(d, data = mx.nd.empty(data_shape, d))

executors = []
for d in devs:
    executors.append(loss.simple_bind(d, data = mx.nd.empty(data_shape, d)))

# d = devs[0]
# ex = loss.simple_bind(d, data = mx.nd.empty(data_shape, d))
# print ex

# executors = [loss.simple_bind(d, data = mx.nd.empty(data_shape, d)) for d in devs]

# find the params needed to be synchronized between devices
param_names = loss.list_arguments()
sync_indices = [index for index, name in enumerate(param_names)
                if "weight" in name or "bias" in name]
sync_weights = [[e.list_arguments()[0][i] for e in executors] for i in sync_indices]
sync_grads = [[e.list_arguments()[1][i] for e in executors] for i in sync_indices]


# init global shared model
for idx in sync_indices:
    shape = sync_weights[0][idx].shape
    val = mx.nd.zeros(shape)
    if "weight" in param_names[idx]:
        val[:] = np.random.uniform(-0.1, 0.1, shape)
    mx.kvstore.init(idx, val)

# init local variables
for e in executors:
    for idx, data in enumerate(e.list_arguments()[0]):
        if "gamma" in param_names[idx]:
            data = 1.0
        if "beta" in param_names[idx]:
            data = 0.0

# data reader
get_data.GetCifar10()
train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        input_shape=(3,28,28),
        batch_size=batch_size,
        nthread=1)
val_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        input_shape=(3,28,28),
        batch_size=batch_size,
        nthread=1)


def progress(count, total, epoch, toc):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    tic = time.time()
    speed = batch_size / float(tic - toc)
    suffix = "Epoch %d, Speed: %.2f pic/sec" % (epoch, speed)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))

def cal_acc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

def train():
    acc_train = 0.
    acc_val = 0.
    k = batch_size / num_devs
    batch_splits = [range(d*k, (d+1)*k) for d in range(num_devs)]
    print("Start training...")
    data_in = [e.list_arguments()[0][param_names.index('data')] for e in executors]
    label_in = [e.list_arguments()[0][param_names.index('loss_label')] for e in executors]

    for i in range(epoch):
        # train
        train_acc = 0.0
        val_acc = 0.0
        train_nbatch = 0
        val_nbatch = 0
        all_train_bacth = round(50000 / float(batch_size) + 1)

        for data, label in train_dataiter:
            # pull weight
            mx.kvstore.pull(sync_indices, out = sync_weights)

            # forward and backword
            data = data.asnumpy()
            label = label.asnumpy().flatten()
            for d in range(num_devs):
                rows = batch_splits[d]
                data_in[d] = data[rows, :]
                label_in[d] = label[rows]
                executors[d].forward()
                executors[d].backward()

            # push gradient
            mx.kvstore.push(sync_indices, sync_grads)

            # evaluate
            for d in range(num_devs):
                train_acc += cal_acc(executors[d].outputs[0].asnumpy(),
                                     label[batch_splits[d]])
                train_count += 1

            progress(train_count, all_train_bacth, i, toc)

        # evaluate
        for data, label in test_dataiter:
            # forward
            data = data.asnumpy()
            label = label.asnumpy().flatten()
            for d in range(num_devs):
                rows = batch_splits[d]
                data_in[d] = data[rows,:]
                executors[d].forward()

            # eval
            for d in range(num_devs):
                val_acc += cal_acc(executors[d].outputs[0].asnumpy(),
                                   label[batch_splits[d]])
                val_count += 1

        sys.stdout.write('\n')

        print("Train Acc: %g, Valid Acc: %g" % (
            train_acc / train_count,
            val_acc / val_count))
        train_dataiter.reset()
        test_dataiter.reset()

    mx.kvstore.stop()
if __name__ == "__main__":
    train()
