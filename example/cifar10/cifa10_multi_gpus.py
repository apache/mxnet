# pylint: skip-file
import numpy as np
import mxnet as mx
import copy
import sys
sys.path.append("../../tests/python")
import get_data
import time

# use multiple devices
num_devs = 4
devs = [mx.gpu(i) for i in range(num_devs)]
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
mx.kvstore.set_updater(updater)


#check data
get_data.GetCifar10()

train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        input_shape=(3,28,28),
        batch_size=batch_size,
        nthread=1)
test_dataiter = mx.io.ImageRecordIter(
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


def train():
    acc_train = 0.
    acc_val = 0.
    print("Start training...")
    for i in range(epoch):
        # train
        train_acc = 0.0
        val_acc = 0.0
        train_nbatch = 0
        val_nbatch = 0
        all_train_bacth = round(50000 / float(batch_size) + 1)
        for data, label in train_dataiter:
            toc = time.time()
            label = label.asnumpy().flatten()
            tmp_label[:] = label
            inputs["data"][:] = data
            inputs["sm_label"][:] = tmp_label
            executor.forward()
            pred[:] = out_narray
            train_acc += CalAcc(pred.asnumpy(), label)
            train_nbatch += 1
            #executor.backward([out_narray])
            executor.backward()

            for grad, weight, mom in block:
                Update(grad, weight, mom)
            progress(train_nbatch, all_train_bacth, i, toc)

        # evaluate
        for data, label in test_dataiter:
            label = label.asnumpy().flatten()
            inputs["data"][:] = data
            executor.forward()
            pred[:] = out_narray
            val_acc += CalAcc(pred.asnumpy(), label)
            val_nbatch += 1
        acc_train = train_acc / train_nbatch
        acc_val = val_acc / val_nbatch
        sys.stdout.write('\n')
        print("Train Acc: ", train_acc / train_nbatch)
        print("Valid Acc: ", val_acc / val_nbatch)
        train_dataiter.reset()
        test_dataiter.reset()

if __name__ == "__main__":
    train()
