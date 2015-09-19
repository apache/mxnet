# pylint: skip-file
import numpy as np
import mxnet as mx
import copy
import sys
sys.path.append("../../tests/python/common")
import get_data
import time

"""
CXXNET Result:
step1: wmat_lr = 0.05, bias_lr = 0.1, mom = 0.9
[1] train-error:0.452865  val-error:0.3614
[2] train-error:0.280231  val-error:0.2504
[3] train-error:0.220968  val-error:0.2456
[4] train-error:0.18746 val-error:0.2145
[5] train-error:0.165221  val-error:0.1796
[6] train-error:0.150056  val-error:0.1786
[7] train-error:0.134571  val-error:0.157
[8] train-error:0.122582  val-error:0.1429
[9] train-error:0.113891  val-error:0.1398
[10]  train-error:0.106458  val-error:0.1469
[11]  train-error:0.0985054 val-error:0.1447
[12]  train-error:0.0953684 val-error:0.1494
[13]  train-error:0.0872962 val-error:0.1311
[14]  train-error:0.0832401 val-error:0.1544
[15]  train-error:0.0773857 val-error:0.1268
[16]  train-error:0.0743087 val-error:0.125
[17]  train-error:0.0714114 val-error:0.1189
[18]  train-error:0.066616  val-error:0.1424
[19]  train-error:0.0651175 val-error:0.1322
[20]  train-error:0.0616808 val-error:0.111
step2: lr = 0.01, bias_lr = 0.02, mom = 0.9
[21]  train-error:0.033368  val-error:0.0907
[22]  train-error:0.0250959 val-error:0.0876
[23]  train-error:0.0220388 val-error:0.0867
[24]  train-error:0.0195812 val-error:0.0848
[25]  train-error:0.0173833 val-error:0.0872
[26]  train-error:0.0154052 val-error:0.0878
[27]  train-error:0.0141264 val-error:0.0863
[28]  train-error:0.0134071 val-error:0.0865
[29]  train-error:0.0116688 val-error:0.0878
[30]  train-error:0.0106298 val-error:0.0873
step3: lr = 0.001, bias_lr = 0.002, mom = 0.9
[31]  train-error:-nan  val-error:0.0873
[31]  train-error:0.0067735 val-error:0.0859
[32]  train-error:0.0049952 val-error:0.0835
[33]  train-error:0.00485534  val-error:0.0849
[34]  train-error:0.00367647  val-error:0.0839
[35]  train-error:0.0034367 val-error:0.0844
[36]  train-error:0.00275735  val-error:0.084
[37]  train-error:0.00221787  val-error:0.083
[38]  train-error:0.00171835  val-error:0.0838
[39]  train-error:0.00125879  val-error:0.0833
[40]  train-error:0.000699329 val-error:0.0842
"""
def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]


np.random.seed(1812)

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
loss = mx.symbol.Softmax(data=fc, name="loss")
#########################################################
get_data.GetCifar10()
batch_size = 128
epoch = 3
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

def test_cifar():
    model = mx.model.MXNetModel(ctx=mx.gpu(),
            symbol=loss, data=(batch_size, 3, 28, 28),
            optimizer="sgd", num_round = epoch, batch_size = batch_size,
            learning_rate=0.05, momentum=0.9, weight_decay=0.00001)
    model.fit(X=train_dataiter, eval_set=test_dataiter, eval_metric=CalAcc)


if __name__ == "__main__":
    test_cifar()
