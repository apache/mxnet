# pylint:skip-file
import logging
import sys, random, time
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple

ToyModel = namedtuple("ToyModel", ["ex", "symbol", "param_blocks"])

def get_net(vocab_size):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    pred = mx.sym.FullyConnected(data = data, num_hidden = 100)
    pred = mx.sym.FullyConnected(data = pred, num_hidden = vocab_size)
    sm = mx.sym.SoftmaxOutput(data = pred, label = label)
    return sm

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, count, batch_size, vocab_size, num_label, feature_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.feature_size = feature_size
        self.provide_data = [('data', (batch_size, feature_size))]
        self.provide_label = [('label', (self.batch_size,))]

    def mock_sample(self):
        ret = np.zeros(self.feature_size)
        rn = set()
        while len(rn) < 3:
            rn.add(random.randint(0, self.feature_size - 1))
        s = 0
        for k in rn:
            ret[k] = 1.0
            s *= self.feature_size
            s += k
        return ret, s % self.vocab_size

    def __iter__(self):
        for _ in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                d, l = self.mock_sample()
                data.append(d)
                label.append(l)
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    batch_size = 128
    vocab_size = 10000
    feature_size = 100
    num_label = 6

    data_train = DataIter(100000, batch_size, vocab_size, num_label, feature_size)
    data_test = DataIter(1000, batch_size, vocab_size, num_label, feature_size)
    
    network = get_net(vocab_size)
    devs = mx.cpu()
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 20,
                                 learning_rate = 0.03,
                                 momentum = 0.9,
                                 wd = 0.0000,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    
    model.fit(X = data_train, eval_data = data_test,
              batch_end_callback = mx.callback.Speedometer(batch_size, 50),)

