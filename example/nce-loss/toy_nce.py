# pylint:skip-file
import sys, random, time
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *

def get_net(vocab_size, num_label):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    pred = mx.sym.FullyConnected(data = data, num_hidden = 100)
    ret = nce_loss(data = pred,
                    label = label,
                    label_weight = label_weight,
                    embed_weight = embed_weight,
                    vocab_size = vocab_size,
                    num_hidden = 100,
                    num_label = num_label)    
    return ret

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
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]

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
        la = [s % self.vocab_size] +\
             [random.randint(0, self.vocab_size - 1) for _ in range(self.num_label - 1)]
        return ret, la

    def __iter__(self):
        for _ in range(self.count / self.batch_size):
            data = []
            label = []
            label_weight = []
            for i in range(self.batch_size):
                d, l = self.mock_sample()
                data.append(d)
                label.append(l)
                label_weight.append([1.0] + [0.0 for _ in range(self.num_label - 1)])
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label), mx.nd.array(label_weight)]
            data_names = ['data']
            label_names = ['label', 'label_weight']
            yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass

if __name__ == '__main__':
    batch_size = 128
    vocab_size = 10000
    feature_size = 100
    num_label = 6
    
    data_train = DataIter(100000, batch_size, vocab_size, num_label, feature_size)
    data_test = DataIter(1000, batch_size, vocab_size, num_label, feature_size)
    
    network = get_net(vocab_size, num_label)
    devs = [mx.cpu()]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 20,
                                 learning_rate = 0.03,
                                 momentum = 0.9,
                                 wd = 0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = NceAccuracy()
    model.fit(X = data_train, eval_data = data_test,
              eval_metric = metric,
              batch_end_callback = mx.callback.Speedometer(batch_size, 50),)

