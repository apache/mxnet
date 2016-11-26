# pylint:skip-file
import sys, random, time, math
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *
from operator import itemgetter

ToyModel = namedtuple("ToyModel", ["ex", "symbol", "param_blocks"])

def get_net(vocab_size, num_input, num_label):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    data_embed = mx.sym.Embedding(data = data, input_dim = vocab_size,
                                  weight = embed_weight,
                                  output_dim = 100, name = 'data_embed')
    datavec = mx.sym.SliceChannel(data = data_embed,
                                     num_outputs = num_input,
                                     squeeze_axis = 1, name = 'data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]
    probs = nce_loss(data = pred,
                     label = label,
                     label_weight = label_weight,
                     embed_weight = embed_weight,
                     vocab_size = vocab_size,
                     num_hidden = 100,
                     num_label = num_label)
    return mx.sym.Group(probs)

def load_data(name):
    buf = open(name).read()
    tks = buf.split(' ')
    vocab = {}
    freq = [0]
    data = []
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab) + 1
            freq.append(0)
        wid = vocab[tk]
        data.append(wid)
        freq[wid] += 1
    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]
    return data, negative, vocab, freq

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
    def __init__(self, name, batch_size, num_label):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, self.freq = load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print self.vocab_size
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1))]
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        print 'begin'
        batch_data = []
        batch_label = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.data) - self.num_label - start, self.num_label):
            context = self.data[i: i + self.num_label / 2] \
                      + self.data[i + 1 + self.num_label / 2: i + self.num_label]
            target_word = self.data[i + self.num_label / 2]
            if self.freq[target_word] < 5:
                continue
            target = [target_word] \
                     + [self.sample_ne() for _ in range(self.num_label - 1)]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            batch_data.append(context)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                data_names = ['data']
                label_names = ['label', 'label_weight']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass

class NceAccuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAccuracy, self).__init__('nce-accuracy')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = np.array([x.asnumpy() for x in preds]).transpose()
        for i in range(preds.shape[0]):
            if np.argmax(label_weight[i]) == np.argmax(preds[i]):
                self.sum_metric += 1
            self.num_inst += 1

class NceAuc(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAuc, self).__init__('nce-auc')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = np.array([x.asnumpy() for x in preds]).transpose()
        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key = itemgetter(1), reverse = True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, b in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1

if __name__ == '__main__':
    batch_size = 1024
    num_label = 5

    data_train = DataIter("../data/text8", batch_size, num_label)

    network = get_net(data_train.vocab_size, num_label - 1, num_label)
    # devs = [mx.gpu(i) for i in range(1)]
    devs = [mx.cpu(i) for i in range(4)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 20,
                                 learning_rate = 0.3,
                                 momentum = 0.9,
                                 wd = 0.0000,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    metric = NceAuc()
    model.fit(X = data_train,
              eval_metric = metric,
              #kvstore = 'local_allreduce_device',
              batch_end_callback = mx.callback.Speedometer(batch_size, 50),)
