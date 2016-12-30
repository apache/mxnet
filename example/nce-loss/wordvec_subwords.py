# pylint:skip-file
import logging
import sys, random, time, math
import mxnet as mx
import numpy as np
from nce import *
from operator import itemgetter
from optparse import OptionParser
from collections import Counter

import logging
head = head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)


# ----------------------------------------------------------------------------------------
EMBEDDING_SIZE = 100
BATCH_SIZE = 256
NUM_LABEL = 5
NUM_EPOCH = 20
MIN_COUNT = 5  # only works when doing nagative sampling, keep it same as nce-loss
GRAMS = 3      # here we use triple-letter representation
MAX_SUBWORDS = 10
PADDING_CHAR = '</s>'


# ----------------------------------------------------------------------------------------
def get_net(vocab_size, num_input, num_label):
    data = mx.sym.Variable('data')
    mask = mx.sym.Variable('mask')  # use mask to handle variable-length input.
    label = mx.sym.Variable('label')
    label_mask = mx.sym.Variable('label_mask')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')

    # Get embedding for one-hot input.
    # get sub-word units input.
    unit_embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                  weight=embed_weight,
                                  output_dim=EMBEDDING_SIZE)

    # mask embedding_output to get summation of sub-word units'embedding.
    unit_embed = mx.sym.broadcast_mul(lhs=unit_embed, rhs=mask, name='data_units_embed')

    # sum over all these words then you get word-embedding.
    data_embed = mx.sym.sum(unit_embed, axis=2)

    # Slice input equally along specified axis.
    datavec = mx.sym.SliceChannel(data=data_embed,
                                  num_outputs=num_input,
                                  squeeze_axis=1, name='data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]

    return nce_loss_subwords(data=pred,
                             label=label,
                             label_mask=label_mask,
                             label_weight=label_weight,
                             embed_weight=embed_weight,
                             vocab_size=vocab_size,
                             num_hidden=EMBEDDING_SIZE,
                             num_label=num_label)


def get_subword_units(token, gram=GRAMS):
    """Return subword-units presentation, given a word/token.
    """
    if token == '</s>':  # special token for padding purpose.
        return [token]
    t = '#' + token + '#'
    return [t[i:i + gram] for i in range(0, len(t) - gram + 1)]


def get_subword_representation(wid, vocab_inv, units_vocab, max_len):
    token = vocab_inv[wid]
    units = [units_vocab[unit] for unit in get_subword_units(token)]
    weights = [1] * len(units) + [0] * (max_len - len(units))
    units = units + [units_vocab[PADDING_CHAR]] * (max_len - len(units))
    return units, weights


def prepare_subword_units(tks):
    # statistics on units
    units_vocab = {PADDING_CHAR:1}
    max_len = 0
    unit_set = set()
    logging.info('grams: %d', GRAMS)
    logging.info('counting max len...')
    for tk in tks:
        res = get_subword_units(tk)
        unit_set.update(i for i in res)
        if max_len < len(res):
            max_len = len(res)
    logging.info('preparing units vocab...')
    for unit in unit_set:
        if len(unit) == 0:
            continue
        if unit not in units_vocab:
            units_vocab[unit] = len(units_vocab)
        uid = units_vocab[unit]
    return units_vocab, max_len


def load_data_as_subword_units(name):
    tks = []
    fread = open(name, 'r')
    logging.info('reading corpus from file...')
    for line in fread:
        line = line.strip().decode('utf-8')
        tks.extend(line.split(' '))

    logging.info('Total tokens: %d', len(tks))

    tks = [tk for tk in tks if len(tk) <= MAX_SUBWORDS]
    c = Counter(tks)

    logging.info('Total vocab: %d', len(c))

    vocab = {}
    vocab_inv = {}
    freq = [0]
    data = []
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab)
            freq.append(0)
        wid = vocab[tk]
        vocab_inv[wid] = tk
        data.append(wid)
        freq[wid] += 1

    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < MIN_COUNT:
            continue
        v = int(math.pow(v * 1.0, 0.75))  # sample negative w.r.t. its frequency
        negative += [i for _ in range(v)]

    logging.info('counting subword units...')
    units_vocab, max_len = prepare_subword_units(tks)
    logging.info('vocabulary size: %d', len(vocab))
    logging.info('subword unit size: %d', len(units_vocab))

    logging.info('generating input data...')
    units = []
    weights = []
    for wid in data:
        word_units, weight = get_subword_representation(wid, vocab_inv, units_vocab, max_len)
        units.append(word_units)
        weights.append(weight)

    negative_units = []
    negative_weights = []
    for wid in negative:
        word_units, weight = get_subword_representation(wid, vocab_inv, units_vocab, max_len)
        negative_units.append(word_units)
        negative_weights.append(weight)

    return data, units, weights, negative_units, negative_weights, vocab, units_vocab, freq, max_len


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
    def __init__(self, fname, batch_size, num_label):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data, self.units, self.weights, self.negative_units, self.negative_weights, \
        self.vocab, self.units_vocab, self.freq, self.max_len = load_data_as_subword_units(fname)
        self.vocab_size = len(self.units_vocab)
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1, self.max_len)),
                             ('mask', (batch_size, num_label - 1, self.max_len, 1))]
        self.provide_label = [('label', (self.batch_size, num_label, self.max_len)),
                              ('label_weight', (self.batch_size, num_label)),
                              ('label_mask', (self.batch_size, num_label, self.max_len, 1))]

    def sample_ne(self):
        # a negative sample.
        return self.negative_units[random.randint(0, len(self.negative_units) - 1)]

    def sample_ne_indices(self):
        return [random.randint(0, len(self.negative_units) - 1) for _ in range(self.num_label - 1)]

    def __iter__(self):
        logging.info('DataIter start.')
        batch_data = []
        batch_data_mask = []
        batch_label = []
        batch_label_mask = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.units) - self.num_label - start, self.num_label):
            context_units = self.units[i: i + self.num_label / 2] + \
                            self.units[i + 1 + self.num_label / 2: i + self.num_label]
            context_mask = self.weights[i: i + self.num_label / 2] + \
                           self.weights[i + 1 + self.num_label / 2: i + self.num_label]
            target_units = self.units[i + self.num_label / 2]
            target_word = self.data[i + self.num_label / 2]
            if self.freq[target_word] < MIN_COUNT:
                continue
            indices = self.sample_ne_indices()
            target = [target_units] + [self.negative_units[i] for i in indices]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            target_mask = [self.weights[i + self.num_label / 2]] + [self.negative_weights[i] for i in indices]

            batch_data.append(context_units)
            batch_data_mask.append(context_mask)
            batch_label.append(target)
            batch_label_mask.append(target_mask)
            batch_label_weight.append(target_weight)

            if len(batch_data) == self.batch_size:
                # reshape for broadcast_mul
                batch_data_mask = np.reshape(batch_data_mask, (batch_size, num_label - 1, self.max_len, 1))
                batch_label_mask = np.reshape(batch_label_mask, (batch_size, num_label, self.max_len, 1))
                data_all = [mx.nd.array(batch_data), mx.nd.array(batch_data_mask)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight), mx.nd.array(batch_label_mask)]
                data_names = ['data', 'mask']
                label_names = ['label', 'label_weight', 'label_mask']
                # clean up
                batch_data = []
                batch_data_mask = []
                batch_label = []
                batch_label_weight = []
                batch_label_mask = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False,
                      help="use gpu")

    batch_size = BATCH_SIZE
    num_label = NUM_LABEL

    data_train = DataIter("./data/text8", batch_size, num_label)

    network = get_net(data_train.vocab_size, num_label - 1, num_label)

    options, args = parser.parse_args()
    # devs = mx.cpu()
    devs = [mx.cpu(i) for i in range(4)]
    if options.gpu == True:
        devs = mx.gpu()
    model = mx.model.FeedForward(ctx=devs,
                                 symbol=network,
                                 num_epoch=NUM_EPOCH,
                                 learning_rate=0.3,
                                 momentum=0.9,
                                 wd=0.0000,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    metric = NceAuc()
    model.fit(X=data_train,
              eval_metric=metric,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50), )
