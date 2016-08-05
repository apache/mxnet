# pylint:skip-file
import sys, random, time, math
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *
from operator import itemgetter
from optparse import OptionParser

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def get_net(vocab_size, seq_len, num_label, num_lstm_layer, num_hidden):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
        
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    label_embed_weight = mx.sym.Variable('label_embed_weight')
    data_embed = mx.sym.Embedding(data = data, input_dim = vocab_size,
                                  weight = embed_weight,
                                  output_dim = 100, name = 'data_embed')
    datavec = mx.sym.SliceChannel(data = data_embed,
                                  num_outputs = seq_len,
                                  squeeze_axis = True, name = 'data_slice')
    labelvec = mx.sym.SliceChannel(data = label,
                                   num_outputs = seq_len,
                                   squeeze_axis = True, name = 'label_slice')
    labelweightvec = mx.sym.SliceChannel(data = label_weight,
                                         num_outputs = seq_len,
                                         squeeze_axis = True, name = 'label_weight_slice')
    probs = []
    for seqidx in range(seq_len):
        hidden = datavec[seqidx]
        
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata = hidden,
                              prev_state = last_states[i],
                              param = param_cells[i],
                              seqidx = seqidx, layeridx = i)
            hidden = next_state.h
            last_states[i] = next_state
            
        probs.append(nce_loss(data = hidden,
                              label = labelvec[seqidx],
                              label_weight = labelweightvec[seqidx],
                              embed_weight = label_embed_weight,
                              vocab_size = vocab_size,
                              num_hidden = 100,
                              num_label = num_label))
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
    def __init__(self, name, batch_size, seq_len, num_label, init_states):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, self.freq = load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print self.vocab_size
        self.seq_len = seq_len
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, seq_len))] + init_states
        self.provide_label = [('label', (self.batch_size, seq_len, num_label)),
                              ('label_weight', (self.batch_size, seq_len, num_label))]
        
    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        print 'begin'
        batch_data = []
        batch_label = []
        batch_label_weight = []
        for i in range(0, len(self.data) - self.seq_len - 1, self.seq_len):
            data = self.data[i: i+self.seq_len]
            label = [[self.data[i+k+1]] \
                     + [self.sample_ne() for _ in range(self.num_label-1)]\
                     for k in range(self.seq_len)]
            label_weight = [[1.0] \
                            + [0.0 for _ in range(self.num_label-1)]\
                            for k in range(self.seq_len)]

            batch_data.append(data)
            batch_label.append(label)
            batch_label_weight.append(label_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)] + self.init_state_arrays
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                data_names = ['data'] + self.init_state_names
                label_names = ['label', 'label_weight']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-g", "--gpu", action = "store_true", dest = "gpu", default = False,
                      help = "use gpu")
    batch_size = 1024
    seq_len = 5
    num_label = 6
    num_lstm_layer = 2
    num_hidden = 100

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = DataIter("./data/text8", batch_size, seq_len, num_label,
                          init_states)
    
    network = get_net(data_train.vocab_size, seq_len, num_label, num_lstm_layer, num_hidden)
    options, args = parser.parse_args()
    devs = mx.cpu()
    if options.gpu == True:
        devs = mx.gpu()
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
    
    metric = NceLSTMAuc()
    model.fit(X = data_train,
              eval_metric = metric,
              batch_end_callback = mx.callback.Speedometer(batch_size, 50),)

