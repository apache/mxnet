# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np

from lstm import LSTMState, LSTMParam, LSTMModel, lstm

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def read_content(path):
    with open(path) as input:
        content = input.read()
        content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
        return content

def build_vocab(path):
    content = read_content(path)
    content = content.split(' ')
    idx = 1 # 0 is left for zero-padding
    vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

def text2id(sentence, vocab):
    words = sentence.split(' ')
    words = [vocab[w] for w in words if len(w) > 0]
    return words

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size,
            init_states, n_batch=None,
            data_name='data', label_name='label'):
        content = read_content(path)
        sentences = content.split(' <eos> ')

        self.vocab_size = len(vocab)
        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for k in buckets]

        self.default_bucket_key = buckets[0]

        for sentence in sentences:
            sentence = text2id(sentence, vocab)
            if len(sentence) == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data[i].append(sentence)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, sz in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, sz))

        bucket_size_tot = float(sum(bucket_sizes))
        bucket_ratio_cum = [sum(bucket_sizes[:i+1]) / bucket_size_tot
                for i in range(len(bucket_sizes))]

        self.bucket_ratio_cum = bucket_ratio_cum

        self.batch_size = batch_size
        if n_batch is None:
            n_batch = int(bucket_size_tot / batch_size)
        self.n_batch = n_batch

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size,))
                for t in range(self.default_bucket_key)] + init_states
        self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,))
                for t in range(self.default_bucket_key)]

    def embed_data(self, x):
        return coo_matrix((np.ones(len(x)), (np.arange(len(x)), x)),
                          shape=(self.batch_size, self.vocab_size)).todense()

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_batch in range(self.n_batch):
            # pick a random bucket
            rnd = np.random.rand()
            for i, ratio in enumerate(self.bucket_ratio_cum):
                if ratio >= rnd:
                    i_bucket = i
                    break

            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))

            for i in range(self.batch_size):
                # pick a random sentence from the bucket
                sentence = np.random.choice(self.data[i_bucket])
                data[i, :len(sentence)] = sentence
                label[i, :len(sentence)-1] = sentence[1:]

            data_all = [mx.nd.array(data[:, t])
                    for t in range(self.buckets[i_bucket])] + self.init_state_arrays
            label_all = [mx.nd.array(label[:, t])
                    for t in range(self.buckets[i_bucket])]
            data_names = ['%s/%d' % (self.data_name, t)
                    for t in range(self.buckets[i_bucket])] + init_state_names
            label_names = ['%s/%d' % (self.label_name, t)
                    for t in range(self.buckets[i_bucket])]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                              self.buckets[i_bucket])
            yield data_batch

# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):

    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    loss_all = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("data/%d" % seqidx)

        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=input_size,
                                  output_dim=num_embed,
                                  name="t%d_embed" % seqidx)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias,
                                   num_hidden=num_label)
        sm = mx.sym.SoftmaxOutput(data=fc, label=mx.sym.Variable('label/%d' % seqidx),
                                  name='t%d_sm' % seqidx)
        loss_all.append(sm)

    # for i in range(num_lstm_layer):
    #     state = last_states[i]
    #     state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
    #                       h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
    #     last_states[i] = state
    #
    # unpack_c = [state.c for state in last_states]
    # unpack_h = [state.h for state in last_states]
    #
    # return mx.sym.Group(loss_all + unpack_c + unpack_h)
    return mx.sym.Group(loss_all)


if __name__ == '__main__':
    batch_size = 32
    buckets = [10, 20, 30, 40]
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 25
    learning_rate = 0.1
    momentum = 0.0

    contexts = [mx.context.cpu(i) for i in range(1)]

    vocab = build_vocab("./data/ptb.train.txt")

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter("./data/ptb.train.txt", vocab,
                                    buckets, batch_size, init_states)
    data_val   = BucketSentenceIter("./data/ptb.valid.txt", vocab,
                                    buckets, batch_size, init_states)

    model = mx.model.FeedForward(
            ctx           = contexts,
            symbol        = sym_gen,
            num_epoch     = num_epoch,
            learning_rate = learning_rate,
            momentum      = momentum,
            wd            = 0.00001,
            initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              batch_end_callback = mx.callback.Speedometer(batch_size, 50),)

