# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import mxnet.ndarray as ndarray

from lstm import LSTMState, LSTMParam, LSTMModel, lstm

import logging
import validictory
import math
import numpy
import time

from io_func.feat_io import DataReadStream
from utils.utils import setup_logger, to_bool
from utils.main_runner import run_main
DATASETS = {}

DATASETS["AMI_train"] = {
        "lst_file": "/data/sls/scratch/yzhang87/AMI/ami_sdm_baseline/exp_cntk/sdm1/cntk_train_mxnet.feats",
        "format": "kaldi",
        "in": 80
        }

DATASETS["TIMIT_train"] = {
        "lst_file": "/data/scratch/yzhang87/speech/timit/cntk_train_mxnet.feats",
        "format": "kaldi",
        "in": 40
        }

DATASETS["TIMIT_dev"] = {
        "lst_file": "/data/scratch/yzhang87/speech/timit/cntk_dev_mxnet.feats",
        "format": "kaldi",
        "in": 40
        }



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

class Accuracy2(mx.metric.EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(Accuracy2, self).__init__('Accuracy')

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)
            
            ind = np.nonzero(label.flat)
            pred_label_real = pred_label.flat[ind]
            #print label, pred_label, ind
            label_real = label.flat[ind]
            self.sum_metric += (pred_label_real == label_real).sum()
            self.num_inst += len(pred_label_real)


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, train_sets, buckets, batch_size,
            init_states, n_batch=None,
            data_name='data', label_name='label'):

        self.train_sets=train_sets
        self.train_sets.initialize_read()
           


        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for k in buckets]
        self.label = [[] for k in buckets]

        self.default_bucket_key = buckets[0]

        n = 0
        while True:
            (feats, tgts, utt_id) = self.train_sets.load_next_seq();
            if utt_id == None:
                break
            if feats.shape[0] == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= feats.shape[0]:
                    n = n + 1
                    self.data[i].append((feats,tgts))
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

        self.bucket_sizes = bucket_sizes
        self.batch_size = batch_size
        if n_batch is None:
            n_batch = int(bucket_size_tot / batch_size)
        self.n_batch = n_batch

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size,40))
                for t in range(self.default_bucket_key)] + init_states
        self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,))
                for t in range(self.default_bucket_key)]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        idx_bucket = np.arange(len(self.buckets))
        i = 0
        for i_bucket in idx_bucket:
            idx_list = np.arange(self.bucket_sizes[i_bucket], dtype="int32")
            np.random.shuffle(idx_list)
            
            for s,i_batch in enumerate(idx_list):
                
                if i==0:
                    data = np.zeros((self.batch_size, self.buckets[i_bucket], 40))
                    label = np.zeros((self.batch_size, self.buckets[i_bucket]))
                
                sentence = self.data[i_bucket][i_batch][0]
                tgt = self.data[i_bucket][i_batch][1]
                data[i,:len(sentence),:] = sentence
                label[i, :len(tgt)] = tgt + 1
                
                i = i+1
                if i == self.batch_size or s == (len(idx_list)-1):
                    data_all = [mx.nd.array(data[:,t,:])
                        for t in range(self.buckets[i_bucket])] + self.init_state_arrays
                    label_all = [mx.nd.array(label[:, t])
                        for t in range(self.buckets[i_bucket])]
                    data_names = ['%s/%d' % (self.data_name, t)
                        for t in range(self.buckets[i_bucket])] + init_state_names
                    label_names = ['%s/%d' % (self.label_name, t)
                        for t in range(self.buckets[i_bucket])]

                    data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                  self.buckets[i_bucket])
                    i = 0
                    yield data_batch

# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_label, dropout=0.):

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
        hidden = mx.sym.Variable("data/%d" % seqidx)

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
    batch_size = 40
    buckets = [100, 200, 300, 400, 500]
    num_hidden = 512
    num_lstm_layer = 2

    num_epoch = 50
    learning_rate = 0.002
    momentum = 0.9

    contexts = [mx.context.gpu(i) for i in range(3,4)]

    
    feat_dim = 40
    label_dim = 1955 + 1
    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, feat_dim,
                           num_hidden=num_hidden,
                           num_label=label_dim)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    

    train_data = DATASETS[sys.argv[1] + "_train"]
    dev_data = DATASETS[sys.argv[1] + "_dev"]

    train_data_args = {
            "gpu_chunk": 32768,
            "lst_file": train_data["lst_file"],
            "file_format": train_data["format"],
            "separate_lines":True
            }

    dev_data_args = {
            "gpu_chunk": 32768,
            "lst_file": dev_data["lst_file"],
            "file_format": dev_data["format"],
            "separate_lines":True
            }


    train_sets = DataReadStream(train_data_args, train_data["in"])
    dev_sets = DataReadStream(dev_data_args, dev_data["in"])


    data_train = BucketSentenceIter(train_sets,
                                    buckets, batch_size, init_states)
    data_val   = BucketSentenceIter(dev_sets,
                                    buckets, batch_size, init_states)

    model = mx.model.FeedForward(
            ctx           = contexts,
            symbol        = sym_gen,
            num_epoch     = num_epoch,
            #learning_rate = learning_rate,
            #momentum      = momentum,
            optimizer     = "adam",
            wd            = 0.0,
            initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34))
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val, eval_metric=Accuracy2(),
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), )

