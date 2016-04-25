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

def Acc_exlude_padding(labels, preds):
    labels = labels.T.reshape((-1,))
    sum_metric = 0
    num_inst = 0
    for i in range(preds.shape[0]):
        pred_label = np.argmax(preds[i], axis=0)
        label = labels[i]
            
        ind = np.nonzero(label.flat)
        pred_label_real = pred_label.flat[ind]
        #print label, pred_label, ind
        label_real = label.flat[ind]
        sum_metric += (pred_label_real == label_real).sum()
        num_inst += len(pred_label_real)
    return sum_metric, num_inst


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, train_sets, buckets, batch_size,
            init_states, delay=5, feat_dim=40,  n_batch=None,
            data_name='data', label_name='label'):

        self.train_sets=train_sets
        self.train_sets.initialize_read()
           


        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for k in buckets]
        #self.label = [[] for k in buckets]
        self.feat_dim = feat_dim
        self.default_bucket_key = max(buckets)

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
                    self.data[i].append((feats,tgts+1))
                    break
                        # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i], self.feat_dim)) for i, x in enumerate(self.data)]
        label = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                sentence[1][delay:] = sentence[1][:-delay]
                sentence[1][:delay] = [sentence[1][0]]*delay 
                data[i_bucket][j, :len(sentence[0])] = sentence[0]
                label[i_bucket][j, :len(sentence[1])] = sentence[1]
        self.data = data
        self.label = label


        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, sz in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, sz))

        bucket_size_tot = float(sum(bucket_sizes))

        self.bucket_sizes = bucket_sizes
        self.batch_size = batch_size
        self.make_data_iter_plan()

        if n_batch is None:
            n_batch = int(bucket_size_tot / batch_size)
        self.n_batch = n_batch

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        #self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size,40))
        #        for t in range(self.default_bucket_key)] + init_states
        #self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,))
        #        for t in range(self.default_bucket_key)]
        self.provide_data = [('data', (batch_size, self.default_bucket_key, self.feat_dim))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size),:]
            self.label[i] = self.label[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket], self.feat_dim))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)


    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            data[:] = self.data[i_bucket][idx]
            label[:] = self.label[i_bucket][idx] 
            data_all = [mx.nd.array(data[:,:,:])] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
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

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    
    dataSlice = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = dataSlice[seqidx]

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
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    #label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    #label = [label_slice[t] for t in range(seq_len)]
    #label = mx.sym.Concat(*label, dim=0)
    #label = mx.sym.Reshape(data=label, target_shape=(0,))
    ################################################################################

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm



if __name__ == '__main__':
    batch_size = 40
    buckets = [100, 200, 300, 400, 500, 600, 700, 800]
    num_hidden = 1024
    num_lstm_layer = 3

    num_epoch = 20
    learning_rate = 0.002
    momentum = 0.9

    contexts = [mx.context.gpu(i) for i in range(3,4)]

    
    feat_dim = 40
    label_dim = 1955 + 1

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    
    state_names = [x[0] for x in init_states]
    def sym_gen(seq_len):
        sym = lstm_unroll(num_lstm_layer, seq_len, feat_dim,
                           num_hidden=num_hidden,
                           num_label=label_dim)
        data_names = ['data'] + state_names
        label_names = ['softmax_label']
        return (sym, data_names, label_names)

   

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

    model = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key, context=contexts)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    model.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
              eval_metric=mx.metric.np(Acc_exlude_padding),
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), 
              initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
              optimizer = "adam",
              optimizer_params={'learning_rate':0.002, 'wd': 0.0})
              #optimizer='sgd',
              #optimizer_params={'learning_rate':0.002, 'momentum': 0.9, 'wd': 0.00001})


    #model.score(data_val, eval_metric)
    #for name, val in metric.get_name_value():
    #    logging.info('validation-%s=%f', name, val)
