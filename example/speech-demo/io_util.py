import mxnet as mx
import numpy as np
import sys
from io_func.feat_io import DataReadStream

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
    def __init__(self, data_names, data, label_names, label, bucket_key, utt_id=None):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.utt_id = utt_id

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, train_sets, buckets, batch_size,
            init_states, delay=5, feat_dim=40,  n_batch=None,
            data_name='data', label_name='label', has_label=True):

        self.train_sets=train_sets
        self.train_sets.initialize_read()

        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for k in buckets]
        #self.label = [[] for k in buckets]
        self.utt_id = [[] for k in buckets]
        self.feat_dim = feat_dim
        self.default_bucket_key = max(buckets)
        self.has_label = has_label

        sys.stderr.write("Loading data...\n")
        n = 0
        while True:
            (feats, tgts, utt_id) = self.train_sets.load_next_seq();
            if utt_id is None:
                break
            if tgts is None and self.has_label:
                continue
            if feats.shape[0] == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= feats.shape[0]:
                    n = n + 1
                    if self.has_label:
                        self.data[i].append((feats,tgts+1))
                    else:
                        self.data[i].append((feats))
                    self.utt_id[i].append(utt_id);
                    break
                    # we just ignore the sentence it is longer than the maximum bucket size here

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        self.batch_size = batch_size
        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i], self.feat_dim))
                if len(x) % self.batch_size == 0  else np.zeros(((len(x)/self.batch_size + 1) *self.batch_size, buckets[i], self.feat_dim)) for i, x in enumerate(self.data)]

        label = [np.zeros((len(x), buckets[i]))
                if len(x) % self.batch_size == 0  else np.zeros(((len(x)/self.batch_size + 1) *self.batch_size, buckets[i])) for i, x in enumerate(self.data)]

        utt_id = [[] for k in buckets]
        for i, x in enumerate(data):
            utt_id[i] = ["GAP_UTT"] * len(x)
        #print utt_id
        #label = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                if self.has_label:
                    sentence[1][delay:] = sentence[1][:-delay]
                    sentence[1][:delay] = [sentence[1][0]]*delay
                    data[i_bucket][j, :len(sentence[0])] = sentence[0]
                    label[i_bucket][j, :len(sentence[1])] = sentence[1]
                else:
                    data[i_bucket][j, :len(sentence)] = sentence
                    label[i_bucket][j, :len(sentence)] += len(sentence)
                utt_id[i_bucket][j] = self.utt_id[i_bucket][j]

        self.data = data
        self.label = label
        self.utt_id = utt_id

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        sys.stderr.write("Summary of dataset ==================\n")
        for bkt, sz in zip(buckets, bucket_sizes):
            sys.stderr.write("bucket of len %3d : %d samples\n" % (bkt, sz))

        bucket_size_tot = float(sum(bucket_sizes))

        self.bucket_sizes = bucket_sizes
        self.make_data_iter_plan()

        if n_batch is None:
            n_batch = int(bucket_size_tot / batch_size)
        self.n_batch = n_batch

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

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
            data = mx.nd.zeros((self.batch_size, self.buckets[i_bucket], self.feat_dim))
            label = mx.nd.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        data_names = ['data'] + init_state_names
        label_names = ['softmax_label']

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            data[:] = self.data[i_bucket][idx]
            label[:] = self.label[i_bucket][idx]
            data_all = [data] + self.init_state_arrays
            label_all = [label]
            utt_id = np.array(self.utt_id[i_bucket])[idx]
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket], utt_id)
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

