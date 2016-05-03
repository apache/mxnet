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

class TruncatedSentenceIter(mx.io.DataIter):
    """DataIter used in Truncated-BPTT.

    Each sentence is split into chunks of fixed lengths. The states are
    forwarded during forward, but the backward is only computed within
    chunks. This mechanism does not require bucketing, and it sometimes
    avoid gradient exploding problems in very long sequences.
    """
    def __init__(self, train_sets, batch_size, init_states, truncate_len=20, delay=5,
                 feat_dim=40, data_name='data', label_name='label',
                 has_label=True, do_shuffling=True):

        self.train_sets = train_sets
        self.train_sets.initialize_read()

        self.data_name = data_name
        self.label_name = label_name

        self.feat_dim = feat_dim
        self.has_label = has_label
        self.batch_size = batch_size
        self.truncate_len = truncate_len
        self.delay = delay

        self.do_shuffling = do_shuffling

        self.data = [mx.nd.zeros(batch_size, truncate_len, feat_dim)]
        self.label = None
        if has_label:
            self.label = [mx.nd.zeros(batch_size, truncate_len)]

        self.init_state_names = [x[0] for x in init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, self.data[0].shape)] + init_states
        self.provide_label = []
        if has_label:
            self.provide_label = [(label_name, self.label[0].shape)]

        self._load_data()
        self._make_data_plan()

    def _load_data(self):
        sys.stderr.write('Loading data into memory...\n')
        self.features = []
        self.labels = []
        self.utt_ids = []
        while True:
            (feats, tgs, utt_id) = self.train_sets.load_next_seq()
            if utt_id is None:
                break
            if tgs is None and self.has_label:
                continue
            if feats.shape[0] == 0:
                continue

            if self.has_label and self.delay > 0:
                # delay the labels
                tgs[delay:] = tgs[:-delay]
                tgs[:delay] = tgs[0] # boradcast assign

            self.features.append(feats)
            if self.has_label:
                self.labels.append(tgs+1)
            self.utt_ids.append(utt_id)
        sys.stderr.write('%d utterances loaded...\n' % len(self.utt_ids))

    def _make_data_plan(self):
        if do_shuffling:
            # TODO: should we group utterances of similar length together?
            self._data_plan = np.random.permutation(len(self.features))
        else:
            # we might not want to do shuffling for testing for example
            self._data_plan = np.arange(len(self.features))

    def __iter__(self):
        assert len(self._data_plan) >= self.batch_size, \
            "Total number of sentences smaller than batch size, consider using smaller batch size"
        utt_idx = self._data_plan[:self.batch_size]
        utt_inside_idx = [0] * self.batch_size

        next_utt_idx = self.batch_size
        pad = 0

        np_data_buffer = np.zeros((self.batch_size, self.truncate_len, self.feat_dim))
        np_label_buffer = np.zeros((self.batch_size, self.truncate_len))
        utt_id_buffer = [None] * self.batch_size

        data_names = [self.data_name] + self.init_state_names
        label_names = [self.label_name]

        while True:
            for i, idx in enumerate(utt_idx):
                fea_utt = self.features[idx]
                if utt_inside_idx[i] >= fea_utt.shape[0]:
                    # we have consumed this sentence

                    # reset the states
                    # TODO: implement it here

                    # load new sentence
                    if next_utt_idx >= len(self.features):
                        # we consumed the whole dataset, simply repeat this sentence
                        # and set pad
                        pad += 1
                        utt_inside_idx[i] = 0
                    else:
                        # move to the next sentence
                        utt_idx[i] = self._data_plan[next_utt_idx]
                        idx = utt_idx[i]
                        fea_utt = self.features[idx]
                        utt_inside_idx[i] = 0
                        next_utt_idx += 1

                idx_take = utt_inside_idx[i]:min(utt_inside_idx[i]+self.truncate_len, fea_utt.shape[0])
                np_data_buffer[i][:len(idx_take)] = fea_utt[idx_take]
                np_label_buffer[i][:len(idx_take)] = self.labels[idx][idx_take]
                if len(idx_take) < self.truncate_len:
                    np_data_buffer[i][len(idx_take)] = 0
                    np_label_buffer[i][len(idx_take)] = 0

                utt_inside_idx[i] += len(idx_take)
                utt_id_buffer[i] = self.utt_ids[idx]

            if pad == self.batch_size:
                # finished all the senteces
                break

            self.data[0][:] = np_data_buffer
            self.label[0][:] = np_label_buffer

            data_batch = SimpleBatch(data_names, self.data + self.init_state_arrays,
                                     label_names, self.label, bucket_key=None,
                                     utt_id=utt_id_buffer)

            yield data_batch

    def reset(self):
        self._make_data_plan()


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, train_sets, buckets, batch_size,
            init_states, delay=5, feat_dim=40,  n_batch=None,
            data_name='data', label_name='softmax_label', has_label=True):

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
            (feats, tgts, utt_id) = self.train_sets.load_next_seq()
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

        self.provide_data = [(data_name, (batch_size, self.default_bucket_key, self.feat_dim))] + init_states
        self.provide_label = [(label_name, (self.batch_size, self.default_bucket_key))]

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
        data_names = [self.data_name] + init_state_names
        label_names = [self.label_name]

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

