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
    def __init__(self, data_names, data, label_names, label, bucket_key,
                 utt_id=None, utt_len=0, effective_sample_count=None):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.utt_id = utt_id
        self.utt_len = utt_len
        self.effective_sample_count = effective_sample_count

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        if len(self.label_names):
            return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
        else:
            return None

class SimpleIter(mx.io.DataIter):
    """DataIter used in Calculate Statistics (in progress).

    Parameters
    ----------
    pad_zeros : bool
        Default `False`. Control the behavior of padding when we run
        out of the whole dataset. When true, we will pad with all-zeros.
        When false, will pad with a random sentence in the dataset.
        Usually, for training we would like to use `False`, but
        for testing use `True` so that the evaluation metric can
        choose to ignore the padding by detecting the zero-labels.
    """
    def __init__(self, train_sets, batch_size,
            init_states, delay=5, feat_dim=40, label_dim=1955,
            label_mean_sets=None, data_name='data',
            label_name='softmax_label', has_label=True, load_label_mean=True):

        self.train_sets = train_sets
        self.label_mean_sets = label_mean_sets
        self.train_sets.initialize_read()

        self.data_name = data_name
        if has_label:
            self.label_name = label_name

        features = []
        labels = []
        utt_lens = []
        utt_ids = []
        buckets = []
        self.has_label = has_label

        if label_mean_sets is not None:
            self.label_mean_sets.initialize_read()
            (feats, tgts, utt_id) = self.label_mean_sets.load_next_seq()

            self.label_mean = feats/np.sum(feats)
            for i, v in enumerate(feats):
                if v <= 1.0:
                    self.label_mean[i] = 1

        sys.stderr.write("Loading data...\n")
        buckets_map = {}
        n = 0
        while True:
            (feats, tgts, utt_id) = self.train_sets.load_next_seq()
            if utt_id is None:
                break
            if tgts is None and self.has_label:
                continue
            if feats.shape[0] == 0:
                continue
            features.append(feats)
            utt_lens.append(feats.shape[0])
            utt_ids.append(utt_id)
            if self.has_label:
                labels.append(tgts+1)
            if feats.shape[0] not in buckets:
                buckets_map[feats.shape[0]] = feats.shape[0]

        for k, v in buckets_map.iteritems():
            buckets.append(k)

        buckets.sort()
        i_max_bucket = len(buckets)-1
        max_bucket = buckets[i_max_bucket]
        self.buckets = buckets
        self.data = [[] for k in buckets]
        self.utt_id = [[] for k in buckets]
        self.utt_lens = [[] for k in buckets]
        self.feat_dim = feat_dim
        self.default_bucket_key = max(buckets)

        for i, feats in enumerate(features):
            if has_label:
                tgts = labels[i]
            utt_len = utt_lens[i]
            utt_id = utt_ids[i]

            for i, bkt in enumerate(buckets):
                if bkt >= utt_len:
                    i_bucket = i
                    break

            if self.has_label:
                self.data[i_bucket].append((feats, tgts))
            else:
                self.data[i_bucket].append(feats)
            self.utt_id[i_bucket].append(utt_id)
            self.utt_lens[i_bucket].append(utt_len)

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        self.batch_size = batch_size
        # convert data into ndarrays for better speed during training

        data = [np.zeros((len(x), buckets[i], self.feat_dim))
                if len(x) % self.batch_size == 0
                else np.zeros(((len(x)/self.batch_size + 1) * self.batch_size, buckets[i], self.feat_dim))
                for i, x in enumerate(self.data)]

        label = [np.zeros((len(x), buckets[i]))
                 if len(x) % self.batch_size == 0
                 else np.zeros(((len(x)/self.batch_size + 1) * self.batch_size, buckets[i]))
                 for i, x in enumerate(self.data)]

        utt_id = [[] for k in buckets]
        for i, x in enumerate(data):
            utt_id[i] = ["GAP_UTT"] * len(x)
        utt_lens = [[] for k in buckets]
        for i, x in enumerate(data):
            utt_lens[i] = [0] * len(x)


        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                if self.has_label:
                    sentence[1][delay:] = sentence[1][:-delay]
                    sentence[1][:delay] = sentence[1][0] # broadcast assignment
                    data[i_bucket][j, :len(sentence[0])] = sentence[0]
                    label[i_bucket][j, :len(sentence[1])] = sentence[1]
                else:
                    data[i_bucket][j, :len(sentence)] = sentence
                    # borrow this place to pass in sentence length. TODO: use a less hacky way.
                    label[i_bucket][j, :len(sentence)] += len(sentence)

                utt_id[i_bucket][j] = self.utt_id[i_bucket][j]
                utt_lens[i_bucket][j] = self.utt_lens[i_bucket][j]

        self.data = data
        self.label = label
        self.utt_id = utt_id
        self.utt_lens = utt_lens


        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        sys.stderr.write("Summary of dataset ==================\n")
        for bkt, sz in zip(buckets, bucket_sizes):
            sys.stderr.write("bucket of len %3d : %d samples\n" % (bkt, sz))

        bucket_size_tot = float(sum(bucket_sizes))

        self.bucket_sizes = bucket_sizes
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, (batch_size, self.default_bucket_key, self.feat_dim))] + init_states
        self.provide_label = None
        if has_label:
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
        label_names = []
        if self.has_label:
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
            utt_len = np.array(self.utt_lens[i_bucket])[idx]
            effective_sample_count = mx.nd.sum(label)
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket], utt_id, utt_len,
                                     effective_sample_count=effective_sample_count)
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

class TruncatedSentenceIter(mx.io.DataIter):
    """DataIter used in Truncated-BPTT.

    Each sentence is split into chunks of fixed lengths. The states are
    forwarded during forward, but the backward is only computed within
    chunks. This mechanism does not require bucketing, and it sometimes
    avoid gradient exploding problems in very long sequences.

    Parameters
    ----------
    pad_zeros : bool
        Default `False`. Control the behavior of padding when we run
        out of the whole dataset. When true, we will pad with all-zeros.
        When false, will pad with a random sentence in the dataset.
        Usually, for training we would like to use `False`, but
        for testing use `True` so that the evaluation metric can
        choose to ignore the padding by detecting the zero-labels.
    """
    def __init__(self, train_sets, batch_size, init_states, truncate_len=20, delay=5,
                 feat_dim=40, data_name='data', label_name='softmax_label',
                 has_label=True, do_shuffling=True, pad_zeros=False, time_major=False):

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
        self.pad_zeros = pad_zeros

        self.time_major = time_major

        self.label = None
        if self.time_major:
            self.data = [mx.nd.zeros((truncate_len, batch_size, feat_dim))]
            if has_label:
                self.label = [mx.nd.zeros((truncate_len, batch_size))]
        else:
            self.data = [mx.nd.zeros((batch_size, truncate_len, feat_dim))]
            if has_label:
                self.label = [mx.nd.zeros((batch_size, truncate_len))]
 
        self.init_state_names = [x[0] for x in init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, self.data[0].shape)] + init_states
        self.provide_label = None
        if has_label:
            self.provide_label = [(label_name, self.label[0].shape)]

        self._load_data()
        self._make_data_plan()

    def _load_data(self):
        sys.stderr.write('Loading data into memory...\n')
        self.features = []
        self.labels = []
        self.utt_ids = []

        seq_len_tot = 0.0
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
                tgs[self.delay:] = tgs[:-self.delay]
                tgs[:self.delay] = tgs[0]  # boradcast assign
            self.features.append(feats)
            if self.has_label:
                self.labels.append(tgs+1)
            self.utt_ids.append(utt_id)
            seq_len_tot += feats.shape[0]

        sys.stderr.write('    %d utterances loaded...\n' % len(self.utt_ids))
        sys.stderr.write('    avg-sequence-len = %.0f\n' % (seq_len_tot/len(self.utt_ids)))

    def _make_data_plan(self):
        if self.do_shuffling:
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
        is_pad = [False] * self.batch_size
        pad = 0
        
        if self.time_major:
            np_data_buffer = np.zeros((self.truncate_len, self.batch_size, self.feat_dim))
            np_label_buffer = np.zeros((self.truncate_len, self.batch_size))
        else:
            np_data_buffer = np.zeros((self.batch_size, self.truncate_len, self.feat_dim))
            np_label_buffer = np.zeros((self.batch_size, self.truncate_len))
 
        utt_id_buffer = [None] * self.batch_size

        data_names = [self.data_name] + self.init_state_names
        label_names = [self.label_name]

        # reset states
        for state in self.init_state_arrays:
            state[:] = 0.1

        while True:
            effective_sample_count = self.batch_size * self.truncate_len
            for i, idx in enumerate(utt_idx):
                fea_utt = self.features[idx]
                if utt_inside_idx[i] >= fea_utt.shape[0]:
                    # we have consumed this sentence

                    # reset the states
                    for state in self.init_state_arrays:
                        if self.time_major:
                            state[:, i:i+1, :] = 0.1
                        else:
                            state[i:i+1] = 0.1
                    # load new sentence
                    if is_pad[i]:
                        # I am already a padded sentence, just rewind to the
                        # beginning of the sentece
                        utt_inside_idx[i] = 0
                    elif next_utt_idx >= len(self.features):
                        # we consumed the whole dataset, simply repeat this sentence
                        # and set pad
                        pad += 1
                        is_pad[i] = True
                        utt_inside_idx[i] = 0
                    else:
                        # move to the next sentence
                        utt_idx[i] = self._data_plan[next_utt_idx]
                        idx = utt_idx[i]
                        fea_utt = self.features[idx]
                        utt_inside_idx[i] = 0
                        next_utt_idx += 1

                if is_pad[i] and self.pad_zeros:
                    np_data_buffer[i] = 0
                    np_label_buffer[i] = 0
                    effective_sample_count -= self.truncate_len
                else:
                    idx_take = slice(utt_inside_idx[i],
                                     min(utt_inside_idx[i]+self.truncate_len,
                                         fea_utt.shape[0]))
                    n_take = idx_take.stop - idx_take.start
                    if self.time_major:
                        np_data_buffer[:n_take, i, :] = fea_utt[idx_take]
                        np_label_buffer[:n_take, i] = self.labels[idx][idx_take]
                    else:
                        np_data_buffer[i, :n_take, :] = fea_utt[idx_take]
                        np_label_buffer[i, :n_take] = self.labels[idx][idx_take]
 
                    if n_take < self.truncate_len:
                        if self.time_major:
                            np_data_buffer[n_take:, i, :] = 0
                            np_label_buffer[n_take:, i] = 0
                        else:
                            np_data_buffer[i, n_take:, :] = 0
                            np_label_buffer[i, n_take:] = 0
 
                        effective_sample_count -= self.truncate_len - n_take

                    utt_inside_idx[i] += n_take

                utt_id_buffer[i] = self.utt_ids[idx]

            if pad == self.batch_size:
                # finished all the senteces
                break
            
            self.data[0][:] = np_data_buffer
            self.label[0][:] = np_label_buffer
 
            data_batch = SimpleBatch(data_names, 
                                     self.data + self.init_state_arrays,
                                     label_names, self.label, bucket_key=None,
                                     utt_id=utt_id_buffer,
                                     effective_sample_count=effective_sample_count)

            # Instead of using the 'pad' property, we use an array 'is_pad'. Because
            # our padded sentence could be in the middle of a batch. A sample is pad
            # if we are running out of the data set and they are just some previously
            # seen data to be filled for a whole batch. In prediction, those data
            # should be ignored
            data_batch.is_pad = is_pad

            yield data_batch

    def reset(self):
        self._make_data_plan()


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, train_sets, buckets, batch_size,
                 init_states, delay=5, feat_dim=40,
                 data_name='data', label_name='softmax_label', has_label=True):

        self.train_sets = train_sets
        self.train_sets.initialize_read()

        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        i_max_bucket = len(buckets)-1
        max_bucket = buckets[i_max_bucket]

        if has_label != True:
            buckets = [i for i in range(1, max_bucket)]
            i_max_bucket = len(buckets)-1
            max_bucket = buckets[i_max_bucket]

        self.buckets = buckets
        self.data = [[] for k in buckets]
        self.utt_id = [[] for k in buckets]
        self.feat_dim = feat_dim
        self.default_bucket_key = max(buckets)
        self.has_label = has_label

        sys.stderr.write("Loading data...\n")
        T_OVERLAP = buckets[0]/2
        n = 0
        while True:
            (feats, tgts, utt_id) = self.train_sets.load_next_seq()
            if utt_id is None:
                break
            if tgts is None and self.has_label:
                continue
            if feats.shape[0] == 0:
                continue

            # we split sentence into overlapping segments if it is
            # longer than the largest bucket
            t_start = 0
            t_end = feats.shape[0]
            while t_start < t_end:
                if t_end - t_start > max_bucket:
                    t_take = max_bucket
                    i_bucket = i_max_bucket
                else:
                    for i, bkt in enumerate(buckets):
                        if bkt >= t_end-t_start:
                            t_take = t_end-t_start
                            i_bucket = i
                            break

                n += 1
                if self.has_label:
                    self.data[i_bucket].append((feats[t_start:t_start+t_take],
                                                tgts[t_start:t_start+t_take]+1))
                else:
                    self.data[i_bucket].append(feats[t_start:t_start+t_take])

                self.utt_id[i_bucket].append(utt_id)
                t_start += t_take
                if t_start >= t_end:
                    # this sentence is consumed
                    break
                t_start -= T_OVERLAP

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        self.batch_size = batch_size
        # convert data into ndarrays for better speed during training

        data = [np.zeros((len(x), buckets[i], self.feat_dim))
                if len(x) % self.batch_size == 0
                else np.zeros(((len(x)/self.batch_size + 1) * self.batch_size, buckets[i],
                               self.feat_dim))
                for i, x in enumerate(self.data)]

        label = [np.zeros((len(x), buckets[i]))
                 if len(x) % self.batch_size == 0
                 else np.zeros(((len(x)/self.batch_size + 1) * self.batch_size, buckets[i]))
                 for i, x in enumerate(self.data)]

        utt_id = [[] for k in buckets]
        for i, x in enumerate(data):
            utt_id[i] = ["GAP_UTT"] * len(x)

        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                if self.has_label:
                    sentence[1][delay:] = sentence[1][:-delay]
                    sentence[1][:delay] = sentence[1][0]  # broadcast assignment
                    data[i_bucket][j, :len(sentence[0])] = sentence[0]
                    label[i_bucket][j, :len(sentence[1])] = sentence[1]
                else:
                    data[i_bucket][j, :len(sentence)] = sentence
                    # borrow this place to pass in sentence length. TODO: use a less hacky way.
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

        self.bucket_sizes = bucket_sizes
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, (batch_size, self.default_bucket_key, self.feat_dim))] + \
            init_states
        self.provide_label = [(label_name, (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size), :]
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
            effective_sample_count = mx.nd.sum(label)
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket], utt_id,
                                     effective_sample_count=effective_sample_count)
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

