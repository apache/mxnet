# -*- coding:utf-8 -*-
# @author: Yuanqin Lu

import numpy as np
import mxnet as mx
import json
import string
import h5py
from collections import defaultdict


def cap2words(caption):
    return str(caption).lower().translate(None, string.punctuation).strip().split()



def build_vocab(input_json, threshold=5):
    imgs = json.load(open(input_json, 'r'))
    word_count = defaultdict(int)
    for img in imgs:
        for cpt in img['captions']:
            words = cap2words(cpt)
            for w in words:
                word_count[w] += 1

    the_vocab = {'START': 0,
                 'UNK': 1}  # 'START' is 0 and 0 is 'EOS'
    idx = 2
    for word, count in word_count.items():
        if count < threshold:
            continue
        the_vocab[word] = idx
        idx += 1
    return the_vocab



def default_cap2idx(caption, the_vocab):
    words = cap2words(caption)
    words = [0] + [the_vocab.get(w, 1) for w in words if len(w) > 0]
    return words





class SimpltBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]



class DummyIter(mx.io.DataIter):
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size
        self.default_bucket_key = real_iter.default_bucket_key

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch



class BucketImageSentenceIter(mx.io.DataIter):
    def __init__(self, input_json, input_hdf5, vocab, buckets, batch_size,
                 init_states, data_name='data', label_name='label',
                 text2id=None, is_train=True):
        super(BucketImageSentenceIter, self).__init__()

        if text2id:
            self.text2id = text2id
        else:
            self.text2id = default_cap2idx

        self.vocab_size = vocab
        self.data_name = data_name
        self.label_name = label_name
        self.vgg_mean = np.array([123.68, 116.779, 103.939]).reshape(1, 3, 1, 1)
        self.is_train = is_train

        buckets.sort()
        self.buckets = buckets

        self.caps = [[] for _ in buckets]
        self.imgs_idx = [[] for _ in buckets] # img idx of hdf5 file
        self.ids  = [[] for _ in buckets]
        self.imgs = h5py.File(input_hdf5, 'r').get("images")


        self.default_bucket_key = buckets[0]

        raw_data = json.load(open(input_json, 'r'))
        for img in raw_data:
            img_idx = img['img_idx']
            id = img['id']
            captions = img['captions']
            for cpt in captions:
                cpt = self.text2id(cpt, vocab)
                if len(cpt) == 0:
                    continue
                for i, bkt in enumerate(buckets):
                    if bkt >= len(cpt):
                        self.caps[i].append(cpt)
                        self.imgs_idx[i].append(img_idx)
                        self.ids[i].append(id)
                        break

        # data (int, int, array)
        # convert data into ndarrays
        caps = [np.zeros( (len(x), buckets[i]) ) for i, x in enumerate(self.caps)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.caps[i_bucket])):
                cap = self.caps[i_bucket][j]
                caps[i_bucket][j, :len(cap)] = cap
        self.caps = caps
        imgs_idx = [np.array(x) for x in self.imgs_idx]
        self.imgs_idx = imgs_idx

        bucket_size = [len(x) for x in self.caps]

        print("Summary of dataset")
        for bkt, size in zip(buckets, bucket_size):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('%s/%d' %(self.data_name, 0), (self.batch_size, 3, 224, 224))] \
                              + [('%s/%d' % (self.data_name, t), (self.batch_size,))
                             for t in range(1, self.default_bucket_key + 1)] + init_states  # the first data is img
        self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size, ))
                              for t in range(1, self.default_bucket_key + 1)]  # label should start with 1



    def make_data_iter_plan(self):
        bucket_n_batches = []
        for i in range(len(self.caps)):
            bucket_n_batches.append(len(self.caps[i]) / self.batch_size)
            real_size = int(bucket_n_batches[i] * self.batch_size)
            self.caps[i] = self.caps[i][:int(real_size)]
            self.imgs_idx[i] = self.imgs_idx[i][:int(real_size)]
            # TODO: ids?, for language evaluation

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.caps]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.caps]

        self.img_buffer = np.zeros((self.batch_size, 3, 256, 256))
        self.seq_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.caps)):
            seq = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.seq_buffer.append(seq)
            self.label_buffer.append(label)


    def data_augmentation(self, imgs, data_augment=True):
        # only do random crop here
        h, w = imgs.shape[2:]
        cnn_input_size = 224

        if h > cnn_input_size or w > cnn_input_size:
            xoff, yoff = 0, 0
            if data_augment:
                xoff, yoff = np.random.randint(0, w-cnn_input_size), np.random.randint(0, h-cnn_input_size)
            else:
                xoff, yoff = (w - cnn_input_size) / 2, (h - cnn_input_size) / 2
            # crop
            imgs = imgs[:, :, yoff:yoff+cnn_input_size, xoff:xoff+cnn_input_size]

        return imgs - self.vgg_mean




    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:
            img = self.img_buffer
            seq = self.seq_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size


            # TODO: remove loop
            for i, img_idx in enumerate(self.imgs_idx[i_bucket][idx]):
                img[i] = self.imgs[img_idx]

            seq[:] = self.caps[i_bucket][idx]
            label[:, :-1] = seq[:, 1:]
            label[:, -1] = 0

            data_all = [mx.nd.array(self.data_augmentation(img, self.is_train))] + [mx.nd.array(seq[:, t]) for t in range(self.buckets[i_bucket])] \
                        + self.init_state_arrays
            label_all = [mx.nd.array(label[:, t]) for t in range(self.buckets[i_bucket])]

            data_names = ['%s/%d' % (self.data_name, t)
                          for t in range(self.buckets[i_bucket] + 1)] + init_state_names
            label_names = ['%s/%d' % (self.label_name, t)
                           for t in range(1, self.buckets[i_bucket] + 1)]

            data_batch = SimpltBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])

            yield data_batch


    def reset(self):
        self.bucket_curr_idx = [0 for x in self.caps]


