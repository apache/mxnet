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
    def __init__(self, input_hdf5, batch_size, init_states, seq_per_img,
                 img_name='image', seq_name='seq', label_name='label', is_train=True):
        super(BucketImageSentenceIter, self).__init__()

        self.img_name = img_name
        self.seq_name = seq_name
        self.label_name = label_name
        self.vgg_mean = np.array([123.68, 116.779, 103.939]).reshape(1, 3, 1, 1)
        self.is_train = is_train

        self.images, self.labels, self.label_start_ix, self.label_end_ix = \
                self.load_h5(input_hdf5)
        self.label_start_ix = np.array(self.label_start_ix)
        self.label_end_ix = np.array(self.label_end_ix)
        assert(self.images.shape[0] == self.label_start_ix.shape[0] == self.label_end_ix.shape[0])

        self.default_bucket_key = self.labels.shape[1] + 1

        self.batch_size = batch_size
        self.seq_per_img = seq_per_img
        self.data_size = self.label_start_ix.shape[0] // self.batch_size * self.batch_size
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('image', (self.batch_size, 3, 224, 224)), \
                             ('seq', (self.batch_size * self.seq_per_img, self.default_bucket_key))] \
                             + init_states
        self.provide_label = [('label', (self.batch_size * self.seq_per_img, self.default_bucket_key))]

        self.image_buffer = np.zeros((batch_size, 3, 256, 256))
        self.seq_buffer = np.zeros((batch_size * seq_per_img, self.default_bucket_key))
        self.label_buffer = np.zeros((batch_size * seq_per_img, self.default_bucket_key - 1))

        self.idx = np.random.permutation(self.data_size)

    @property
    def seq_length(self):
        return self.default_bucket_key


    def load_h5(self, input_hdf5):
        h5_file = h5py.File(input_hdf5, 'r')
        images = h5_file.get("images")
        labels = h5_file.get("labels")
        label_start_ix = h5_file.get("label_start_ix")
        label_end_ix = h5_file.get("label_end_ix")
        return images, labels, label_start_ix, label_end_ix


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

        for curr_idx in xrange(0, self.data_size, self.batch_size):

            image = self.image_buffer
            seq = self.seq_buffer
            label = self.label_buffer
            for i in range(self.batch_size):
                idx = self.idx[curr_idx+i]
                ix1 = self.label_start_ix[idx]
                ix2 = self.label_end_ix[idx]
                image[i] = self.images[idx]
                seq[i * self.seq_per_img:(i + 1) * self.seq_per_img,1:] = \
                        self.labels[ix1:ix2+1]
                label[i * self.seq_per_img:(i + 1) * self.seq_per_img] = \
                        self.labels[ix1:ix2+1]

            data = [mx.nd.array(self.data_augmentation(image, self.is_train))] \
                    + [mx.nd.array(seq)] + self.init_state_arrays
            label = [mx.nd.array(label)]
            data_names = ['image', 'seq'] + init_state_names
            label_names = ['label']

            data_batch = SimpltBatch(data_names, data, label_names, label, self.default_bucket_key)
            yield data_batch

    def reset(self):
        self.idx = np.random.permutation(self.data_size)



