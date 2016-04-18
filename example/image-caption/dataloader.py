#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import mxnet as mx
import json
import h5py


class ImageCaptionIter(mx.io.DataIter):
    def __init__(self, input_hdf5, batch_size, init_states, vocab_size,
                 image_name='image', caption_name='caption', label_name='label'):
        super(ImageCaptionIter, self).__init__()

        self.image_name = image_name
        self.caption_name = caption_name
        self.label_name = label_name
        self.vgg_mean = np.array([123.68, 116.779, 103.939]).reshape(1, 3, 1, 1)

        # hdf5 file
        # images: n x 3 x 256 x 256
        # captions: 5n x l
        # idx pairs: 5n
        # caption_length: 5n
        self.images, self.captions, self.idx_pairs, self.caption_length = self.load_h5(input_hdf5)

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.data_size = self.idx_pairs.shape[0] // self.batch_size * self.batch_size
        self.seq_len = self.captions.shape[1] + 1
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(self.image_name, (self.batch_size, 3, 224, 224)), \
                             (self.caption_name, (self.batch_size, self.seq_length))] + init_states
        self.provide_label = [(self.label_name, (self.batch_size, self.seq_length))]

        self.pad = 0
        self.index = None

        self.idx = np.random.permutation(self.data_size)

        self._image = np.zeros((batch_size, 3, 224, 224))
        self._caption = np.zeros((batch_size, self.seq_len))
        self._label = np.zeros((batch_size, self.seq_len))

    def _preprocess(self, imgs):
        return imgs - self.vgg_mean

    @property
    def seq_length(self):
        return self.seq_len

    def load_h5(self, input_hdf5):
        h5_file = h5py.File(input_hdf5, 'r')
        images = h5_file.get("images")
        captions = h5_file.get("labels")
        idx_pairs = np.array(h5_file.get("idx_pairs"))
        cap_length = np.array(h5_file.get("label_length"))
        return images, captions, idx_pairs, cap_length

    def __iter__(self):

        for curr_idx in xrange(0, self.data_size, self.batch_size):

            for i in xrange(self.batch_size):
                true_idx = self.idx[i]
                i_idx = self.idx_pairs[true_idx]
                cap_len = self.caption_length[true_idx]
                self._image[i] = self.images[i_idx]
                self._caption[i, 1:cap_len+1] = self.captions[true_idx][:cap_len]
                self._caption[i, 0] = self.vocab_size
                self._label[i, :cap_len] = self.captions[true_idx][:cap_len]
                self._label[i, cap_len] = self.vocab_size
            self.data = [mx.nd.array(self._image), mx.nd.array(self._caption)] + self.init_state_arrays
            self.label = [mx.nd.array(self._label)]
            #print self._caption, self._label

            yield self

    def reset(self):
        self.idx = np.random.permutation(self.data_size)

