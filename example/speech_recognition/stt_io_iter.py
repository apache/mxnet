from __future__ import print_function

import sys

sys.path.insert(0, "../../python")
import mxnet as mx

import random


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None
        self.effective_sample_count = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class STTIter(mx.io.DataIter):
    def __init__(self, count, datagen, batch_size, num_label, init_states, seq_length, width, height,
                 sort_by_duration=True,
                 is_bi_graphemes=False, partition="train",):
        super(STTIter, self).__init__()
        self.batch_size = batch_size
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.datagen = datagen
        self.provide_data = [('data', (batch_size, seq_length, width * height))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.count = count
        self.is_bi_graphemes = is_bi_graphemes
        # self.partition = datagen.partition
        if partition == 'train':
            durations = datagen.train_durations
            audio_paths = datagen.train_audio_paths
            texts = datagen.train_texts
        elif partition == 'validation':
            durations = datagen.val_durations
            audio_paths = datagen.val_audio_paths
            texts = datagen.val_texts
        elif partition == 'test':
            durations = datagen.test_durations
            audio_paths = datagen.test_audio_paths
            texts = datagen.test_texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        # if sortagrad
        if sort_by_duration:
            durations, audio_paths, texts = datagen.sort_by_duration(durations,
                                                                     audio_paths,
                                                                     texts)
        else:
            durations = durations
            audio_paths = audio_paths
            texts = texts

        self.trainDataList = zip(durations, audio_paths, texts)
        # to shuffle data
        if not sort_by_duration:
            random.shuffle(self.trainDataList)

        self.trainDataIter = iter(self.trainDataList)
        self.is_first_epoch = True

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(int(self.count / self.batch_size)):

            audio_paths = []
            texts = []
            for i in range(self.batch_size):
                try:
                    duration, audio_path, text = self.trainDataIter.next()
                except:
                    random.shuffle(self.trainDataList)
                    self.trainDataIter = iter(self.trainDataList)
                    duration, audio_path, text = self.trainDataIter.next()
                audio_paths.append(audio_path)
                texts.append(text)
            if self.is_first_epoch:
                data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=True, is_bi_graphemes=self.is_bi_graphemes)
            else:
                data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=False, is_bi_graphemes=self.is_bi_graphemes)

            data_all = [mx.nd.array(data_set['x'])] + self.init_state_arrays
            label_all = [mx.nd.array(data_set['y'])]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch
        self.is_first_epoch = False

    def reset(self):
        pass
