# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import, division, print_function

import json
import random
import numpy as np
from stt_utils import calc_feat_dim, spectrogram_from_file

from config_util import generate_file_path
from log_util import LogUtil
from label_util import LabelUtil
from stt_bi_graphemes_util import generate_bi_graphemes_label
from multiprocessing import cpu_count, Process, Manager

logUtil = LogUtil.getInstance()

class DataGenerator(object):
    def __init__(self, save_dir, model_name, step=10, window=20, max_freq=8000, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        #calc_feat_dim returns int(0.001*window*max_freq)+1
        super(DataGenerator, self).__init__()
        # feat_dim=0.001*20*8000+1=161
        self.feat_dim = calc_feat_dim(window, max_freq)
        # 1d 161 length of array filled with zeros
        self.feats_mean = np.zeros((self.feat_dim,))
        # 1d 161 length of array filled with 1s
        self.feats_std = np.ones((self.feat_dim,))
        self.max_input_length = 0
        self.max_length_list_in_batch = []
        # 1d 161 length of array filled with random value
        #[0.0, 1.0)
        self.rng = random.Random()
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.save_dir = save_dir
        self.model_name = model_name

    def get_meta_from_file(self, feats_mean, feats_std):
        self.feats_mean = feats_mean
        self.feats_std = feats_std

    def featurize(self, audio_clip, overwrite=False, save_feature_as_csvfile=False):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq, overwrite=overwrite,
            save_feature_as_csvfile=save_feature_as_csvfile)

    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=16.0,):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        logger = logUtil.getlogger()
        logger.info('Reading description file: {} for partition: {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    logger.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    logger.warn(str(e))

        if partition == 'train':
            self.count = len(audio_paths)
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
            self.val_count = len(audio_paths)
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, desc_file, max_duration):
        self.load_metadata_from_desc_file(desc_file, 'train', max_duration=max_duration)

    def load_validation_data(self, desc_file, max_duration):
        self.load_metadata_from_desc_file(desc_file, 'validation', max_duration=max_duration)

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts):
        return zip(*sorted(zip(durations, audio_paths, texts)))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def get_max_label_length(self, partition, is_bi_graphemes=False):
        if partition == 'train':
            texts = self.train_texts + self.val_texts
        elif partition == 'test':
            texts = self.train_texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        if is_bi_graphemes:
            self.max_label_length = max([len(generate_bi_graphemes_label(text)) for text in texts])
        else:
            self.max_label_length = max([len(text) for text in texts])
        return self.max_label_length

    def get_max_seq_length(self, partition):
        if partition == 'train':
            audio_paths = self.train_audio_paths + self.val_audio_paths
            durations = self.train_durations + self.val_durations
        elif partition == 'test':
            audio_paths = self.train_audio_paths
            durations = self.train_durations
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        max_duration_indexes = durations.index(max(durations))
        max_seq_length = self.featurize(audio_paths[max_duration_indexes]).shape[0]
        self.max_seq_length = max_seq_length
        return max_seq_length

    def prepare_minibatch(self, audio_paths, texts, overwrite=False,
                          is_bi_graphemes=False, seq_length=-1, save_feature_as_csvfile=False):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts),\
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a, overwrite=overwrite, save_feature_as_csvfile=save_feature_as_csvfile) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        if seq_length == -1:
            x = np.zeros((mb_size, self.max_seq_length, feature_dim))
        else:
            x = np.zeros((mb_size, seq_length, feature_dim))
        y = np.zeros((mb_size, self.max_label_length))
        labelUtil = LabelUtil.getInstance()
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            if is_bi_graphemes:
                label = generate_bi_graphemes_label(texts[i])
                label = labelUtil.convert_bi_graphemes_to_num(label)
                y[i, :len(label)] = label
            else:
                label = labelUtil.convert_word_to_num(texts[i])
                y[i, :len(texts[i])] = label
            label_lengths.append(len(label))
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths,  # list(int) Length of each label
        }

    def iterate_test(self, minibatch_size=16):
        return self.iterate(self.test_audio_paths, self.test_texts,
                            minibatch_size)

    def iterate_validation(self, minibatch_size=16):
        return self.iterate(self.val_audio_paths, self.val_texts,
                            minibatch_size)

    def preprocess_sample_normalize(self, threadIndex, audio_paths, overwrite, return_dict):
        if len(audio_paths) > 0:
            audio_clip = audio_paths[0]
            feat = self.featurize(audio_clip=audio_clip, overwrite=overwrite)
            feat_squared = np.square(feat)
            count = float(feat.shape[0])
            dim = feat.shape[1]
            if len(audio_paths) > 1:
                for audio_path in audio_paths[1:]:
                    next_feat = self.featurize(audio_clip=audio_path, overwrite=overwrite)
                    next_feat_squared = np.square(next_feat)
                    feat_vertically_stacked = np.concatenate((feat, next_feat)).reshape(-1, dim)
                    feat = np.sum(feat_vertically_stacked, axis=0, keepdims=True)
                    feat_squared_vertically_stacked = np.concatenate(
                        (feat_squared, next_feat_squared)).reshape(-1, dim)
                    feat_squared = np.sum(feat_squared_vertically_stacked, axis=0, keepdims=True)
                    count += float(next_feat.shape[0])
            return_dict[threadIndex] = {'feat': feat, 'feat_squared': feat_squared, 'count': count}

    def sample_normalize(self, k_samples=1000, overwrite=False):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        log = logUtil.getlogger()
        log.info("Calculating mean and std from samples")
        # if k_samples is negative then it goes through total dataset
        if k_samples < 0:
            audio_paths = self.audio_paths

        # using sample
        else:
            k_samples = min(k_samples, len(self.train_audio_paths))
            samples = self.rng.sample(self.train_audio_paths, k_samples)
            audio_paths = samples
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        for threadIndex in range(cpu_count()):
            proc = Process(target=self.preprocess_sample_normalize, args=(threadIndex, audio_paths, overwrite, return_dict))
            jobs.append(proc)
            proc.start()
        for proc in jobs:
            proc.join()

        feat = np.sum(np.vstack([item['feat'] for item in return_dict.values()]), axis=0)
        count = sum([item['count'] for item in return_dict.values()])
        feat_squared = np.sum(np.vstack([item['feat_squared'] for item in return_dict.values()]), axis=0)

        self.feats_mean = feat / float(count)
        self.feats_std = np.sqrt(feat_squared / float(count) - np.square(self.feats_mean))
        np.savetxt(
            generate_file_path(self.save_dir, self.model_name, 'feats_mean'), self.feats_mean)
        np.savetxt(
            generate_file_path(self.save_dir, self.model_name, 'feats_std'), self.feats_std)
        log.info("End calculating mean and std from samples")
