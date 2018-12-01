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

# coding: utf-8
# pylint: disable=
""" Audio Dataset container."""
from __future__ import print_function
__all__ = ['AudioFolderDataset']

import os
import warnings
from itertools import islice
import csv
from mxnet.gluon.data import Dataset
from mxnet import ndarray as nd
try:
    import librosa
except ImportError as e:
    raise ImportError("librosa dependency could not be resolved or \
    imported, could not load audio onto the numpy array. pip install librosa")



class AudioFolderDataset(Dataset):
    """A dataset for loading Audio files stored in a folder structure like::

        root/children_playing/0.wav
        root/siren/23.wav
        root/drilling/26.wav
        root/dog_barking/42.wav
            OR
        Files(wav) and a csv file that has file name and associated label

    Parameters
    ----------
    root : str
        Path to root directory.
    transform : callable, default None
        A function that takes data and label and transforms them
    train_csv: str, default None
       train_csv should be populated by the training csv filename
    file_format: str, default '.wav'
        The format of the audio files(.wav)
    skip_header: boolean, default False
        While reading from csv file, whether to skip at the start of the file to avoid reading in header


    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the  `i`th label
    items : list of tuples
        List of all audio in (filename, label) pairs.

    """
    def __init__(self, root, train_csv=None, file_format='.wav', skip_header=False):
        if not librosa:
            warnings.warn("pip install librosa to continue.")
            raise RuntimeError("Librosa not installed. Run pip install librosa and retry this step.")
        self._root = os.path.expanduser(root)
        self._exts = ['.wav']
        self._format = file_format
        self._train_csv = train_csv
        if file_format.lower() not in self._exts:
            raise RuntimeError("Format {} not supported currently.".format(file_format))
        skip_rows = 0
        if skip_header:
            skip_rows = 1
        self._list_audio_files(self._root, skip_rows=skip_rows)


    def _list_audio_files(self, root, skip_rows=0):
        """Populates synsets - a map of index to label for the data items.
        Populates the data in the dataset, making tuples of (data, label)
        """
        self.synsets = []
        self.items = []
        if not self._train_csv:
            # The audio files are organized in folder structure with
            # directory name as label and audios in them
            self._folder_structure(root)
        else:
            # train_csv contains mapping between filename and label
            self._csv_labelled_dataset(root, skip_rows=skip_rows)

        # Generating the synset.txt file now
        if not os.path.exists("./synset.txt"):
            with open("./synset.txt", "w") as synsets_file:
                for item in self.synsets:
                    synsets_file.write(item+os.linesep)
            print("Synsets is generated as synset.txt")
        else:
            warnings.warn("Synset file already exists in the current directory! Not generating synset.txt.")


    def _folder_structure(self, root):
        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring {}, which is not a directory.'.format(path))
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                file_name = os.path.join(path, filename)
                ext = os.path.splitext(file_name)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring {} of type {}. Only support {}'\
                    .format(filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((file_name, label))


    def _csv_labelled_dataset(self, root, skip_rows=0):
        with open(self._train_csv, "r") as traincsv:
            for line in islice(csv.reader(traincsv), skip_rows, None):
                filename = os.path.join(root, line[0])
                label = line[1].strip()
                if label not in self.synsets:
                    self.synsets.append(label)
                if self._format not in filename:
                    filename = filename+self._format
                self.items.append((filename, nd.array([self.synsets.index(label)]).reshape((1,))))


    def __getitem__(self, idx):
        """Retrieve the item (data, label) stored at idx in items"""
        filename, label = self.items[idx]
        # resampling_type is passed as kaiser_fast for a better performance
        X1, _ = librosa.load(filename, res_type='kaiser_fast')
        return nd.array(X1), label


    def __len__(self):
        """Retrieves the number of items in the dataset"""
        return len(self.items)


    def transform_first(self, fn, lazy=False):
        """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.
        lazy=False is passed to transform_first for dataset so that all tramsforms could be performed in
        one shot and not during training. This is a performance consideration.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first element of a sample
            as input and returns the transformed element.
        lazy : bool, default False
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.

        """
        return super(AudioFolderDataset, self).transform_first(fn, lazy=lazy)
