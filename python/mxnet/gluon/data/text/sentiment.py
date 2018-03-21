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
"""Sentiment analysis datasets."""

__all__ = ['IMDB']

import glob
import io
import json
import os
import tarfile

from ..dataset import SimpleDataset
from ...utils import download, check_sha1, _get_repo_file_url


class IMDB(SimpleDataset):
    """IMDB reviews for sentiment analysis.

    From
    http://ai.stanford.edu/~amaas/data/sentiment/

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imdb'
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'test', and 'unsup' for unsupervised.
    """
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'imdb'),
                 segment='train'):
        self._data_file = {'train': ('train.json',
                                     '516a0ba06bca4e32ee11da2e129f4f871dff85dc'),
                           'test': ('test.json',
                                    '7d59bd8899841afdc1c75242815260467495b64a'),
                           'unsup': ('unsup.json',
                                     'f908a632b7e7d7ecf113f74c968ef03fadfc3c6c')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(IMDB, self).__init__(self._read_data())


    def _get_data(self):
        data_file_name, data_hash = self._data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url('gluon/dataset/imdb', data_file_name),
                                        path=root, sha1_hash=data_hash)


    def _read_data(self):
        with open(os.path.join(self._root, self._segment+'.json')) as f:
            samples = json.load(f)
        return samples
