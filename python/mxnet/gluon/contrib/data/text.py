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
"""Text datasets."""
__all__ = ['WikiText2', 'WikiText103']

import io
import os
import zipfile
import shutil
import numpy as np

from ...data import dataset
from ...utils import download, check_sha1
from ....contrib import text
from .... import nd


class WikiText2(dataset._DownloadedDataset):
    """WikiText-2 word-level dataset for language modeling, from Salesforce research.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/cifar10'
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    indexer : :class:`~mxnet.contrib.text.indexer.TokenIndexer`, default None
        The indexer to use for indexing the text dataset. If None, a default indexer is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::

        transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'wikitext-2'),
                 segment='train', indexer=None, seq_len=35, transform=None):
        self._archive_file = ('wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')
        self._data_file = {'train': ('wiki.train.tokens',
                                     '863f29c46ef9d167fff4940ec821195882fe29d1'),
                           'validation': ('wiki.valid.tokens',
                                          '0418625c8b4da6e4b5c7a0b9e78d4ae8f7ee5422'),
                           'test': ('wiki.test.tokens',
                                    'c7b8ce0aa086fb34dab808c5c49224211eb2b172')}
        self._segment = segment
        self._seq_len = seq_len
        self.indexer = indexer
        super(WikiText2, self).__init__('wikitext-2', root, transform)

    def _read_batch(self, filename):
        with io.open(filename, 'r', encoding='utf8') as fin:
            content = fin.read()
        eos_token = '<eos>'
        if not self.indexer:
            counter = text.utils.count_tokens_from_str(content)
            self.indexer = text.indexer.TokenIndexer(counter=counter,
                                                     reserved_tokens=[eos_token])
        raw_data = [line for line in [x.strip().split() for x in content.splitlines()]
                    if line]
        for line in raw_data:
            line.append(eos_token)
        raw_data = [x for x in line for line in raw_data if x]
        raw_data = self.indexer.to_indices(raw_data)
        data = raw_data[0:-1]
        label = raw_data[1:]
        return np.array(data, dtype=np.int32), np.array(label, dtype=np.int32)


    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
        data_file_name, data_hash = self._data_file[self._segment]
        path = os.path.join(self._root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(self._get_url(archive_file_name),
                                            path=self._root,
                                            sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(self._root, filename)
                        with zf.open(member) as source, \
                             open(dest, "wb") as target:
                            shutil.copyfileobj(source, target)

        data, label = self._read_batch(os.path.join(self._root, data_file_name))

        self._data = nd.array(data, dtype=data.dtype).reshape((-1, self._seq_len))
        self._label = nd.array(label, dtype=label.dtype).reshape((-1, self._seq_len))


class WikiText103(WikiText2):
    """WikiText-103 word-level dataset for language modeling, from Salesforce research.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/cifar10'
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    indexer : :class:`~mxnet.contrib.text.indexer.TokenIndexer`, default None
        The indexer to use for indexing the text dataset. If None, a default indexer is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::

        transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'wikitext-103'),
                 segment='train', indexer=None, seq_len=35, transform=None):
        self._archive_file = ('wikitext-103-v1.zip', '0aec09a7537b58d4bb65362fee27650eeaba625a')
        self._data_file = {'train': ('wiki.train.tokens',
                                     'b7497e2dfe77e72cfef5e3dbc61b7b53712ac211'),
                           'validation': ('wiki.valid.tokens',
                                          'c326ac59dc587676d58c422eb8a03e119582f92b'),
                           'test': ('wiki.test.tokens',
                                    '8a5befc548865cec54ed4273cf87dbbad60d1e47')}
        self._segment = segment
        self._seq_len = seq_len
        self.indexer = indexer
        super(WikiText2, self).__init__('wikitext-103', root, transform) # pylint: disable=bad-super-call
