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

import os
import zipfile
import shutil
import numpy as np

from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base

class _LanguageModelDataset(dataset._DownloadedDataset): # pylint: disable=abstract-method
    def __init__(self, root, namespace, vocabulary):
        self._vocab = vocabulary
        self._counter = None
        self._namespace = namespace
        super(_LanguageModelDataset, self).__init__(root, None)

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def frequencies(self):
        return self._counter

    def _build_vocab(self, content):
        if not self._counter:
            self._counter = text.utils.count_tokens_from_str(content)
        if not self._vocab:
            self._vocab = text.vocab.Vocabulary(counter=self.frequencies,
                                                reserved_tokens=[C.EOS_TOKEN])


class _WikiText(_LanguageModelDataset):

    def _read_batch(self, filename):
        with open(filename, 'r', encoding='utf8') as fin:
            content = fin.read()
        self._build_vocab(content)

        raw_data = [line for line in [x.strip().split() for x in content.splitlines()]
                    if line]
        for line in raw_data:
            line.append(C.EOS_TOKEN)
        raw_data = self.vocabulary.to_indices([x for line in raw_data for x in line if x])
        data = raw_data[0:-1]
        label = raw_data[1:]
        return np.array(data, dtype=np.int32), np.array(label, dtype=np.int32)

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
        data_file_name, data_hash = self._data_file[self._segment]
        path = os.path.join(self._root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            namespace = 'gluon/dataset/'+self._namespace
            downloaded_file_path = download(_get_repo_file_url(namespace, archive_file_name),
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

        data, label = self._read_batch(path)

        # https://github.com/apache/incubator-mxnet/issues/18886 breaks this unless array size is
        # multiple of self._seq_len. Truncating the source is consistent with pre #18886 outcome
        seq_len_mult = len(data) // self._seq_len * self._seq_len
        self._data = nd.array(data, dtype=data.dtype)[:seq_len_mult].reshape((-1, self._seq_len))
        self._label = nd.array(label, dtype=label.dtype)[:seq_len_mult].reshape((-1, self._seq_len))

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)


class WikiText2(_WikiText):
    """WikiText-2 word-level dataset for language modeling, from Salesforce research.

    From
    https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/wikitext-2
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    vocab : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        The vocabulary to use for indexing the text dataset.
        If None, a default vocabulary is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.

    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'wikitext-2'),
                 segment='train', vocab=None, seq_len=35):
        self._archive_file = ('wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')
        self._data_file = {'train': ('wiki.train.tokens',
                                     '863f29c46ef9d167fff4940ec821195882fe29d1'),
                           'validation': ('wiki.valid.tokens',
                                          '0418625c8b4da6e4b5c7a0b9e78d4ae8f7ee5422'),
                           'test': ('wiki.test.tokens',
                                    'c7b8ce0aa086fb34dab808c5c49224211eb2b172')}
        self._segment = segment
        self._seq_len = seq_len
        super(WikiText2, self).__init__(root, 'wikitext-2', vocab)


class WikiText103(_WikiText):
    """WikiText-103 word-level dataset for language modeling, from Salesforce research.

    From
    https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/wikitext-103
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    vocab : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        The vocabulary to use for indexing the text dataset.
        If None, a default vocabulary is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.
    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'wikitext-103'),
                 segment='train', vocab=None, seq_len=35):
        self._archive_file = ('wikitext-103-v1.zip', '0aec09a7537b58d4bb65362fee27650eeaba625a')
        self._data_file = {'train': ('wiki.train.tokens',
                                     'b7497e2dfe77e72cfef5e3dbc61b7b53712ac211'),
                           'validation': ('wiki.valid.tokens',
                                          'c326ac59dc587676d58c422eb8a03e119582f92b'),
                           'test': ('wiki.test.tokens',
                                    '8a5befc548865cec54ed4273cf87dbbad60d1e47')}
        self._segment = segment
        self._seq_len = seq_len
        super(WikiText103, self).__init__(root, 'wikitext-103', vocab)
