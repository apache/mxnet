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

"""Base classes for text datasets and readers."""

__all__ = ['WordLanguageReader']

import io
import os

from ..dataset import SimpleDataset
from ..datareader import DataReader
from .utils import flatten_samples, collate, pair

class WordLanguageReader(DataReader):
    """Text reader that reads a whole corpus and produces samples based on provided sample splitter
    and word tokenizer.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    encoding : str, default 'utf8'
        File encoding format.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function, default str.split
        A function that splits each sample string into list of tokens.
    seq_len : int or None
        The length of each of the samples. If None, samples are divided according to
        `sample_splitter` only, and may have variable lengths.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is added.
    """
    def __init__(self, filename, encoding='utf8', sample_splitter=lambda s: s.splitlines(),
                 tokenizer=lambda s: s.split(), seq_len=None, bos=None, eos=None):
        self._filename = os.path.expanduser(filename)
        self._encoding = encoding
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer

        if bos and eos:
            def process(s):
                s.insert(0, bos)
                s.append(eos)
                return pair(s)
        elif bos:
            def process(s):
                s.insert(0, bos)
                return pair(s)
        elif eos:
            def process(s):
                s.append(eos)
                return pair(s)
        else:
            def process(s):
                return pair(s)
        self._process = process
        self._seq_len = seq_len

    def read(self):
        with io.open(self._filename, 'r', encoding=self._encoding) as fin:
            content = fin.read()
        samples = [s.strip() for s in self._sample_splitter(content)]
        samples = [self._process(self._tokenizer(s)) for s in samples if s]
        if self._seq_len:
            samples = collate(flatten_samples(samples), self._seq_len)
        samples = [list(zip(*s)) for s in samples]
        return SimpleDataset(samples)
