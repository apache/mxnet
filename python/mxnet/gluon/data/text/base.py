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
    pad : str or None, default None
        The padding token to add at the end of dataset if `seq_len` is specified and the total
        number of tokens in the corpus don't evenly divide `seq_len`. If pad is None or seq_len
        is None, no padding is added. Otherwise, padding token is added to the last sample if
        its length is less than `seq_len`. If `pad` is None and `seq_len` is specified, the last
        sample is discarded if it's shorter than `seq_len`.
    """
    def __init__(self, filename, encoding='utf8', sample_splitter=lambda s: s.splitlines(),
                 tokenizer=lambda s: s.split(), seq_len=None, bos=None, eos=None, pad=None):
        self._filename = os.path.expanduser(filename)
        self._encoding = encoding
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer

        if bos and eos:
            def process(s):
                out = [bos]
                out.extend(s)
                out.append(eos)
                return pair(out)
        elif bos:
            def process(s):
                out = [bos]
                out.extend(s)
                return pair(out)
        elif eos:
            def process(s):
                s.append(eos)
                return pair(s)
        else:
            def process(s):
                return pair(s)
        self._process = process
        self._seq_len = seq_len
        self._pad = pad

    def read(self):
        with io.open(self._filename, 'r', encoding=self._encoding) as fin:
            content = fin.read()
        samples = [s.strip() for s in self._sample_splitter(content)]
        samples = [self._process(self._tokenizer(s)) for s in samples if s]
        if self._seq_len:
            samples = flatten_samples(samples)
            if self._pad and len(samples) % self._seq_len:
                pad_len = self._seq_len - len(samples) % self._seq_len
                samples.extend([self._pad] * pad_len)
            samples = collate(samples, self._seq_len)
        samples = [list(zip(*s)) for s in samples]
        return SimpleDataset(samples)
