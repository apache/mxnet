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

__all__ = ['CorpusReader', 'WordLanguageReader']

import io
import os

from ..dataset import SimpleDataset
from ..datareader import DataReader
from .utils import flatten_samples, collate, collate_pad_length

class CorpusReader(DataReader):
    """Text reader that reads a whole corpus and produces a dataset based on provided
    sample splitter and word tokenizer.

    The returned dataset includes samples, each of which can either be a list of tokens if tokenizer
    is specified, or a single string segment from the result of sample_splitter.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    encoding : str, default 'utf8'
        File encoding format.
    flatten : bool, default False
        Whether to return all samples as flattened tokens. If True, each sample is a token.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function or None, default str.split
        A function that splits each sample string into list of tokens. If None, raw samples are
        returned according to `sample_splitter`.
    """
    def __init__(self, filename, encoding='utf8', flatten=False,
                 sample_splitter=lambda s: s.splitlines(),
                 tokenizer=lambda s: s.split()):
        assert sample_splitter, 'sample_splitter must be specified.'
        self._filename = os.path.expanduser(filename)
        self._encoding = encoding
        self._flatten = flatten
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer

    def read(self):
        with io.open(self._filename, 'r', encoding=self._encoding) as fin:
            content = fin.read()
        samples = (s.strip() for s in self._sample_splitter(content))
        if self._tokenizer:
            samples = [self._tokenizer(s) for s in samples if s]
            if self._flatten:
                samples = flatten_samples(samples)
        else:
            samples = [s for s in samples if s]
        return SimpleDataset(samples)


class WordLanguageReader(CorpusReader):
    """Text reader that reads a whole corpus and produces a language modeling dataset given
    the provided sample splitter and word tokenizer.

    The returned dataset includes data (current word) and label (next word).

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
        The length of each of the samples regardless of sample boundary.
        If None, samples are divided according to `sample_splitter` only,
        and may have variable lengths.
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
        assert tokenizer, "Tokenizer must be specified for reading word language model corpus."
        super(WordLanguageReader, self).__init__(filename, encoding, False,
                                                 sample_splitter, tokenizer)
        def process(s):
            tokens = [bos] if bos else []
            tokens.extend(s)
            if eos:
                tokens.append(eos)
            return tokens
        self._seq_len = seq_len
        self._process = process
        self._pad = pad

    def read(self):
        samples = super(WordLanguageReader, self).read()
        samples = [self._process(s) for s in samples]
        if self._seq_len:
            samples = flatten_samples(samples)
            pad_len = collate_pad_length(len(samples), self._seq_len+1, 1)
            if self._pad:
                samples.extend([self._pad] * pad_len)
            samples = collate(samples, self._seq_len+1, 1)

        return SimpleDataset(samples).transform(lambda x: (x[:-1], x[1:]))
