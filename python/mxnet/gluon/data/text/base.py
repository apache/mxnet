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

__all__ = ['TextTokenReader']

import io
import os

from ..dataset import Dataset
from ..datareader import DataReader

class TextTokenReader(DataReader):
    """Text reader that produces lists of tokens.

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
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is added.
    """
    def __init__(self, filename, encoding='utf8', sample_splitter=lambda s: s.splitlines(),
                 tokenizer=lambda s: s.split(), bos=None, eos=None):
        filename = os.path.expanduser(filename)
        with io.open(filename, 'r', encoding=encoding) as fin:
            content = fin.read()

        samples = [s for s in [tokenizer(x) for x in sample_splitter(content)] if s]
        if bos or eos:
            for tokens in samples:
                if bos:
                    tokens.insert(0, bos)
                if eos:
                    tokens.append(eos)

        self._samples = samples

    def __len__(self):
        return self._samples.__len__

    def __getitem__(self, idx):
        return self._samples[idx]

class TextDataset(Dataset):
    """Abstract dataset class for text data.

    Subclasses need to override `__getitem__`, which returns the i-th
    element, and `__len__`, which returns the total number elements."""

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _flatten(self, samples):
        """Flatten lists of tokens into a single list of tokens."""
        return [token for sample in samples for token in sample if token]

    def _collate(self, flat_sample, seq_len):
        num_samples = len(flat_sample) // seq_len
        return [flat_sample[i*seq_len:(i+1)*seq_len] for i in range(num_samples)]
