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
"""Dataset sampler."""
__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler', 'BatchSampler']

import random

class Sampler(object):
    """Base class for samplers.

    All samplers should subclass `Sampler` and define `__iter__` and `__len__`
    methods.
    """
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements from [0, length) sequentially.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, length):
        self._length = length

    def __iter__(self):
        return iter(range(self._length))

    def __len__(self):
        return self._length


class RandomSampler(Sampler):
    """Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, length):
        self._length = length

    def __iter__(self):
        indices = list(range(self._length))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self._length


class BatchSampler(Sampler):
    """Wraps over another `Sampler` and return mini-batches of samples.

    Parameters
    ----------
    sampler : Sampler
        The source Sampler.
    batch_size : int
        Size of mini-batch.
    last_batch : {'keep', 'discard', 'rollover'}
        Specifies how the last batch is handled if batch_size does not evenly
        divide sequence length.

        If 'keep', the last batch will be returned directly, but will contain
        less element than `batch_size` requires.

        If 'discard', the last batch will be discarded.

        If 'rollover', the remaining elements will be rolled over to the next
        iteration.

    Examples
    --------
    >>> sampler = gluon.data.SequentialSampler(10)
    >>> batch_sampler = gluon.data.BatchSampler(sampler, 3, 'keep')
    >>> list(batch_sampler)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    def __init__(self, sampler, batch_size, last_batch='keep'):
        self._sampler = sampler
        self._batch_size = batch_size
        self._last_batch = last_batch
        self._prev = []

    def __iter__(self):
        batch, self._prev = self._prev, []
        for i in self._sampler:
            batch.append(i)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch:
            if self._last_batch == 'keep':
                yield batch
            elif self._last_batch == 'discard':
                return
            elif self._last_batch == 'rollover':
                self._prev = batch
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', 'discard', or 'rollover', " \
                    "but got %s"%self._last_batch)

    def __len__(self):
        if self._last_batch == 'keep':
            return (len(self._sampler) + self._batch_size - 1) // self._batch_size
        if self._last_batch == 'discard':
            return len(self._sampler) // self._batch_size
        if self._last_batch == 'rollover':
            return (len(self._prev) + len(self._sampler)) // self._batch_size
        raise ValueError(
            "last_batch must be one of 'keep', 'discard', or 'rollover', " \
            "but got %s"%self._last_batch)
