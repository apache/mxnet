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
__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler', 'FilterSampler', 'BatchSampler',
           'IntervalSampler']

import numpy as np

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
    """Samples elements from [start, start+length) sequentially.

    Parameters
    ----------
    length : int
        Length of the sequence.
    start : int, default is 0
        The start of the sequence index.
    """
    def __init__(self, length, start=0):
        self._length = length
        self._start = start

    def __iter__(self):
        return iter(range(self._start, self._start + self._length))

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
        indices = np.arange(self._length)
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self._length

class FilterSampler(Sampler):
    """Samples elements from a Dataset for which `fn` returns True.

    Parameters
    ----------
    fn : callable
        A callable function that takes a sample and returns a boolean
    dataset : Dataset
        The dataset to filter.
    """
    def __init__(self, fn, dataset):
        self._fn = fn
        self._dataset = dataset
        self._indices = [i for i, sample in enumerate(dataset) if fn(sample)]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


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


class IntervalSampler(Sampler):
    """Samples elements from [0, length) at fixed intervals.

    Parameters
    ----------
    length : int
        Length of the sequence.
    interval : int
        The number of items to skip between two samples.
    rollover : bool, default True
        Whether to start again from the first skipped item after reaching the end.
        If true, this sampler would start again from the first skipped item until all items
        are visited.
        Otherwise, iteration stops when end is reached and skipped items are ignored.

    Examples
    --------
    >>> sampler = contrib.data.IntervalSampler(13, interval=3)
    >>> list(sampler)
    [0, 3, 6, 9, 12, 1, 4, 7, 10, 2, 5, 8, 11]
    >>> sampler = contrib.data.IntervalSampler(13, interval=3, rollover=False)
    >>> list(sampler)
    [0, 3, 6, 9, 12]
    """
    def __init__(self, length, interval, rollover=True):
        assert interval <= length, \
            "Interval {} must be smaller than or equal to length {}".format(interval, length)
        self._length = length
        self._interval = interval
        self._rollover = rollover

    def __iter__(self):
        for i in range(self._interval if self._rollover else 1):
            for j in range(i, self._length, self._interval):
                yield j

    def __len__(self):
        return self._length
