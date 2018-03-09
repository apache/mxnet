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
__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler', 'BatchSampler', 'AliasMethodSampler']

import random
from ... import nd


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


class AliasMethodSampler(object):
    """ The Alias Method: Efficient Sampling with Many Discrete Outcomes.
    Can be use in NCELoss.

    Parameters
    ----------
    K : int
        Number of events.
    probs : array
        Probability of each events, corresponds to K.

    References
    -----------
        https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, K, probs):
        if K != len(probs):
            raise ValueError("K should be equal to len(probs). K:%d, len(probs):%d" % (K, len(probs)))
        self.K = K
        self.prob = nd.zeros(K)
        self.alias = nd.zeros(K, dtype='int32')

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def draw(self, n):
        """Draw N samples from multinomial
        """
        samples = nd.zeros(n, dtype='int32')

        kk = nd.floor(nd.random.uniform(0, self.K, shape=n), dtype='int32')
        rand = nd.random.uniform(shape=n)

        prob = self.prob[kk]
        alias = self.alias[kk]

        for i in xrange(n):
            if rand[i] < prob[i]:
                samples[i] = kk[i]
            else:
                samples[i] = alias[i]
        return samples

