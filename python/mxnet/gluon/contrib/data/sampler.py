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
__all__ = ['IntervalSampler']

from ...data import sampler

class IntervalSampler(sampler.Sampler):
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
        assert interval < length, \
            "Interval {} must be smaller than length {}".format(interval, length)
        self._length = length
        self._interval = interval
        self._rollover = rollover

    def __iter__(self):
        for i in range(self._interval if self._rollover else 1):
            for j in range(i, self._length, self._interval):
                yield j

    def __len__(self):
        return self._length
