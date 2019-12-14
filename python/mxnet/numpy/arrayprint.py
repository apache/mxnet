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

"""ndarray print format controller."""

from __future__ import absolute_import, print_function

import numpy as onp
from ..util import set_module

__all__ = ['set_printoptions']


@set_module('mxnet.numpy')
def set_printoptions(precision=None, threshold=None, **kwarg):
    """
    Set printing options.

    These options determine the way floating point numbers and arrays are displayed.

    Parameters
    ----------
    precision : int or None, optional
        Number of digits of precision for floating point output (default 8).
        May be `None` if `floatmode` is not `fixed`, to print as many digits as
        necessary to uniquely specify the value.
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).

    Examples
    --------
    Floating point precision can be set:

    >>> np.set_printoptions(precision=4)
    >>> print(np.array([1.123456789]))
    [ 1.1235]

    Long arrays can be summarised:

    >>> np.set_printoptions(threshold=5)
    >>> print(np.arange(10))
    [0. 1. 2. ... 7. 8. 9.]
    """
    if kwarg:
        raise NotImplementedError('mxnet.numpy.set_printoptions only supports parameters'
                                  ' precision and threshold for now.')
    onp.set_printoptions(precision=precision, threshold=threshold, **kwarg)
