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

"""Numpy index tricks."""
from __future__ import absolute_import
import math
from . import _op as _mx_nd_np

__all__ = ['mgrid', 'ogrid']


# pylint: disable=useless-object-inheritance, too-few-public-methods
class nd_grid(object):
    """
    Construct a multi-dimensional "meshgrid".
    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.
    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.
    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.
    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.
    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`, approximately defined as::
        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)
    Users should use these pre-defined instances instead of using `nd_grid`
    directly.
    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):  # pylint: disable=too-many-branches
        try:
            size = []
            typ = int
            for k in range(len(key)):  # pylint: disable=consider-using-enumerate
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    size.append(int(abs(step)))
                    typ = float
                else:
                    size.append(int(math.ceil((key[k].stop - start)/(step*1.0))))
                if (isinstance(step, float) or
                    isinstance(start, float) or
                    isinstance(key[k].stop, float)):
                    typ = float
            if self.sparse:
                nn = [_mx_nd_np.arange(_x, dtype=_t)
                      for _x, _t in zip(size, (typ,)*len(size))]
            else:
                nn = _mx_nd_np.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    step = int(abs(step))
                    if step != 1:
                        step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [1]*len(size)
                for k in range(len(size)):
                    slobj[k] = -1
                    nn[k] = nn[k].reshape(tuple(slobj))
                    slobj[k] = 1
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):  # pylint: disable=no-else-return
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                stop = key.stop + step
                return _mx_nd_np.arange(0, length, 1, float)*step + start
            else:
                return _mx_nd_np.arange(start, stop, step)
# pylint: enable=useless-object-inheritance, too-few-public-methods


class MGridClass(nd_grid):  # pylint: disable=too-few-public-methods
    """
    `nd_grid` instance which returns a dense multi-dimensional "meshgrid".
    An instance of `python.mxnet.numpy.index_tricks.nd_grid` which returns
    an dense(or fleshed out) mesh-grid when indexed, so that each returned
    argument has the same shape.  The dimensions and number of the output
    arrays are equal to the number of indexing dimensions. If the step length
    is not a complex number, then the stop is not inclusive.
    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    ----------
    mesh-grid `ndarrays` all of the same dimensions

    Examples
    --------
    >>> np.mgrid[0:5,0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],

           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]], dtype=int64)

    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ], dtype=float64)
    """
    def __init__(self):
        super(MGridClass, self).__init__(sparse=False)


mgrid = MGridClass()


class OGridClass(nd_grid):  # pylint: disable=too-few-public-methods
    """
    `nd_grid` instance which returns an open multi-dimensional "meshgrid".
    An instance of `python.mxnet.numpy.index_tricks.nd_grid` which returns an open
    (i.e. not fleshed out) mesh-grid when indexed, so that only one dimension
    of each returned array is greater than 1.  The dimension and number of the
    output arrays are equal to the number of indexing dimensions.  If the step
    length is not a complex number, then the stop is not inclusive.
    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid
        `ndarrays` with only one dimension not equal to 1

    Examples
    --------
    >>> np.ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ], dtype=float64)
    >>> np.ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]], dtype=int64), array([[0, 1, 2, 3, 4]], dtype=int64)]
    """
    def __init__(self):
        super(OGridClass, self).__init__(sparse=True)


ogrid = OGridClass()
