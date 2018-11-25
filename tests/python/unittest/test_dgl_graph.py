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

# pylint: skip-file
from __future__ import print_function
import numpy as np
import scipy as sp
import mxnet as mx
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
import unittest

def test_edge_id():
    shape = rand_shape_2d()
    data = rand_ndarray(shape, stype='csr', density=0.4)
    ground_truth = np.zeros(shape, dtype=np.float32)
    ground_truth -= 1.0
    indptr_np = data.indptr.asnumpy()
    data_np = data.data.asnumpy()
    indices_np = data.indices.asnumpy()
    for i in range(shape[0]):
        for j in range(indptr_np[i], indptr_np[i+1]):
            idx = indices_np[j]
            ground_truth[i, idx] = data_np[j]

    np_u = np.random.randint(0, shape[0], size=(5, ))
    np_v = np.random.randint(0, shape[1], size=(5, ))
    mx_u = mx.nd.array(np_u)
    mx_v = mx.nd.array(np_v)
    assert_almost_equal(mx.nd.contrib.edge_id(data, mx_u, mx_v).asnumpy(),
                        ground_truth[np_u, np_v], rtol=1e-5, atol=1e-6)

def generate_graph(n):
    arr = sp.sparse.random(n, n, density=0.2, format='coo')
    arr.data = np.arange(0, len(arr.row), dtype=np.float32)
    return arr.tocsr(), mx.nd.sparse.csr_matrix(arr.tocsr()).astype(np.int64)

def test_subgraph():
    sp_g, g = generate_graph(100)
    vertices = np.unique(np.random.randint(0, 100, size=(20)))
    subgs = mx.nd.contrib.dgl_subgraph(g, mx.nd.array(vertices, dtype=np.int64),
                                       return_mapping=True)
    subgs[0].check_format()
    subgs[1].check_format()
    assert_array_equal(subgs[0].indptr, subgs[1].indptr)
    assert_array_equal(subgs[0].indices, subgs[1].indices)
    sp_subg = subgs[1].asscipy()
    for i in range(len(subgs[0].indptr) - 1):
        subv1 = i
        v1 = vertices[subv1]
        row_start = int(subgs[0].indptr[subv1].asnumpy()[0])
        row_end = int(subgs[0].indptr[subv1 + 1].asnumpy()[0])
        if row_start >= len(subgs[0].indices):
            remain = subgs[0].indptr[subv1:].asnumpy()
            assert np.sum(remain == row_start) == len(remain)
            break
        row = subgs[0].indices[row_start:row_end]
        for j, subv2 in enumerate(row.asnumpy()):
            v2 = vertices[subv2]
            assert sp_g[v1, v2] == sp_subg[subv1, subv2]

def test_adjacency():
    sp_g, g = generate_graph(100)
    start = time.time()
    adj = mx.nd.contrib.dgl_adjacency(g)
    assert adj.dtype == np.float32
    assert adj.shape == g.shape
    assert_array_equal(adj.indptr, g.indptr)
    assert_array_equal(adj.indices, g.indices)
    assert_array_equal(adj.data, mx.nd.ones(shape=g.indices.shape))

if __name__ == "__main__":
    import nose
    nose.runmodule()
