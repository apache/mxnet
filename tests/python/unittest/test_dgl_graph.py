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

def check_uniform(out, num_hops, max_num_vertices):
    sample_id = out[0]
    sub_csr = out[1]
    layer = out[2]
    # check sample_id
    assert (len(sample_id) == max_num_vertices+1)
    num_vertices = sample_id[-1].asnumpy()[0]
    # check sub_csr
    sub_csr.check_format(full_check=True)
    assert np.all((sub_csr.indptr[num_vertices:] == sub_csr.indptr[num_vertices]).asnumpy())
    # check layer
    for data in layer[:num_vertices]:
        assert(data <= num_hops)

def check_non_uniform(out, num_hops, max_num_vertices):
    sample_id = out[0]
    sub_csr = out[1]
    prob = out[2]
    layer = out[3]
    # check sample_id
    assert (len(sample_id) == max_num_vertices+1)
    num_vertices = sample_id[-1].asnumpy()[0]
    # check sub_csr
    sub_csr.check_format(full_check=True)
    assert np.all((sub_csr.indptr[num_vertices:] == sub_csr.indptr[num_vertices]).asnumpy())
    # check prob
    assert (len(prob) == max_num_vertices)
    # check layer
    for data in layer[:num_vertices]:
        assert(data <= num_hops)

def check_compact(csr, id_arr, num_nodes):
    compact = mx.nd.contrib.dgl_graph_compact(csr, id_arr, graph_sizes=num_nodes, return_mapping=False)
    assert compact.shape[0] == num_nodes
    assert compact.shape[1] == num_nodes
    assert mx.nd.sum(compact.indptr == csr.indptr[0:(num_nodes + 1)]).asnumpy() == num_nodes + 1
    sub_indices = compact.indices.asnumpy()
    indices = csr.indices.asnumpy()
    id_arr = id_arr.asnumpy()
    for i in range(len(sub_indices)):
        sub_id = sub_indices[i]
        assert id_arr[sub_id] == indices[i]

def test_uniform_sample():
    shape = (5, 5)
    data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)
    indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)
    indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)
    a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)

    seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 3)
    check_uniform(out, num_hops=1, max_num_vertices=5)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=1, max_num_vertices=4)
    assert (len(out) == 3)
    check_uniform(out, num_hops=1, max_num_vertices=4)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=2, num_neighbor=1, max_num_vertices=3)
    assert (len(out) == 3)
    check_uniform(out, num_hops=2, max_num_vertices=3)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0,2,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 3)
    check_uniform(out, num_hops=1, max_num_vertices=5)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 3)
    check_uniform(out, num_hops=1, max_num_vertices=5)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=2, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 3)
    check_uniform(out, num_hops=2, max_num_vertices=5)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 3)
    check_uniform(out, num_hops=1, max_num_vertices=5)
    num_nodes = out[0][-1].asnumpy()
    assert num_nodes > 0
    assert num_nodes < len(out[0])
    check_compact(out[1], out[0], num_nodes)

def test_non_uniform_sample():
    shape = (5, 5)
    prob = mx.nd.array([0.9, 0.8, 0.2, 0.4, 0.1], dtype=np.float32)
    data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)
    indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)
    indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)
    a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)

    seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=1, max_num_vertices=5)

    seed = mx.nd.array([0], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=1, max_num_vertices=4)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=1, max_num_vertices=4)

    seed = mx.nd.array([0], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=2, num_neighbor=1, max_num_vertices=4)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=2, max_num_vertices=4)

    seed = mx.nd.array([0,2,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=1, max_num_vertices=5)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=1, max_num_vertices=5)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=2, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=2, max_num_vertices=5)

    seed = mx.nd.array([0,4], dtype=np.int64)
    out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=2, max_num_vertices=5)
    assert (len(out) == 4)
    check_non_uniform(out, num_hops=1, max_num_vertices=5)

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
    adj = mx.nd.contrib.dgl_adjacency(g)
    assert adj.dtype == np.float32
    assert adj.shape == g.shape
    assert_array_equal(adj.indptr, g.indptr)
    assert_array_equal(adj.indices, g.indices)
    assert_array_equal(adj.data, mx.nd.ones(shape=g.indices.shape))

if __name__ == "__main__":
    import nose
    nose.runmodule()
