/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file csr_neighborhood_sample.cc
 * \brief
 */
#include "./csr_neighborhood_sample-inl.h"

namespace mxnet {
namespace op {

/*
Usage:

import mxnet as mx
import numpy as np

shape = (5, 5)
data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)
indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)
indptr_np = np.array([0, 4,8,12,16,20], dtype=np.int64)
a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
a.asnumpy()

seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=1, num_neighbor=2, max_num_vertices=5)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=1, num_neighbor=1, max_num_vertices=4)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=2, num_neighbor=1, max_num_vertices=4)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0,2,4], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=1, num_neighbor=2, max_num_vertices=5)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0,4], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=1, num_neighbor=2, max_num_vertices=5)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0,4], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=2, num_neighbor=2, max_num_vertices=5)
out[0]
out[1].asnumpy()

seed = mx.nd.array([0,4], dtype=np.int64)
out = mx.nd.contrib.neighbor_sample(a, seed, num_hops=1, num_neighbor=2, max_num_vertices=5)
out[0]
out[1].asnumpy()

*/

DMLC_REGISTER_PARAMETER(NeighborSampleParam);

NNVM_REGISTER_OP(_contrib_neighbor_sample)
.MXNET_DESCRIBE("")
.set_attr_parser(ParamParser<NeighborSampleParam>)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FInferStorageType>("FInferStorageType", CSRNeighborSampleStorageType)
.set_attr<nnvm::FInferShape>("FInferShape", CSRNeighborSampleShape)
.set_attr<nnvm::FInferType>("FInferType", CSRNeighborSampleType)
.set_attr<FComputeEx>("FComputeEx<cpu>", CSRNeighborSampleComputeExCPU)
.add_argument("csr_matrix", "NDArray-or-Symbol", "csr matrix")
.add_argument("seed_array", "NDArray-or-Symbol", "seed vertices")
.add_arguments(NeighborSampleParam::__FIELDS__());    

}  // op
}  // mxnet