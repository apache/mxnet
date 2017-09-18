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
 * \file indexing_op.cu
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>);

NNVM_REGISTER_OP(take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpBackward<gpu>);

NNVM_REGISTER_OP(batch_take)
.set_attr<FCompute>("FCompute<gpu>", BatchTakeOpForward<gpu>);

NNVM_REGISTER_OP(one_hot)
.set_attr<FCompute>("FCompute<gpu>", OneHotOpForward<gpu>);

NNVM_REGISTER_OP(gather_nd)
.set_attr<FCompute>("FCompute<gpu>", GatherNDForward<gpu>);

NNVM_REGISTER_OP(scatter_nd)
.set_attr<FCompute>("FCompute<gpu>", ScatterNDForward<gpu>);
}  // namespace op
}  // namespace mxnet
