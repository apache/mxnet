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
  *  Copyright (c) 2017 by Contributors
  * \file bounding_box.cu
  * \brief Bounding box util functions and operators
  * \author Joshua Zhang
  */

#include "./bounding_box-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSBackward<gpu>);

NNVM_REGISTER_OP(_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapBackward<gpu>);

NNVM_REGISTER_OP(_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingBackward<gpu>);
}  // namespace op
}  // namespace mxnet
