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
 *  Copyright (c) 2018 by Contributors
 * \file default_subgraph_op.cu
 * \brief GPU Implementation of subgraph operations
 */

#include <mxnet/ndarray.h>
#include "./common.h"
#include "../../imperative/imperative_utils.h"
#include "../../imperative/cached_op.h"

namespace mxnet {
namespace op {

void DefaultSubgraphOpForward(const OpStatePtr& state_ptr,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs);

NNVM_REGISTER_OP(_default_subgraph_op)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", DefaultSubgraphOpForward);

}  // namespace op
}  // namespace mxnet
