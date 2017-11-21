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
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op.cu
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sgd_update)
.set_attr<FCompute>("FCompute<gpu>", SGDUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SGDUpdateEx<gpu>);

NNVM_REGISTER_OP(sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", SGDMomUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SGDMomUpdateEx<gpu>);

NNVM_REGISTER_OP(mp_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", MP_SGDUpdate<gpu>);

NNVM_REGISTER_OP(mp_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", MP_SGDMomUpdate<gpu>);

NNVM_REGISTER_OP(adam_update)
.set_attr<FCompute>("FCompute<gpu>", AdamUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", AdamUpdateEx<gpu>);

NNVM_REGISTER_OP(rmsprop_update)
.set_attr<FCompute>("FCompute<gpu>", RMSPropUpdate<gpu>);

NNVM_REGISTER_OP(rmspropalex_update)
.set_attr<FCompute>("FCompute<gpu>", RMSPropAlexUpdate<gpu>);

NNVM_REGISTER_OP(ftrl_update)
.set_attr<FCompute>("FCompute<gpu>", FtrlUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", FtrlUpdateEx<gpu>);

}  // namespace op
}  // namespace mxnet
