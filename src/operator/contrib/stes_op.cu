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
  *  Copyright (c) 2019 by Contributors
  * \file stes_op.cu
  * \Straight-through-estimators round and sign operators.
  * \author Itay Golan
  */

#include "stes_op.h"

namespace mxnet {
namespace op {

// Round STE
NNVM_REGISTER_OP(_contrib_round_ste)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::round>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::round>);

// Sign STE
NNVM_REGISTER_OP(_contrib_sign_ste)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::sign>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::sign>);

}  // namespace op
}  // namespace mxnet
