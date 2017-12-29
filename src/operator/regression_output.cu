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
 * Copyright (c) 2015 by Contributors
 * \file regression_output.cu
 * \brief regression output operator
*/
#include "./regression_output-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateRegressionOutputOp<gpu>(reg_enum::RegressionOutputType type,
                                        RegressionOutputParam param) {
  switch (type) {
    case reg_enum::kLinear:
      return new RegressionOutputOp<gpu, op::mshadow_op::identity, op::mshadow_op::minus>(param);
    case reg_enum::kLogistic:
      return new RegressionOutputOp<gpu, mshadow_op::sigmoid, op::mshadow_op::minus>(param);
    case reg_enum::kMAE:
      return new RegressionOutputOp<gpu, op::mshadow_op::identity, mshadow_op::minus_sign>(param);
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return NULL;
}
}  // namespace op
}  // namespace mxnet

