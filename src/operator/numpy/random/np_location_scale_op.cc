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
 * Copyright (c) 2019 by Contributors
 * \file np_location_scale_op.cc
 * \brief Operator for numpy sampling from location scale distributions.
 */
#include "./np_location_scale_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyLocationScaleParam);

MXNET_OPERATOR_REGISTER_LOCATION_SCALE(_npi_logistic)
.set_attr<FCompute>("FCompute<cpu>", NumpyLocationScaleForward<cpu,
                    mxnet_op::logistic_two_scalar_kernel, mxnet_op::logistic_one_scalar_kernel,
                    mxnet_op::logistic_kernel>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_broadcast_logistic"});

MXNET_OPERATOR_REGISTER_LOCATION_SCALE_BACKWARD(_backward_broadcast_logistic)

MXNET_OPERATOR_REGISTER_LOCATION_SCALE(_npi_gumbel)
.set_attr<FCompute>("FCompute<cpu>", NumpyLocationScaleForward<cpu,
                    mxnet_op::gumbel_two_scalar_kernel, mxnet_op::gumbel_one_scalar_kernel,
                    mxnet_op::gumbel_kernel>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_broadcast_gumbel"});

MXNET_OPERATOR_REGISTER_LOCATION_SCALE_BACKWARD(_backward_broadcast_gumbel)

}  // namespace op
}  // namespace mxnet
