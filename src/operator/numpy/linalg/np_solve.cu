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
 * \file np_solve.cu
 * \brief GPU implementation of the Solve Operator
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "./np_solve-inl.h"

namespace mxnet {
namespace op {

#if MXNET_USE_CUSOLVER == 1

NNVM_REGISTER_OP(_npi_solve)
.set_attr<FCompute>("FCompute<gpu>", LaOpForwardSolve<gpu, 2, 2, 2, 1, solve>);

NNVM_REGISTER_OP(_backward_npi_solve)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackwardSolve<gpu, 2, 2, 4, 2, solve_backward>);

#endif

}  // namespace op
}  // namespace mxnet
