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
 * Copyright (c) 2020 by Contributors
 * \file np_matrix_rank.cu
 * \brief GPU implementation of the matrix_rank Operator
 */
#include <mxnet/operator_util.h>
#include "./np_matrix_rank-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_matrix_rank_none_tol)
.set_attr<FCompute>("FCompute<gpu>", MatrixRankNoneTolForward<gpu>);

NNVM_REGISTER_OP(_npi_matrix_rank)
.set_attr<FCompute>("FCompute<gpu>", MatrixRankForward<gpu>);

}  // namespace op
}  // namespace mxnet
