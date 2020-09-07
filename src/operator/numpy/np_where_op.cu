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
 * Copyright (c) 2017 by Contributors
 * \file np_where_op.cu
 * \brief GPU Implementation of numpy operator where
 */

#include "np_where_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_where)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_where)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereOpBackward<gpu>);

NNVM_REGISTER_OP(_npi_where_lscalar)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereScalarOpForward<gpu, true>);

NNVM_REGISTER_OP(_npi_where_rscalar)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereScalarOpForward<gpu, false>);

NNVM_REGISTER_OP(_backward_np_where_lscalar)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereScalarOpBackward<gpu, true>);

NNVM_REGISTER_OP(_backward_np_where_rscalar)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereScalarOpBackward<gpu, false>);

NNVM_REGISTER_OP(_npi_where_scalar2)
.set_attr<FCompute>("FCompute<gpu>", NumpyWhereScalar2OpForward<gpu>);

}  // namespace op
}  // namespace mxnet
