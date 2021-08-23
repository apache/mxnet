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
 * \file np_init_op.cu
 * \brief GPU Implementation of numpy init op
 */

#include "../tensor/init_op.h"
#include "./np_init_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_zeros)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(_npi_ones)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

NNVM_REGISTER_OP(_npi_identity)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

NNVM_REGISTER_OP(_npi_full_like)
.set_attr<FCompute>("FCompute<gpu>", FullLikeOpCompute<gpu>);

NNVM_REGISTER_OP(_npi_full)
.set_attr<FCompute>("FCompute<gpu>", InitFillWithScalarCompute<gpu>);

NNVM_REGISTER_OP(_npi_atleast_1d)
.set_attr<FCompute>("FCompute<gpu>", AtleastNDCompute<gpu>);

NNVM_REGISTER_OP(_npi_atleast_2d)
.set_attr<FCompute>("FCompute<gpu>", AtleastNDCompute<gpu>);

NNVM_REGISTER_OP(_npi_atleast_3d)
.set_attr<FCompute>("FCompute<gpu>", AtleastNDCompute<gpu>);

NNVM_REGISTER_OP(_npi_arange)
.set_attr<FCompute>("FCompute<gpu>", RangeCompute<gpu, RangeParam>);

NNVM_REGISTER_OP(_npi_eye)
.set_attr<FCompute>("FCompute<gpu>", NumpyEyeFill<gpu>);

NNVM_REGISTER_OP(_npi_indices)
.set_attr<FCompute>("FCompute<gpu>", IndicesCompute<gpu>);

NNVM_REGISTER_OP(_npi_linspace)
.set_attr<FCompute>("FCompute<gpu>", LinspaceCompute<gpu>);

NNVM_REGISTER_OP(_npi_logspace)
.set_attr<FCompute>("FCompute<gpu>", LogspaceCompute<gpu>);

}  // namespace op
}  // namespace mxnet
