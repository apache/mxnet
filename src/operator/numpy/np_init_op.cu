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

#include "./np_init_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_zeros)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(_npi_ones)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

NNVM_REGISTER_OP(_np_zeros_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(_np_ones_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

NNVM_REGISTER_OP(_npi_arange)
.set_attr<FCompute>("FCompute<gpu>", RangeCompute<gpu>);

NNVM_REGISTER_OP(_npi_eye)
.set_attr<FCompute>("FCompute<gpu>", NumpyEyeFill<gpu>);

}  // namespace op
}  // namespace mxnet
