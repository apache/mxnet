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
 * \file reset_arrays.cu
 * \brief setting all array element values to zeros
 * \author Moises Hernandez-Fernandez, Andrei Ivanov
 */
#include "./reset_arrays-inl.h"

namespace mxnet {
namespace op {

template<>
void ResetMemory<gpu>(void *pntr, size_t len, mshadow::Stream<gpu> *s) {
  CUDA_CALL(cudaMemsetAsync(pntr, 0, len, mshadow::Stream<gpu>::GetStream(s)));
}

NNVM_REGISTER_OP(reset_arrays)
.set_attr<FCompute>("FCompute<gpu>", ResetArrays<gpu>);

}  // namespace op
}  // namespace mxnet
