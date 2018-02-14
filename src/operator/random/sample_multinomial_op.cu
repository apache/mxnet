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
 * \file sample_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#include "./sample_multinomial_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_sample_multinomial)
.set_attr<FCompute>("FCompute<gpu>", SampleMultinomialForward<gpu>);


struct SampleMultinomialBackwardGPUKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t K, index_t M,
                                  DType* ograd, DType* dist, IType* out,
                                  DType* igrad) {
    for (index_t j = 0; j < M; ++j) {
      atomicAdd(&igrad[i*K + out[i*M + j]], ograd[i*M + j] / dist[i*K + out[i*M + j]]);
    }
  }
};


NNVM_REGISTER_OP(_backward_sample_multinomial)
.set_attr<FCompute>("FCompute<gpu>",
  SampleMultinomialBackward<SampleMultinomialBackwardGPUKernel, gpu>);


}  // namespace op
}  // namespace mxnet
