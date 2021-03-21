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
 *  Copyright (c) 2021 by Contributors
 * \file adabelief.cu
 * \brief Optimizer operators
 * \author khaotik
 */
#include "./adabelief-inl.h"

namespace mxnet {
namespace op {
namespace adabelief {
template<>
void GetScaleFloat<gpu>(mshadow::Stream<gpu> *s, const TBlob &scale_blob, float *pScalef) {
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType, {
    DType scale = 0;
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    CUDA_CALL(cudaMemcpyAsync(&scale, scale_blob.dptr<DType>(), sizeof(DType),
                              cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    *pScalef = static_cast<float>(scale);
  })
}
}  // namespace adabelief

NNVM_REGISTER_OP(_adabelief_update)
.set_attr<FCompute>("FCompute<gpu>", adabelief::MPUpdate<gpu, adabelief::AdaBeliefUpdate<gpu>>);

NNVM_REGISTER_OP(_mp_adabelief_update)
.set_attr<FCompute>("FCompute<gpu>", adabelief::MPUpdate<gpu, adabelief::MPAdaBeliefUpdate<gpu>>);

NNVM_REGISTER_OP(_multi_adabelief_update)
.set_attr<FCompute>("FCompute<gpu>", adabelief::multiMPUpdate<gpu, false>);

NNVM_REGISTER_OP(_multi_mp_adabelief_update)
.set_attr<FCompute>("FCompute<gpu>", adabelief::multiMPUpdate<gpu, true>);

}  // namespace op
}  // namespace mxnet
