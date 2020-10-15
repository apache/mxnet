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
 * Copyright (c) 2015-2020 by Contributors
 * \file np_broadcast_reduce-inl.cuh
 * \brief GPU implementations for numpy binary broadcast ops
 * \author Zhaoqi Zhu
*/
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_

using namespace mshadow::cuda;
using namespace mshadow;
using namespace broadcast;

#define KERNEL_UNROLL_SWITCH(do_unroll, unrollAmount, unrollVar, ...) \
  if (do_unroll) {                                                    \
    const int unrollVar = unrollAmount;                               \
    {__VA_ARGS__}                                                     \
  } else {                                                            \
    const int unrollVar = 1;                                          \
    {__VA_ARGS__}                                                     \
  }

template<typename Reducer, int NDim, typename DType, typename OType>
void NumpyArgMinMaxReduce(Stream<gpu> *s, const TBlob& in_data, const TBlob& out_data,
                          const Tensor<gpu, 1, char>& workspace) {
 cudaStream_t stream = Stream<gpu>::GetStream(s);
 ReduceImplConfig config(out_data.shape_, in_data.shape_, nullptr, nullptr, sizeof(OType));
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, NDim, OType, DType, OType, mxnet::op::mshadow_op::arg_min_max_map<DType, OType>>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, false, in_data.dptr<DType>(), reinterpret_cast<OType*>(out_data.dptr_), in_data.shape_.get<NDim>(),
      out_data.shape_.get<NDim>());
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel_M1);
  }
  else {
    OType* out_dptr = reinterpret_cast<OType*>(out_data.dptr_);
    bool addto = false;
    if (config.Mnext > 1) {
      // out_dptr[] is N*Mnext*sizeof(DType) bytes
      out_dptr = reinterpret_cast<OType*>(workspace.dptr_);
      addto = false;
      // Check that the workspace is contigiuous
      CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      CHECK_GE(workspace.size(0), config.workspace_size);
    }
    const int by = (config.kernel_1.do_transpose) ?
      config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = ( config.M / (by*config.Mnext) >= unroll_reduce );
    KERNEL_UNROLL_SWITCH(do_unroll, unroll_reduce, UNROLL, {
      reduce_kernel<Reducer, NDim, OType, DType, OType, mxnet::op::mshadow_op::arg_min_max_map<DType, OType>, UNROLL, true>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
        config.N, config.M, addto, in_data.dptr<DType>(), out_dptr, in_data.shape_.get<NDim>(),
        out_data.shape_.get<NDim>(), config.rshape.get<NDim>(), config.rstride.get<NDim>(),
        config.Mnext, config.kernel_1.do_transpose);
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);
    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, OType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, false, config.N, out_dptr, reinterpret_cast<OType*>(out_data.dptr_));
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_lines_kernel);
    }
  }
}

#endif // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_
