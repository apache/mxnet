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
 *  Copyright (c) 2020 by Contributors
 * \file elemwise_unary_op.cuh
 * \brief GPU helpers for unary elementwise operators
 */

#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_CUH_

#include <cuda_runtime.h>
#include "../operator_common.h"
#include "../../common/cuda_vectorization.cuh"

#include <vector>

#if MXNET_USE_CUDA

namespace mxnet {
namespace op {

namespace unary {

using common::cuda::VectorizedKernelLauncher;
using common::cuda::VectorizedLoader;
using common::cuda::VectorizedStorer;

template <typename DType, int NumInputs, int NumOutputs>
struct VectorizedKernelParams {
  const DType* inputs[NumInputs];
  DType* outputs[NumOutputs];
};

template <bool aligned, typename DType, typename LType, typename OP, int req>
__global__ void VectorizedUnaryScalarKernelFwd(const VectorizedKernelParams<DType, 1, 1> params,
                                               const index_t N) {
  VectorizedLoader<DType, LType, aligned> loader(params.inputs[0], N);
  VectorizedStorer<DType, LType, aligned> storer(params.outputs[0], N);

  const index_t M = loader.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (req == kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < loader.nvec(); ++i) {
      DType temp = OP::Map(loader.separate()[i]);

      if (req == kAddTo) {
        storer.separate()[i] += temp;
      } else {
        storer.separate()[i] = temp;
      }
    }
    storer.store(tid, N);
  }
}

template <typename DType, typename OP, int req>
class VectorizedUnaryScalarFwd {
 public:
  using ParamType = VectorizedKernelParams<DType, 1, 1>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedUnaryScalarKernelFwd<aligned, DType, LType, OP, req>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

}  // namespace unary

template<typename OP>
void UnaryOp::Compute_(const nnvm::NodeAttrs& attrs,
                     mshadow::Stream<gpu>* s,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace unary;
  if (req[0] == kNullOp) return;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      using LType = uint4;
      using Kernel = VectorizedUnaryScalarFwd<DType, OP, Req>;

      const index_t size = outputs[0].Size();
      typename Kernel::ParamType params;
      params.inputs[0] = inputs[0].dptr<DType>();
      params.outputs[0] = outputs[0].dptr<DType>();

      VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_CUH_
