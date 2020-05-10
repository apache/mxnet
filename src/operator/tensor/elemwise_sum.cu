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
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.cu
 * \brief GPU implementation of elementwise sum operator
*/
#include "./elemwise_sum.h"
#include "../../ndarray/ndarray_function.h"
#include "../../common/cuda_vectorization.cuh"

namespace mxnet {
namespace op {

using common::cuda::VectorizedKernelLauncher;
using common::cuda::VectorizedLoader;
using common::cuda::VectorizedStorer;

namespace {

constexpr size_t num_inputs_per_kernel = 4;

template <typename DType, int NumInputs>
struct VectorizedElementwiseSumKernelParams {
  int num_inputs;
  const DType* inputs[NumInputs];
  DType* outputs[1];
};

template <bool aligned, typename DType, typename LType, int req>
__launch_bounds__(mxnet::common::cuda::vectorized_kernel_thread_num)
__global__ void VectorizedElementwiseSumKernel(
    const VectorizedElementwiseSumKernelParams<DType, num_inputs_per_kernel> params,
    const index_t N) {
  VectorizedStorer<DType, LType, aligned> storer(params.outputs[0], N);

  const index_t M = storer.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      tid < M;
      tid += gridDim.x * blockDim.x) {
    if (req == kAddTo) {
      storer.load(tid, N);
    } else {
#pragma unroll
      for (int i = 0; i < storer.nvec(); ++i) {
        storer.separate()[i] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < num_inputs_per_kernel; ++i) {
      if (i < params.num_inputs) {
        VectorizedLoader<DType, LType, aligned> loader(params.inputs[i], N);
        loader.load(tid, N);
#pragma unroll
        for (int i = 0; i < loader.nvec(); ++i) {
          storer.separate()[i] += loader.separate()[i];
        }
      }
    }

    storer.store(tid, N);
  }
}


template <typename DType, int req>
class VectorizedElementwiseSumFwd {
 public:
  using ParamType = VectorizedElementwiseSumKernelParams<DType, num_inputs_per_kernel>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedElementwiseSumKernel<aligned, DType, LType, req>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

void VectorizedElementwiseSum(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  if (req[0] == kNullOp) return;
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      using LType = uint2;
      const index_t size = inputs[0].Size();
      for (size_t i = 0; i < inputs.size(); i += num_inputs_per_kernel) {
        if (i == 0) {
          using Kernel = VectorizedElementwiseSumFwd<DType, Req>;
          typename Kernel::ParamType params;
          params.num_inputs = std::min(num_inputs_per_kernel, inputs.size() - i);
          for (int j = 0; j < params.num_inputs; ++j) {
            params.inputs[j] = inputs[i + j].dptr<DType>();
          }
          params.outputs[0] = outputs[0].dptr<DType>();
          VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
        } else {
          /* During subsequent launches we need to
             accumulate into the previous outputs
          */
          using Kernel = VectorizedElementwiseSumFwd<DType, kAddTo>;
          typename Kernel::ParamType params;
          params.num_inputs = std::min(num_inputs_per_kernel, inputs.size() - i);
          for (int j = 0; j < params.num_inputs; ++j) {
            params.inputs[j] = inputs[i + j].dptr<DType>();
          }
          params.outputs[0] = outputs[0].dptr<DType>();
          VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
        }
      }
    });
  });
}

void ElementWiseSumComputeExGPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  CHECK_EQ(req[0], kWriteTo) << "ElementWiseSumComputeExGPU only supports req = kWriteTo";
  if (common::ContainsOnlyStorage(inputs, kRowSparseStorage) ||
      (inputs.size() == 3U && inputs[0].storage_type() == kDefaultStorage &&
       inputs[1].storage_type() == kCSRStorage && inputs[2].storage_type() == kDefaultStorage) ||
      (inputs.size() > 4U && common::ContainsStorageType(inputs, kDefaultStorage) &&
       outputs[0].storage_type() == kDefaultStorage)) {
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
    NDArray out_nd = outputs[0];
    mxnet::ndarray::ElementwiseSum<gpu>(s, ctx.requested[0], inputs, &out_nd);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace

NNVM_REGISTER_OP(add_n)
.set_attr<FCompute>("FCompute<gpu>", VectorizedElementwiseSum)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElementWiseSumComputeExGPU);

}  // namespace op
}  // namespace mxnet
