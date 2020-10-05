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
#include "../../common/cuda/rtc.h"
#include "../../common/cuda/rtc/vectorization-inl.h"

namespace mxnet {
namespace op {

namespace {

constexpr size_t num_inputs_per_kernel = 4;

struct elementwise_sum_params {
  int num_inputs;
  const void* inputs[num_inputs_per_kernel];
  void* outputs[1];
};

const char elementwise_sum_kernel[] = R"code(
constexpr size_t num_inputs_per_kernel = 4;

struct elementwise_sum_params {
  int num_inputs;
  const void* inputs[num_inputs_per_kernel];
  void* outputs[1];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void elementwise_sum_kernel(
    const elementwise_sum_params params,
    const index_t lead_dim,
    const index_t other_dim,
    const index_t N,
    const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);

  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      tid < M;
      tid += gridDim.x * blockDim.x) {
    typename OType::type temp[nvec];
    if (req == OpReqType::kAddTo) {
      storer.load(tid, N);
#pragma unroll
      for (int i = 0; i < nvec; ++i) {
        temp[i] = OType::from(storer.separate()[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < nvec; ++i) {
        temp[i] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < num_inputs_per_kernel; ++i) {
      if (i < params.num_inputs) {
        VectorizedLoader<InputType0, nvec, aligned> loader(
          reinterpret_cast<const InputType0*>(params.inputs[i]), N);
        loader.load(tid, N);
#pragma unroll
        for (int i = 0; i < nvec; ++i) {
          temp[i] += IType::from(loader.separate()[i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      storer.separate()[i] = OType::to(temp[i]);
    }

    storer.store(tid, N);
  }
}
)code";

void VectorizedElementwiseSum(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  using namespace mxnet::common::cuda::rtc;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  if (req[0] == kNullOp) return;
  CHECK_EQ(outputs.size(), 1U);
  size_t output_type_size = common::mshadow_type_info(outputs[0].type_flag_).size;
  const int nvec = output_type_size <= sizeof(uint2)
                     ? (sizeof(uint2) / output_type_size)
                     : 1;
  const index_t size = inputs[0].Size();
  for (size_t i = 0; i < inputs.size(); i += num_inputs_per_kernel) {
    const std::string code = std::string("const OpReqType req = ") +
                             util::to_string(i == 0 ? req[0] : kAddTo) +
                             ";\n";
    elementwise_sum_params params{};
    params.num_inputs = std::min(num_inputs_per_kernel, inputs.size() - i);
    for (int j = 0; j < params.num_inputs; ++j) {
      params.inputs[j] = inputs[i + j].dptr_;
    }
    params.outputs[0] = outputs[0].dptr_;
    const std::vector<TBlob> new_inputs(inputs.begin() + i,
                                        inputs.begin() + i + params.num_inputs);
    VectorizedKernelRTCLauncher(code, "elementwise_sum_kernel",
                                elementwise_sum_kernel, nvec,
                                size, 1, s, params,
                                new_inputs, outputs,
                                ctx.run_ctx.get_ctx().dev_id);
  }
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
