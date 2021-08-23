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

#include <string>
#include "elemwise_unary_op.h"
#include "elemwise_binary_op.h"

#if MXNET_USE_CUDA
#include "../../common/cuda/rtc/vectorization-inl.h"
#include "../../common/cuda/rtc.h"
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace op {

#if MXNET_USE_CUDA

struct unary_kernel_params {
  const void *inputs[1];
  void *outputs[1];
};

const char unary_kernel_fwd[] = R"code(

struct unary_kernel_params {
  const void *inputs[1];
  void *outputs[1];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void unary_kernel(const unary_kernel_params params,
                             const index_t lead_dim,
                             const index_t other_dim,
                             const index_t N,
                             const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);

  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (req == OpReqType::kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto input = IType::from(loader.separate()[i]);
      const auto temp = OP(input);  // enables returning different type

      if (req == OpReqType::kAddTo) {
        // temp2 may have a wider type than either temp
        // or OType
        const auto temp2 = op::add(temp, OType::from(storer.separate()[i]));
        storer.separate()[i] = OType::to(temp2);
      } else {
        storer.separate()[i] = OType::to(temp);
      }
    }
    storer.store(tid, N);
  }
}

)code";

void UnaryRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const std::string code = std::string("const OpReqType req = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "#define OP op::" +
                           OP +
                           "\n";
  const int nvec = outputs[0].type_flag_ == mshadow::kFloat64 ? 2 : 4;

  const index_t size = outputs[0].Size();
  unary_kernel_params params = { {inputs[0].dptr_},
                                 {outputs[0].dptr_} };

  VectorizedKernelRTCLauncher(code, "unary_kernel",
                              unary_kernel_fwd, nvec,
                              size, 1, s, params,
                              inputs, outputs,
                              ctx.run_ctx.get_ctx().dev_id);
}

void UnaryRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  InitStorageGeometry<1, 1>(attrs, inputs, outputs);
  CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
    << "This function works only for sparse types.";
  CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type())
    << "The storage type of both inputs and outputs needs to be the same.";
  AllocateGeometry(&outputs[0], req[0], &inputs[0]);
  CopyGeometryBlobs<gpu>(ctx.get_stream<gpu>(), &outputs[0], req[0], inputs[0]);
  outputs[0].CheckAndAllocData(inputs[0].storage_shape());
  if (inputs[0].storage_shape().Size()) {
    std::vector<TBlob> in_blobs, out_blobs;
    in_blobs.reserve(inputs.size());
    out_blobs.reserve(outputs.size());
    for (auto &input : inputs) {
      in_blobs.emplace_back(input.data());
    }
    for (auto &output : outputs) {
      out_blobs.emplace_back(output.data());
    }
    this->operator()(attrs, ctx, in_blobs, req, out_blobs);
  }
}

void UnaryBwdInOutRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                         const OpContext& ctx,
                                         const std::vector<TBlob>& inputs,
                                         const std::vector<OpReqType>& req,
                                         const std::vector<TBlob>& outputs) {
  ElemwiseBinaryRTCCompute {OP} (attrs, ctx, {inputs[0], inputs[2]}, req, outputs);
}

#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
