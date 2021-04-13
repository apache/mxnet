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
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op.cc
 * \brief CPU implementation of elementwise binary operators
 */

#include "./elemwise_binary_op.h"

#if MXNET_USE_CUDA
#include "../../common/cuda/rtc/vectorization-inl.h"
#include "../../common/cuda/rtc.h"
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace op {

bool ElemwiseBinaryOp::SparseSparseWithDenseResult(const nnvm::NodeAttrs& attrs,
                                                   const int dev_mask,
                                                   DispatchMode* dispatch_mode,
                                                   std::vector<int> *in_attrs,
                                                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), 1U) << " in operator " << attrs.name;
  const auto& lhs_stype = in_attrs->at(0);
  const auto& rhs_stype = in_attrs->at(1);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex = invalid_ctx ?
                           DispatchMode::kFComputeFallback : DispatchMode::kFComputeEx;
  if (!dispatched && (lhs_stype == kDefaultStorage || rhs_stype == kDefaultStorage)) {
    // dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage) {
    // rsp, rsp -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode, dispatch_ex);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

bool ElemwiseBinaryOp::BackwardUseInStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  using namespace common;
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 2U);
  bool dispatched = false;
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex = invalid_ctx ? DispatchMode::kFComputeFallback :
                           DispatchMode::kFComputeEx;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    if (common::ContainsOnlyStorage(*in_attrs, kRowSparseStorage)
      && common::ContainsOnlyStorage(*out_attrs, kRowSparseStorage)) {
      dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                       dispatch_mode, dispatch_ex);
    }
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

#if MXNET_USE_CUDA

struct binary_kernel_params {
  const void *inputs[3];
  void *outputs[2];
};

const char binary_kernel_fwd[] = R"code(

struct binary_kernel_params {
  const void *inputs[3];
  void *outputs[2];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void binary_kernel(const binary_kernel_params params,
                              const index_t lead_dim,
                              const index_t other_dim,
                              const index_t N,
                              const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> loader0(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedLoader<InputType1, nvec, aligned> loader1(
    reinterpret_cast<const InputType1*>(params.inputs[1]), N);
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);

  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using OType = AccType<OutputType0>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader0.load(tid, N);
    loader1.load(tid, N);
    if (req == OpReqType::kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto input0 = IType0::from(loader0.separate()[i]);
      const auto input1 = IType1::from(loader1.separate()[i]);
      const auto temp = OP(input0, input1);  // enables returning different type

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

void ElemwiseBinaryRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  std::string code = "const OpReqType req = ";
  code += util::to_string(req[0]);
  code += ";\n"
          "#define OP op::";
  code += OP;
  code += "\n";
  const int nvec = outputs[0].type_flag_ == mshadow::kFloat64 ? 2 : 4;

  const index_t size = outputs[0].Size();
  binary_kernel_params params = { {inputs[0].dptr_, inputs[1].dptr_, nullptr},
                                  {outputs[0].dptr_, nullptr} };

  VectorizedKernelRTCLauncher(code, "binary_kernel",
                              binary_kernel_fwd, nvec,
                              size, 1, s, params,
                              inputs, outputs,
                              ctx.run_ctx.get_ctx().dev_id);
}

const char binary_kernel_bwd_use_none[] = R"code(

struct binary_kernel_params {
  const void *inputs[3];
  void *outputs[2];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void binary_kernel_bwd(const binary_kernel_params params,
                                  const index_t lead_dim,
                                  const index_t other_dim,
                                  const index_t N,
                                  const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedStorer<OutputType0, nvec, aligned> lstorer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);
  VectorizedStorer<OutputType1, nvec, aligned> rstorer(
    reinterpret_cast<OutputType1*>(params.outputs[1]), N);

  using IType = AccType<InputType0>;
  using OType0 = AccType<OutputType0>;
  using OType1 = AccType<OutputType1>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (lreq == OpReqType::kAddTo) {
      lstorer.load(tid, N);
    }
    if (rreq == OpReqType::kAddTo) {
      rstorer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto input = IType::from(loader.separate()[i]);
      if (write_left_output) {
        const auto temp = LOP(input);
        if (lreq == OpReqType::kAddTo) {
          // temp2 may have a wider type than either temp
          // or OType
          const auto temp2 = op::add(temp, OType0::from(lstorer.separate()[i]));
          lstorer.separate()[i] = OType0::to(temp2);
        } else {
          lstorer.separate()[i] = OType0::to(temp);
        }
      }
      if (write_right_output) {
        const auto temp = ROP(input);
        if (rreq == OpReqType::kAddTo) {
          // temp2 may have a wider type than either temp
          // or OType
          const auto temp2 = op::add(temp, OType1::from(rstorer.separate()[i]));
          rstorer.separate()[i] = OType1::to(temp2);
        } else {
          rstorer.separate()[i] = OType1::to(temp);
        }
      }
    }
    if (write_left_output) {
      lstorer.store(tid, N);
    }
    if (write_right_output) {
      rstorer.store(tid, N);
    }
  }
}
)code";

void ElemwiseBinaryRTCBwdUseNone::operator()(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);

  bool write_left_output = req[0] != kNullOp &&
                           (req[0] != kWriteInplace ||
                           (req[0] == kWriteInplace && LOP != "identity"));

  bool write_right_output = req[1] != kNullOp &&
                            (req[1] != kWriteInplace ||
                            (req[1] == kWriteInplace && LOP != "identity"));

  const std::string code = std::string("const OpReqType lreq = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "const OpReqType rreq = " +
                           util::to_string(req[1]) +
                           ";\n"
                           "#define ROP op::" +
                           ROP +
                           "\n"
                           "#define LOP op::" +
                           LOP +
                           "\n"
                           "const bool write_left_output = " +
                           std::to_string(write_left_output) +
                           ";\n"
                           "const bool write_right_output = " +
                           std::to_string(write_right_output) +
                           ";\n";
  const int nvec = outputs[0].type_flag_ == mshadow::kFloat64 ? 2 : 4;

  const index_t size = outputs[0].Size();
  binary_kernel_params params = { {inputs[0].dptr_, nullptr, nullptr},
                                  {outputs[0].dptr_, outputs[1].dptr_} };

  VectorizedKernelRTCLauncher(code, "binary_kernel_bwd",
                              binary_kernel_bwd_use_none, nvec,
                              size, 1, s, params,
                              inputs, outputs,
                              ctx.run_ctx.get_ctx().dev_id);
}

const char binary_kernel_bwd_use_in[] = R"code(

struct binary_kernel_params {
  const void *inputs[3];
  void *outputs[2];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void binary_kernel_bwd(const binary_kernel_params params,
                                  const index_t lead_dim,
                                  const index_t other_dim,
                                  const index_t N,
                                  const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> ograd_loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedLoader<InputType1, nvec, aligned> linput_loader(
    reinterpret_cast<const InputType1*>(params.inputs[1]), N);
  VectorizedLoader<InputType2, nvec, aligned> rinput_loader(
    reinterpret_cast<const InputType2*>(params.inputs[2]), N);

  VectorizedStorer<OutputType0, nvec, aligned> lstorer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);
  VectorizedStorer<OutputType1, nvec, aligned> rstorer(
    reinterpret_cast<OutputType1*>(params.outputs[1]), N);

  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using IType2 = AccType<InputType2>;
  using OType0 = AccType<OutputType0>;
  using OType1 = AccType<OutputType1>;


  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    ograd_loader.load(tid, N);
    linput_loader.load(tid, N);
    rinput_loader.load(tid, N);
    if (lreq == OpReqType::kAddTo) {
      lstorer.load(tid, N);
    }
    if (rreq == OpReqType::kAddTo) {
      rstorer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto ograd = IType0::from(ograd_loader.separate()[i]);
      const auto linput = IType1::from(linput_loader.separate()[i]);
      const auto rinput = IType2::from(rinput_loader.separate()[i]);

      if (lreq != OpReqType::kNullOp) {
        const auto temp = op::mul(ograd, LOP(linput, rinput));
        if (lreq == OpReqType::kAddTo) {
          const auto temp2 = op::add(temp, OType0::from(lstorer.separate()[i]));
          lstorer.separate()[i] = OType0::to(temp2);
        } else {
          lstorer.separate()[i] = OType0::to(temp);
        }
      }

      if (rreq != OpReqType::kNullOp) {
        const auto temp = op::mul(ograd, ROP(linput, rinput));
        if (rreq == OpReqType::kAddTo) {
          const auto temp2 = op::add(temp, OType1::from(rstorer.separate()[i]));
          rstorer.separate()[i] = OType1::to(temp2);
        } else {
          rstorer.separate()[i] = OType1::to(temp);
        }
      }
    }
    if (lreq != OpReqType::kNullOp) {
      lstorer.store(tid, N);
    }
    if (rreq != OpReqType::kNullOp) {
      rstorer.store(tid, N);
    }
  }
}
)code";

void ElemwiseBinaryRTCBwdUseIn::operator()(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const std::string code = std::string("const OpReqType lreq = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "const OpReqType rreq = " +
                           util::to_string(req[1]) +
                           ";\n"
                           "#define ROP op::" +
                           ROP +
                           "\n"
                           "#define LOP op::" +
                           LOP +
                           "\n";
  // Using 64 bit loads to reduce register pressure
  size_t output_type_size = common::mshadow_type_info(outputs[0].type_flag_).size;
  const int nvec = output_type_size <= sizeof(uint64_t)
                     ? (sizeof(uint64_t) / output_type_size)
                     : 1;

  const index_t size = outputs[0].Size();
  binary_kernel_params params = { {inputs[0].dptr_, inputs[1].dptr_, inputs[2].dptr_},
                                  {outputs[0].dptr_, outputs[1].dptr_} };

  VectorizedKernelRTCLauncher(code, "binary_kernel_bwd",
                              binary_kernel_bwd_use_in, nvec,
                              size, 1, s, params,
                              inputs, outputs,
                              ctx.run_ctx.get_ctx().dev_id);
}


#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
