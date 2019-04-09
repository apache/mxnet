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

#include <nvrtc.h>
#include <cuda.h>
#include "./fused_op.h"
#include "./fused_op-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../executor/exec_pass.h"
#include <nnvm/pass_functions.h>

namespace mxnet {

namespace detail {

inline std::string mshadowTypeToString(int type) {
  switch (type) {
    case mshadow::kFloat32:
      return "float";
    case mshadow::kFloat64:
      return "double";
    case mshadow::kFloat16:
      return "half";
    case mshadow::kUint8:
      return "unsigned char";
    case mshadow::kInt8:
      return "char";
    case mshadow::kInt32:
      return "int";
    case mshadow::kInt64:
      return "long long";
    default:
      LOG(FATAL) << "Unknown type enum " << type;
  }
  return "";
}

}  // namespace detail

FusedOp::FusedOp(const FusedOpConfig& config) {
  this->code_ = config.code;
  this->inputs_ = std::vector<FusedOpEntry>(config.num_inputs);
  this->outputs_ = std::vector<FusedOpEntry>(config.num_outputs);
  this->symbol_ = nnvm::pass::LoadJSON(config.symbol_json);
  this->initialized_ = false;
}

template <>
void FusedOp::Forward<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  CHECK_GE(outputs.size(), 1) << "There needs to be at least 1 output.";

  if (!initialized_) {
    int device;
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUDA_CALL(cudaGetDevice(&device))
    CUDA_DRIVER_CALL(cuDeviceGet(&cuDevice, device));
    CUDA_DRIVER_CALL(cuDevicePrimaryCtxRetain(&context, cuDevice));
    CUDA_DRIVER_CALL(cuModuleLoadData(&module, &ptx_[0]));
    CUDA_DRIVER_CALL(cuModuleGetFunction(&kernel_,
                                         module,
                                         kernel_name_.c_str()));
    initialized_ = true;
  }
  Stream<gpu>* s = ctx.get_stream<gpu>();
  auto stream = Stream<gpu>::GetStream(s);
  std::vector<void*> args;
  size_t N = outputs[0].shape_.Size();
  args.push_back(&N);
  unsigned int num_blocks = (N + FusedOp::NTHREADS - 1) / FusedOp::NTHREADS;
  std::vector<void*> ptrs;
  for (const auto &data : inputs) {
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      Tensor<gpu, 1, DType> tensor = data.FlatTo1D<gpu, DType>(s);
      ptrs.push_back(tensor.dptr_);
    });
  }
  for (const auto &data : outputs) {
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      Tensor<gpu, 1, DType> tensor = data.FlatTo1D<gpu, DType>(s);
      ptrs.push_back(tensor.dptr_);
    });
  }
  for (auto &ptr : ptrs) {
    args.push_back(reinterpret_cast<void *>(&ptr));
  }
  CUDA_DRIVER_CALL(
      cuLaunchKernel(kernel_,
        num_blocks, 1, 1,          // grid dim
        FusedOp::NTHREADS, 1, 1,   // block dim
        0, stream,                 // shared mem and stream
        &(args[0]), 0));           // arguments
}

template <>
void FusedOp::Backward<gpu>(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  std::cout << "Backward!" << std::endl;
}

template <>
bool FusedOp::InferShape<gpu>(const nnvm::NodeAttrs &attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  std::vector<TShape> input_shapes(*in_attrs);
  this->symbol_ = mxnet::exec::InferShape(std::move(this->symbol_),
                                          std::move(input_shapes),
                                          "__shape__");

  const auto& g = this->symbol_.indexed_graph();

  std::vector<TShape> out_shapes;
  const std::vector<TShape> shapes = this->symbol_.GetAttr<nnvm::ShapeVector>("shape");
  for (auto& e : g.outputs()) {
    out_shapes.push_back(shapes[g.entry_id(e)]);
  }
  CHECK_EQ(out_shapes.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::shape_assign(&(out_attrs->at(i)), out_shapes[i]);
  }
  bool inferred = true;
  for (const auto& attr : *in_attrs) {
    inferred = inferred && !op::shape_is_none(attr);
  }
  for (const auto& attr : *out_attrs) {
    inferred = inferred && !op::shape_is_none(attr);
  }
  return inferred;
}

template <>
bool FusedOp::InferType<gpu>(const nnvm::NodeAttrs &attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  std::vector<int> input_types(*in_attrs);
  this->symbol_ = mxnet::exec::InferType(std::move(this->symbol_),
                                         std::move(input_types),
                                         "__dtype__");

  const auto& g = this->symbol_.indexed_graph();

  std::vector<int> out_types;
  const std::vector<int> types = this->symbol_.GetAttr<nnvm::DTypeVector>("dtype");
  for (auto& e : g.outputs()) {
    out_types.push_back(types[g.entry_id(e)]);
  }
  CHECK_EQ(out_types.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::type_assign(&(out_attrs->at(i)), out_types[i]);
  }
  bool inferred = true;
  for (const auto& attr : *in_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  for (const auto& attr : *out_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  const bool types_known = inferred;
  if (types_known) {
    LOG(INFO) << "Without types";
    LOG(INFO) << code_;
    LOG(INFO) << "Filling type information";
    std::string aux_code = "";
    std::string kernel_params = "";
    size_t num_params = in_attrs->size() + out_attrs->size();
    size_t i = 0;
    for (const auto &type : *in_attrs) {
      std::string type_name = detail::mshadowTypeToString(type);
      aux_code = "using DType" + std::to_string(i) + " = " + type_name + ";\n" + aux_code;
      kernel_params += "DType" + std::to_string(i) + "* input" + std::to_string(i);
      ++i;
      if (i < num_params) {
        kernel_params += ", ";
      }
    }
    for (const auto &type : *out_attrs) {
      std::string type_name = detail::mshadowTypeToString(type);
      aux_code = "using DType" + std::to_string(i) + " = " + type_name + ";\n" + aux_code;
      kernel_params += "DType" + std::to_string(i) + "* output" +
                       std::to_string(i - in_attrs->size());
      ++i;
      if (i < num_params) {
        kernel_params += ", ";
      }
    }
    code_ = detail::fp16_support_string + "\n" +
            detail::fused_op_function_definitions + "\n" +
            aux_code + "\n" +
            "__global__ void FusedKernel_" + attrs.name +
            "(size_t N, " + kernel_params + ") {\n" +
            detail::fused_op_kernel_begin + "\n" +
            code_ + "\n" +
            detail::fused_op_kernel_end;
    LOG(INFO) << code_;
    nvrtcProgram program;
    NVRTC_CALL(
        nvrtcCreateProgram(&program,                                 // prog
                           &code_[0],                                // buffer
                           (attrs.name + "_kernel.cu").c_str(),      // name
                           0,                                        // numHeaders
                           NULL,                                     // headers
                           NULL));                                   // includeNames
    const char *opts[] = {"--gpu-architecture=compute_70",
                          "--std=c++11",
                          "-default-device"};
    const std::string kernel_name_demangled = "FusedKernel_" + attrs.name;
    NVRTC_CALL(nvrtcAddNameExpression(program, (kernel_name_demangled).c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(program,  // prog
                                                    3,        // numOptions
                                                    opts);    // options
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_CALL(nvrtcGetProgramLogSize(program, &logSize));
    std::string log(logSize, '\0');
    NVRTC_CALL(nvrtcGetProgramLog(program, &log[0]));
    CHECK_EQ(compileResult, NVRTC_SUCCESS) << "NVRTC Compilation failed.\n" << log;
    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_CALL(nvrtcGetPTXSize(program, &ptxSize));
    ptx_.reserve(ptxSize);
    NVRTC_CALL(nvrtcGetPTX(program, &ptx_[0]));
    const char *name;
    NVRTC_CALL(nvrtcGetLoweredName(program,
                                   kernel_name_demangled.c_str(),
                                   &name));
    kernel_name_ = name;
    // Destroy the program.
    NVRTC_CALL(nvrtcDestroyProgram(&program));
  }
  return types_known;
}



void FusedOpForwardGPU(const nnvm::NodeAttrs& attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->Forward<gpu>(attrs, ctx, inputs, req, outputs);
}
void FusedOpBackwardGPU(const nnvm::NodeAttrs& attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->Backward<gpu>(attrs, ctx, inputs, req, outputs);
}

bool FusedOpInferShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferShape<gpu>(attrs, in_attrs, out_attrs);
}

bool FusedOpInferType(const nnvm::NodeAttrs& attrs,
                      std::vector<int> *in_attrs,
                      std::vector<int> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferType<gpu>(attrs, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(FusedOp)
.set_attr<nnvm::FInferShape>("FInferShape", FusedOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", FusedOpInferType)
.set_attr<FCompute>("FCompute<gpu>", FusedOpForwardGPU);

}  // namespace mxnet
