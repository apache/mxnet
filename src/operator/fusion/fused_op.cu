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
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <mutex>
#include "./fused_op.h"
#include "./fused_op-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../executor/exec_pass.h"
#include "../../common/cuda_utils.h"

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

void FusedOp::GenerateCode(const std::vector<OpReqType> &req) {
  const auto& g = this->symbol_.indexed_graph();
  std::string code = "";
  int temp_name_counter = 0;
  using NodeEntry = nnvm::IndexedGraph::NodeEntry;
  std::map<std::pair<int, int>, std::string> variables;

  std::vector<uint32_t> outputs(g.num_nodes());

  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    if (node.source != nullptr) {
      outputs[i] = node.source->num_outputs();
    } else {
      outputs[i] = 0;
    }
  }

  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
      std::string var_name = "temp" + std::to_string(temp_name_counter++);
      if (source->is_variable()) {
        code += "const auto " + var_name + " = load(" + source->attrs.name + ", i);\n";
        CHECK_EQ(outputs[i], 1);
        variables[{i, 0}] = var_name;
      } else {
        std::string op_name = source->op()->name;
        if (detail::fused_op_binary_ops.find(op_name) != detail::fused_op_binary_ops.end()) {
          std::string op = detail::fused_op_binary_ops.at(op_name);
          const auto& arg1 = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          const auto& arg2 = variables[{node.inputs[1].node_id, node.inputs[1].index}];
          code += "const auto " + var_name + " = " + op +
                  "(" + arg1 + ", " + arg2 + ");\n";
          CHECK_EQ(outputs[i], 1);
          variables[{i, 0}] = var_name;
          continue;
        }

        if (detail::fused_op_unary_ops.find(op_name) != detail::fused_op_unary_ops.end()) {
          std::string op = detail::fused_op_unary_ops.at(op_name);
          const auto& arg1 = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          code += "const auto " + var_name + " = " + op +
                  "(" + arg1 + ");\n";
          CHECK_EQ(outputs[i], 1);
          variables[{i, 0}] = var_name;
          continue;
        }

        if (detail::fused_op_special_ops.find(op_name) != detail::fused_op_special_ops.end()) {
          const std::vector<std::string>& op_desc = detail::fused_op_special_ops.at(op_name);
          std::string fmt = op_desc[0];
          for (size_t j = 1; j < op_desc.size(); ++j) {
            const std::string& desc = op_desc[j];
            std::string sub;
            if (desc[0] == '_') {
              // Argument
              int arg_id = std::stoi(desc.substr(1));
              sub = variables[{node.inputs[arg_id].node_id, node.inputs[arg_id].index}];
            } else {
              sub = source->attrs.dict.at(desc);
            }
            size_t pos = fmt.find("%");
            CHECK_NE(pos, std::string::npos);
            fmt.replace(pos, 1, sub);
          }
          code += "const auto " + var_name + " = " + fmt + ";\n";
          CHECK_EQ(outputs[i], 1);
          variables[{i, 0}] = var_name;
          continue;
        }

        if (detail::fused_op_mimo_ops.find(op_name) != detail::fused_op_mimo_ops.end()) {
          const std::vector<std::vector<std::string>>& op_descs =
            detail::fused_op_mimo_ops.at(op_name);
          CHECK_EQ(outputs[i], op_descs.size());
          size_t count = 0;
          for (const auto& op_desc : op_descs) {
            var_name = "temp" + std::to_string(temp_name_counter++);
            std::string fmt = op_desc[0];
            for (size_t j = 1; j < op_desc.size(); ++j) {
              const std::string& desc = op_desc[j];
              std::string sub;
              if (desc[0] == '_') {
                // Argument
                int arg_id = std::stoi(desc.substr(1));
                sub = variables[{node.inputs[arg_id].node_id, node.inputs[arg_id].index}];
              } else {
                sub = source->attrs.dict.at(desc);
              }
              size_t pos = fmt.find("%");
              CHECK_NE(pos, std::string::npos);
              fmt.replace(pos, 1, sub);
            }
            code += "const auto " + var_name + " = " + fmt + ";\n";
            variables[{i, count}] = var_name;
            ++count;
          }
          continue;
        }

        // Special cases with variable number
        // of inputs/outputs, listed in
        // detail::fused_op_variable_io_ops
        if (op_name == "add_n") {
          CHECK_EQ(outputs[i], 1);
          const auto& arg = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          code += "auto " + var_name + " = " + arg + ";\n";
          for (size_t inp = 1; inp < node.inputs.size(); ++inp) {
            const auto& temp_arg = variables[{node.inputs[inp].node_id, node.inputs[inp].index}];
            code += var_name + " = add(" + var_name + ", " + temp_arg + ");\n";
          }
          variables[{i, 0}] = var_name;
          continue;
        }

        if (op_name == "_backward_Activation") {
          CHECK_EQ(outputs[i], 1);
          std::string act_type = node.source->attrs.dict.at("act_type");
          std::string rhs, lhs;
          rhs = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          if (act_type == "relu" ||
              act_type == "sigmoid" ||
              act_type == "tanh") {
            lhs = variables[{node.inputs[1].node_id, node.inputs[1].index}];
          } else {
            lhs = variables[{node.inputs[2].node_id, node.inputs[2].index}];
          }
          code += "const auto " + var_name + " = backward_" + act_type +
                  "(" + lhs + ", " + rhs + ");\n";

          variables[{i, 0}] = var_name;
          continue;
        }
        LOG(FATAL) << "Unrecognized op " + op_name;
      }
    } else {
      LOG(FATAL) << "Encountered node with NULL source.";
    }
  }

  int counter = 0;
  for (const auto& entry : g.outputs()) {
    const std::string& var = variables[{entry.node_id, entry.index}];
    if (req[counter] == kWriteTo || req[counter] == kWriteInplace) {
      code += "store(" + var + ", i, output" + std::to_string(counter) + ");\n";
    } else if (req[counter] == kAddTo) {
      code += "storeadd(" + var + ", i, output" + std::to_string(counter) + ");\n";
    } else if (req[counter] == kNullOp) {
      // NULL req, do not do anything
    } else {
      LOG(FATAL) << "Encountered unexpected req.";
    }
    ++counter;
  }

  this->code_ = code;
}

template <>
void FusedOp::Forward<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  std::lock_guard<std::mutex> lock(my_mutex_);
  CHECK_GE(outputs.size(), 1) << "There needs to be at least 1 output.";

  std::vector<int> in_dtypes;
  std::vector<int> out_dtypes;

  size_t counter = 0;
  for (const auto& blob : inputs) {
    in_dtypes.push_back(blob.type_flag_);
    initialized_ = initialized_ && (blob.type_flag_ == inputs_[counter].dtype);
    inputs_[counter].dtype = blob.type_flag_;
    ++counter;
  }

  counter = 0;
  for (const auto& blob : outputs) {
    out_dtypes.push_back(blob.type_flag_);
    initialized_ = initialized_ && (blob.type_flag_ == outputs_[counter].dtype);
    outputs_[counter].dtype = blob.type_flag_;
    ++counter;
  }

  // Get compute capability of the current GPU
  int dev_id = ctx.run_ctx.ctx.dev_id;
  int cc_major = ComputeCapabilityMajor(dev_id);
  int cc_minor = ComputeCapabilityMinor(dev_id);

  initialized_ = initialized_ && cc_major == this->cc_major_;
  initialized_ = initialized_ && cc_minor == this->cc_minor_;
  this->cc_major_ = cc_major;
  this->cc_minor_ = cc_minor;

  initialized_ = initialized_ && (req == saved_reqs_);
  saved_reqs_ = req;

  if (!initialized_) {
    this->GenerateCode(req);
    LOG(INFO) << code_;
    std::string aux_code = "";
    std::string kernel_params = "";
    nnvm::Symbol sym;
    sym.outputs = this->symbol_.outputs;
    const std::vector<std::string> input_names = sym.ListInputNames(nnvm::Symbol::kAll);
    size_t num_params = in_dtypes.size() + out_dtypes.size();
    size_t i = 0;
    for (const auto &type : in_dtypes) {
      std::string type_name = detail::mshadowTypeToString(type);
      aux_code = "using DType" + std::to_string(i) + " = " + type_name + ";\n" + aux_code;
      kernel_params += "DType" + std::to_string(i) + "* " +input_names[i];
      ++i;
      if (i < num_params) {
        kernel_params += ", ";
      }
    }
    for (const auto &type : out_dtypes) {
      std::string type_name = detail::mshadowTypeToString(type);
      aux_code = "using DType" + std::to_string(i) + " = " + type_name + ";\n" + aux_code;
      kernel_params += "DType" + std::to_string(i) + "* output" +
                       std::to_string(i - in_dtypes.size());
      ++i;
      if (i < num_params) {
        kernel_params += ", ";
      }
    }
    code_ = std::string(detail::fp16_support_string) + "\n" +
            detail::type_support_string + "\n" +
            detail::fused_op_function_definitions + "\n" +
            aux_code + "\n" +
            "__global__ void FusedKernel_" + attrs.name +
            "(size_t N, " + kernel_params + ") {\n" +
            detail::fused_op_kernel_begin + "\n" +
            code_ + "\n" +
            detail::fused_op_kernel_end;
    // Guard NVRTC calls
    std::lock_guard<std::mutex> lock_nvrtc(mutex_);
    nvrtcProgram program;
    NVRTC_CALL(
        nvrtcCreateProgram(&program,                                 // prog
                           &code_[0],                                // buffer
                           (attrs.name + "_kernel.cu").c_str(),      // name
                           0,                                        // numHeaders
                           NULL,                                     // headers
                           NULL));                                   // includeNames
    std::string gpu_arch = "--gpu-architecture=compute_" +
                           std::to_string(this->cc_major_) +
                           std::to_string(this->cc_minor_);

    const char *opts[] = {gpu_arch.c_str(),
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

void FusedOpForwardGPU(const nnvm::NodeAttrs& attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->Forward<gpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_FusedOp)
.set_attr<FCompute>("FCompute<gpu>", FusedOpForwardGPU);

}  // namespace mxnet
