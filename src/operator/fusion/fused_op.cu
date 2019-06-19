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

#include <sys/stat.h>
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

std::string FindCUDAIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    return cuda_include_path;
  }

#if defined(__linux__)
  struct stat st;
  cuda_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return cuda_include_path;
  }
#endif
  LOG(FATAL) << "Cannot find cuda include path."
             << "CUDA_PATH is not set or CUDA is not installed in the default installation path."
             << "In other than linux, it is necessary to set CUDA_PATH.";
  return cuda_include_path;
}


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

inline int mshadowTypeToVectorLength(int type) {
  switch (type) {
    case mshadow::kFloat32:
      return 1;
    case mshadow::kFloat64:
      return 1;
    case mshadow::kFloat16:
      return 2;
    case mshadow::kUint8:
      return 4;
    case mshadow::kInt8:
      return 4;
    case mshadow::kInt32:
      return 1;
    case mshadow::kInt64:
      return 1;
    default:
      LOG(FATAL) << "Unknown type enum " << type;
  }
  return 0;
}


}  // namespace detail

void FusedOp::GenerateCode(const std::vector<OpReqType> &req) {
  const auto& g = this->symbol_.indexed_graph();
  std::string code = "";
  int temp_name_counter = 0;
  using NodeEntry = nnvm::IndexedGraph::NodeEntry;
  std::map<std::pair<int, int>, std::string> variables;
  std::map<int, int> load_index;

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
        if (source->is_variable()) {
            load_index[i] = 1;
        } else {
            std::string op_name = source->op()->name;
            if (detail::fused_op_slice_ops.find(op_name) != detail::fused_op_slice_ops.end()) {
                load_index[node.inputs[0].node_id] = 0;
            }
        }
    }
  }
  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
        if (source->is_variable()) {
            if (load_index[i]) {
              const auto& var_name = source->attrs.name;
              code += "const auto vec_" + var_name + " = load_index<nvec>(" + \
                       var_name + ", offset);\n";
              variables[{i, 0}] = var_name;
            }
            CHECK_EQ(outputs[i], 1);
        } else {
            std::string op_name = source->op()->name;
            if (detail::fused_op_slice_ops.find(op_name) != detail::fused_op_slice_ops.end()) {
                int arg_id = node.inputs[0].node_id;
                const auto& var_name = g[arg_id].source->attrs.name;
                load_index[arg_id] = 0;
                std::string begin = source->attrs.dict.at("begin");
                std::string end = source->attrs.dict.at("end");
                if (end == "None") {
                    end = "((1<<31)-1)";
                }
                std::string axis = source->attrs.dict.at("axis");
                const auto vec_name = "vec_" + var_name + "_" + std::to_string(i);
                code += "const auto " + vec_name + " = load_slice<nvec, "+ axis + ">(" + \
                        var_name + ", " + var_name + "_strides," + begin + \
                        "," + end + ", offset);\n";
                CHECK_EQ(outputs[i], 1);
                variables[{i, 0}] = vec_name;
                continue;
            }
        }
    }
  }

  int counter = 0;
  for (const auto& entry : g.outputs()) {
    const auto var_name = "output" + std::to_string(counter);
    code += "VectorType<remove_pointer<decltype(" + var_name + \
            ")>::type, nvec> vec_output" + std::to_string(counter) + ";\n";
    ++counter;
  }

  code += "for (int j = 0; j < nvec; j++ ) {\n";


  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
      std::string var_name = "temp" + std::to_string(temp_name_counter++);
      if (source->is_variable()) {
        if (load_index[i]) {
            code += "const auto " + var_name + " = load(vec_" + variables[{i, 0}] + ".x[j]);\n";
            CHECK_EQ(outputs[i], 1);
            variables[{i, 0}] = var_name;
        }
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

        if (detail::fused_op_slice_ops.find(op_name) != detail::fused_op_slice_ops.end()) {
          code += "const auto " + var_name + " = load(" + variables[{i, 0}] + ".x[j]);\n";
          variables[{i, 0}] = var_name;
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

  counter = 0;
  for (const auto& entry : g.outputs()) {
    const std::string& var = variables[{entry.node_id, entry.index}];
    const auto var_name = "output" + std::to_string(counter);
    code += "vec_" + var_name + ".x[j] = store("+ var +", " + var_name + ");\n";
    ++counter;
  }

  code += "}\n";

  counter = 0;

  for (const auto& entry : g.outputs()) {
    const std::string& var = variables[{entry.node_id, entry.index}];
    if (req[counter] == kWriteTo || req[counter] == kWriteInplace) {
      const auto var_name = "output" + std::to_string(counter);
      code += "store_index(vec_" + var_name + ", i, " + var_name + ");\n";
    } else if (req[counter] == kAddTo) {
      const auto var_name = "output" + std::to_string(counter);
      code += "store_add_index(vec_" + var_name + ", i, " + var_name + ");\n";
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
  std::vector<int> in_ndims;
  std::vector<int> out_dtypes;
  int ndim = outputs[0].ndim();
  int nvec = 1;

  size_t counter = 0;
  for (const auto& blob : inputs) {
    in_dtypes.push_back(blob.type_flag_);
    in_ndims.push_back(blob.ndim());
    initialized_ = initialized_ && (blob.type_flag_ == inputs_[counter].dtype);
    inputs_[counter].dtype = blob.type_flag_;
    nvec = max(nvec, detail::mshadowTypeToVectorLength(blob.type_flag_));
    ++counter;
  }

  counter = 0;
  for (const auto& blob : outputs) {
    out_dtypes.push_back(blob.type_flag_);
    initialized_ = initialized_ && (blob.type_flag_ == outputs_[counter].dtype);
    outputs_[counter].dtype = blob.type_flag_;
    nvec = max(nvec, detail::mshadowTypeToVectorLength(blob.type_flag_));
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
    std::string tensor_params = "";
    nnvm::Symbol sym;
    sym.outputs = this->symbol_.outputs;
    const std::vector<std::string> input_names = sym.ListInputNames(nnvm::Symbol::kAll);
    size_t num_params = in_dtypes.size() + out_dtypes.size();
    size_t i = 0;
    aux_code += "static const int nvec = " + std::to_string(nvec) + ";\n";
    for (const auto &type : in_dtypes) {
      std::string type_name = detail::mshadowTypeToString(type);
      std::string dtype_var = "DType" + std::to_string(i);
      std::string dim_var = "ndim" + std::to_string(i);
      aux_code = "using " + dtype_var + " = " + type_name + ";\n" + aux_code;
      aux_code = "static const int " + dim_var + " = " + \
                  std::to_string(in_ndims[i]) + ";\n" + aux_code;
      tensor_params += dtype_var + "* " +input_names[i];
      kernel_params += " const Strides<" + dim_var + "> " + input_names[i]+"_strides";
      ++i;
      if (i < num_params) {
        tensor_params += ", ";
        kernel_params += ", ";
      }
    }
    for (const auto &type : out_dtypes) {
      std::string type_name = detail::mshadowTypeToString(type);
      std::string dtype_var = "DType" + std::to_string(i);
      aux_code = "using " + dtype_var + " = " + type_name + ";\n" + aux_code;
      tensor_params += dtype_var + "* output" +
                       std::to_string(i - in_dtypes.size());
      ++i;
      if (i < num_params) {
        tensor_params += ", ";
      }
    }
    kernel_params += tensor_params;
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
    std::string cuda_include_path = "-I" + detail::FindCUDAIncludePath();

    const char *opts[] = {gpu_arch.c_str(),
                          "--std=c++11",
                          "-default-device",
                          cuda_include_path.c_str()};
    const std::string kernel_name_demangled = "FusedKernel_" + attrs.name;
    NVRTC_CALL(nvrtcAddNameExpression(program, (kernel_name_demangled).c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(program,  // prog
                                                    4,        // numOptions
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
    CUDA_CALL(cudaGetDevice(&device));
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
  size_t N = (outputs[0].shape_.Size() + nvec - 1)/nvec;
  args.push_back(&N);

  unsigned int num_blocks = (N + FusedOp::NTHREADS - 1) / FusedOp::NTHREADS;
  std::vector<void*> ptrs;
  std::vector<std::vector<int>> strides;
  for (const auto &data : inputs) {
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      int ndim = data.ndim();
      Tensor<gpu, 1, DType> tensor = data.FlatTo1D<gpu, DType>(s);
      ptrs.push_back(tensor.dptr_);
      strides.push_back(std::vector<int>(ndim));
      std::vector<int>& tensor_strides = strides.back();
      tensor_strides[ndim-1] = 1;
      for (int i = ndim-2; i >= 0; i--) {
        tensor_strides[i] = tensor_strides[i+1] * data.shape_[i+1];
      }
    });
  }
  for (const auto &data : outputs) {
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      Tensor<gpu, 1, DType> tensor = data.FlatTo1D<gpu, DType>(s);
      ptrs.push_back(tensor.dptr_);
    });
  }
  for (auto &tensor_strides : strides) {
    args.push_back(tensor_strides.data());
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
