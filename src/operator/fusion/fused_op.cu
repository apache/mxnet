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

FusedOp::FusedOp(const nnvm::NodeAttrs* attrs, const FusedOpConfig& config) {
  this->inputs_ = std::vector<FusedOpEntry>(config.num_inputs);
  this->outputs_ = std::vector<FusedOpEntry>(config.num_outputs);
  this->symbol_ = nnvm::Graph();
  this->symbol_.outputs = attrs->subgraphs[0]->outputs;
  this->initialized_ = false;
  this->cc_major_ = -1;
  this->cc_minor_ = -1;
}

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

template <>
bool FusedOp::InferShape<gpu>(const nnvm::NodeAttrs &attrs,
                              std::vector<mxnet::TShape> *in_attrs,
                              std::vector<mxnet::TShape> *out_attrs) {
  this->symbol_.attrs.erase("shape");
  this->symbol_.attrs.erase("shape_inputs");
  std::vector<mxnet::TShape> input_shapes(*in_attrs);
  this->symbol_ = mxnet::exec::InferShape(std::move(this->symbol_),
                                          std::move(input_shapes),
                                          "__shape__");

  const auto& g = this->symbol_.indexed_graph();
  const auto& input_nids = g.input_nodes();

  std::vector<mxnet::TShape> out_shapes;
  const std::vector<mxnet::TShape> shapes = this->symbol_.GetAttr<mxnet::ShapeVector>("shape");
  for (auto& e : g.outputs()) {
    out_shapes.push_back(shapes[g.entry_id(e)]);
  }
  CHECK_EQ(out_shapes.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::shape_assign(&(out_attrs->at(i)), out_shapes[i]);
  }

  // assign to in_attrs
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    const auto eid = g.entry_id(input_nids[i], 0);
    SHAPE_ASSIGN_CHECK(*in_attrs, i, shapes[eid]);
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
  this->symbol_.attrs.erase("dtype");
  this->symbol_.attrs.erase("dtype_inputs");
  std::vector<int> input_types(*in_attrs);
  this->symbol_ = mxnet::exec::InferType(std::move(this->symbol_),
                                         std::move(input_types),
                                         "__dtype__");

  const auto& g = this->symbol_.indexed_graph();
  const auto& input_nids = g.input_nodes();

  std::vector<int> out_types;
  const std::vector<int> types = this->symbol_.GetAttr<nnvm::DTypeVector>("dtype");
  for (auto& e : g.outputs()) {
    out_types.push_back(types[g.entry_id(e)]);
  }
  CHECK_EQ(out_types.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::type_assign(&(out_attrs->at(i)), out_types[i]);
  }

  // assign to in_attrs
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    const auto eid = g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_attrs, i, types[eid]);
  }

  bool inferred = true;
  for (const auto& attr : *in_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  for (const auto& attr : *out_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  return inferred;
}

template <typename Attr>
std::pair<std::vector<Attr>, std::vector<Attr>> FusedOp::GetAttrs(const std::string& attr_name,
                                                                  const uint32_t node_id) {
  const auto& g = this->symbol_.indexed_graph();
  const std::vector<Attr> attrs = this->symbol_.GetAttr<std::vector<Attr>>(attr_name);
  const auto& node = g[node_id];
  std::vector<Attr> inputs, outputs;
  for (const auto& e : node.inputs) {
    inputs.emplace_back(attrs[g.entry_id(e)]);
  }
  outputs.resize(node.source->num_outputs());
  for (size_t i = 0; i < g.num_nodes(); ++i) {
    if (i == node_id) continue;
    const auto& other_node = g[i];
    for (const auto& e : other_node.inputs) {
      if (e.node_id == node_id) {
        outputs[e.index] = attrs[g.entry_id(e)];
      }
    }
  }
  for (const auto& e : g.outputs()) {
    if (e.node_id == node_id) {
      outputs[e.index] = attrs[g.entry_id(e)];
    }
  }

  return {inputs, outputs};
}

void FusedOpForwardGPU(const nnvm::NodeAttrs& attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->Forward<gpu>(attrs, ctx, inputs, req, outputs);
}

bool FusedOpInferShape(const nnvm::NodeAttrs& attrs,
                       std::vector<mxnet::TShape> *in_attrs,
                       std::vector<mxnet::TShape> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferShape<gpu>(attrs, in_attrs, out_attrs);
}

bool FusedOpInferType(const nnvm::NodeAttrs& attrs,
                      std::vector<int> *in_attrs,
                      std::vector<int> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferType<gpu>(attrs, in_attrs, out_attrs);
}

void FusedOpProvideShape(const nnvm::NodeAttrs& attrs,
                         const std::vector<std::vector<mxnet::TShape>> &in_attrs,
                         const std::vector<std::vector<mxnet::TShape>> &out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->ProvideShape(in_attrs, out_attrs);
}

void FusedOpProvideType(const nnvm::NodeAttrs& attrs,
                        const std::vector<std::vector<int>> &in_attrs,
                        const std::vector<std::vector<int>> &out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->ProvideType(in_attrs, out_attrs);
}

void FusedOpProvideStorageType(const nnvm::NodeAttrs& attrs,
                               const std::vector<std::vector<int>> &in_attrs,
                               const std::vector<std::vector<int>> &out_attrs) {}


NNVM_REGISTER_OP(FusedOp)
.set_attr<exec::TIsFusion>("TIsFusion", true)
.set_attr<exec::FProvideSubgraphShape>("FProvideSubgraphShape", FusedOpProvideShape)
.set_attr<exec::FProvideSubgraphType>("FProvideSubgraphType", FusedOpProvideType)
.set_attr<exec::FProvideSubgraphStorageType>("FProvideSubgraphStorageType",
                                             FusedOpProvideStorageType)
.set_attr<mxnet::FInferShape>("FInferShape", FusedOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", FusedOpInferType)
.set_attr<FCompute>("FCompute<gpu>", FusedOpForwardGPU);

std::pair<std::vector<mxnet::TShape>, std::vector<mxnet::TShape>>
FusedOpHelperShape(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAttrs<mxnet::TShape>("shape", node_id);
}

std::pair<std::vector<int>, std::vector<int>>
FusedOpHelperType(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAttrs<int>("dtype", node_id);
}

NNVM_REGISTER_OP(_FusedOpHelper)
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<nnvm::TIsGhost>("TIsGhost", true)
.set_attr<exec::TIsFusionHelper>("TIsFusionHelper", true)
.set_attr<exec::FAccessSubgraphShape>("FAccessSubgraphShape", FusedOpHelperShape)
.set_attr<exec::FAccessSubgraphType>("FAccessSubgraphType", FusedOpHelperType);


std::pair<std::vector<mxnet::TShape>, std::vector<mxnet::TShape>>
FusedOpOutHelperShape(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAuxShape(node_id);
}

std::pair<std::vector<int>, std::vector<int>>
FusedOpOutHelperType(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAuxType(node_id);
}

NNVM_REGISTER_OP(_FusedOpOutHelper)
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<nnvm::TIsGhost>("TIsGhost", true)
.set_attr<exec::TIsFusionHelper>("TIsFusionHelper", true)
.set_attr<exec::FAccessSubgraphShape>("FAccessSubgraphShape", FusedOpOutHelperShape)
.set_attr<exec::FAccessSubgraphType>("FAccessSubgraphType", FusedOpOutHelperType);
}  // namespace mxnet
