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
 * Copyright (c) 2019 by Contributors
 * \file op_module.cc
 * \brief Invoke registered TVM operators.
 * \author Yizhi Liu
 */
#if MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <string>
#include <vector>
#include "op_module.h"

namespace dmlc {
  DMLC_REGISTRY_ENABLE(::tvm::runtime::TVMOpConfig);
}  // namespace dmlc

namespace tvm {
namespace runtime {

void TVMOpModule::Load(const std::string &filepath) {
  static const PackedFunc *f_load = Registry::Get("module._LoadFromFile");
  std::lock_guard<std::mutex> lock(mutex_);
  Module module = (*f_load)(filepath, "");
  module_ptr_ = std::make_shared<Module>();
  *module_ptr_ = module;
}

PackedFunc GetFunction(const std::shared_ptr<Module> &module,
                       const std::string &op_name,
                       const std::vector<mxnet::TBlob> &args) {
  std::ostringstream func_name;
  func_name << op_name;
  for (const auto &arg : args) {
    switch (arg.type_flag_) {
      case mshadow::kFloat32:
        func_name << "float32";
        break;
      case mshadow::kFloat64:
        func_name << "float64";
        break;
      case mshadow::kFloat16:
        func_name << "float16";
        break;
      case mshadow::kUint8:
        func_name << "uint8";
        break;
      case mshadow::kInt32:
        func_name << "int32";
        break;
      case mshadow::kInt8:
        func_name << "int8";
        break;
      case mshadow::kInt64:
        func_name << "int64";
        break;
      case mshadow::kBool:
        func_name << "bool";
        break;
      default:
        LOG(FATAL) << "Unknown dtype " << arg.type_flag_;
    }
    func_name << "_" << arg.shape_.ndim();
  }
  return module->GetFunction(func_name.str(), false);
}

void TVMOpModule::Call(const std::string &func_name,
                       const mxnet::OpContext& ctx,
                       const std::vector<mxnet::TBlob> &args) const {
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  type_codes.resize(args.size());
  values.resize(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    type_codes[i] = kArrayHandle;
    values[i].v_handle = const_cast<DLTensor *>(&(args[i].dltensor()));
  }

  TVMArgs tvm_args(&values[0], &type_codes[0], args.size());
  TVMRetValue rv;

#if MXNET_USE_CUDA
  int dev_type = (ctx.run_ctx.ctx.dev_type == mxnet::Context::DeviceType::kGPU) ? kDLGPU : kDLCPU;
  int dev_id = ctx.run_ctx.ctx.dev_id;
  if (dev_type == kDLGPU) {
    void *stream = static_cast<void *>(ctx.run_ctx.get_stream<mxnet::gpu>()->stream_);
    TVMSetStream(dev_type, dev_id, stream);
  }
#endif
  GetFunction(module_ptr_, func_name, args).CallPacked(tvm_args, &rv);
#if MXNET_USE_CUDA
  if (dev_type == kDLGPU) {
    TVMSetStream(dev_type, dev_id, nullptr);
  }
#endif
}

void TVMOpModule::CallEx(const std::string &func_name,
                         const mxnet::OpContext& ctx,
                         const std::vector<mxnet::TBlob>& tblobs,
                         TVMArgs tvm_args) const {
  TVMRetValue rv;

#if MXNET_USE_CUDA
  int dev_type = (ctx.run_ctx.ctx.dev_type == mxnet::Context::DeviceType::kGPU) ? kDLGPU : kDLCPU;
  int dev_id = ctx.run_ctx.ctx.dev_id;
  if (dev_type == kDLGPU) {
    void *stream = static_cast<void *>(ctx.run_ctx.get_stream<mxnet::gpu>()->stream_);
    TVMSetStream(dev_type, dev_id, stream);
  }
#endif
  GetFunction(module_ptr_, func_name, tblobs).CallPacked(tvm_args, &rv);
#if MXNET_USE_CUDA
  if (dev_type == kDLGPU) {
    TVMSetStream(dev_type, dev_id, nullptr);
  }
#endif
}

const TVMOpConfig& GetOpConfig(const std::string& name) {
  const TVMOpConfig* ret = ::dmlc::Registry<TVMOpConfig>::Get()->Find(name);
  CHECK(ret != NULL)
    << "op " << name << "does not exist.";
  return *ret;
}

}  // namespace runtime
}  // namespace tvm

#endif  // MXNET_USE_TVM_OP
