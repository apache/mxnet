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

#include <mxnet/rtc.h>
#include <typeinfo>

#include "cuda/utils.h"
#include "../operator/operator_common.h"

#if MXNET_USE_CUDA

namespace mxnet {
namespace rtc {

CudaModule::Chunk::Chunk(const char* source,
                         const std::vector<std::string>& options,
                         const std::vector<std::string>& exports) {
  NVRTC_CALL(nvrtcCreateProgram(&prog_, source, "source.cu", 0, nullptr, nullptr));
  for (const auto& i : exports)
    exports_.insert(i);
#if CUDA_VERSION >= 8000
  for (const auto& func : exports) {
    NVRTC_CALL(nvrtcAddNameExpression(prog_, func.c_str()));
  }
#else
  CHECK_EQ(exports.size(), 0)
      << "Exporting is only supported with CUDA 8.0 and above. "
      << "For lower version of CUDA, please prepend your kernel defintiions "
      << "with extern \"C\" instead.";
#endif
  std::vector<const char*> c_options;
  for (const auto& i : options)
    c_options.push_back(i.c_str());
  nvrtcResult compile_res = nvrtcCompileProgram(prog_, c_options.size(), c_options.data());
  if (compile_res != NVRTC_SUCCESS) {
    size_t err_size;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog_, &err_size));
    std::vector<char> err(err_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog_, err.data()));
    LOG(FATAL) << err.data();
  }

  bool use_ptx = true;
  for (const auto& opt : options) {
    if (opt.find("sm_") != std::string::npos) {
      use_ptx = false;
      break;
    }
  }

  if (use_ptx) {
    size_t ptx_size;
    NVRTC_CALL(nvrtcGetPTXSize(prog_, &ptx_size));
    ptx_.resize(ptx_size);
    NVRTC_CALL(nvrtcGetPTX(prog_, ptx_.data()));
  } else {
#if CUDA_VERSION >= 11010
    size_t cubin_size;
    NVRTC_CALL(nvrtcGetCUBINSize(prog_, &cubin_size));
    ptx_.resize(cubin_size);
    NVRTC_CALL(nvrtcGetCUBIN(prog_, ptx_.data()));
#else
    LOG(FATAL) << "Your CUDA version does not support compiling for sm_XX target. "
               << "Use compute_XX target instead or upgrade to CUDA 11.1 or later.";
#endif
  }
}

CudaModule::Chunk::~Chunk() {
  for (const auto& kv : mod_) {
    CUDA_DRIVER_CALL(cuModuleUnload(kv.second));
  }
  NVRTC_CALL(nvrtcDestroyProgram(&prog_));
}

CUfunction CudaModule::Chunk::GetFunction(const std::string& mangled_name, const Context& ctx) {
  CHECK_EQ(ctx.dev_mask(), Context::kGPU) << "CUDA Runtime compilation only supports Nvidia GPU.";
  auto iter = mod_.find(ctx.dev_id);
  mxnet::common::cuda::DeviceStore device_store;
  CUmodule module;
  if (iter != mod_.end()) {
    module = iter->second;
  } else {
    device_store.SetDevice(ctx.dev_id);
    CUDA_DRIVER_CALL(cuModuleLoadDataEx(&module, ptx_.data(), 0, nullptr, nullptr));
    mod_[ctx.dev_id] = module;
  }
  CUfunction function;
  auto err = cuModuleGetFunction(&function, module, mangled_name.c_str());
  if (err == CUDA_ERROR_NOT_FOUND) {
    LOG(FATAL) << "Cannot find cuda kernel with name '" << mangled_name
               << "'. Please either prepend kernel definition "
               << "with 'extern \"C\"' or add its name to exports "
               << "when creating CudaModule.";
  }
  CUDA_DRIVER_CALL(err);
  return function;
}

std::shared_ptr<CudaModule::Kernel> CudaModule::GetKernel(const std::string& name,
                                                          const std::vector<ArgType>& signature) {
  std::string mangled_name = name;
#if CUDA_VERSION >= 8000
  if (ptr_->exports_.count(name)) {
    const char* c_mangled_name;
    NVRTC_CALL(nvrtcGetLoweredName(ptr_->prog_, name.c_str(), &c_mangled_name));
    mangled_name = c_mangled_name;
  }
#endif
  return std::shared_ptr<Kernel>(new Kernel(ptr_, mangled_name, signature));
}

CudaModule::Kernel::Kernel(const std::shared_ptr<CudaModule::Chunk>& mod,
                           const std::string& mangled_name,
                           const std::vector<ArgType>& signature)
    : mangled_name_(mangled_name), signature_(signature), mod_(mod) {}

void CudaModule::Kernel::Launch(const Context& ctx,
                                const std::vector<dmlc::any>& args,
                                uint32_t grid_dim_x,
                                uint32_t grid_dim_y,
                                uint32_t grid_dim_z,
                                uint32_t block_dim_x,
                                uint32_t block_dim_y,
                                uint32_t block_dim_z,
                                uint32_t shared_mem) {
  CHECK_EQ(ctx.dev_mask(), Context::kGPU) << "CUDA Runtime compilation only supports Nvidia GPU.";

  auto mod       = mod_;
  auto arg_types = signature();

  CUfunction function;
  auto iter = func_.find(ctx.dev_id);
  if (iter != func_.end()) {
    function = iter->second;
  } else {
    function          = mod_->GetFunction(mangled_name_, ctx);
    func_[ctx.dev_id] = function;
  }

  std::vector<Engine::VarHandle> read_vars, write_vars;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (!arg_types[i].is_ndarray)
      continue;
    const auto& array = dmlc::get<NDArray>(args[i]);
    CHECK_EQ(array.dtype(), arg_types[i].dtype)
        << "The i-th argument is expected to be an NDArray of "
        << op::type_string(arg_types[i].dtype) << " type, but got "
        << op::type_string(array.dtype()) << " instead.";
    if (arg_types[i].is_const) {
      read_vars.emplace_back(array.var());
    } else {
      write_vars.emplace_back(array.var());
    }
  }

  Engine::Get()->PushSync(
      [function,
       mod,
       args,
       arg_types,
       grid_dim_x,
       grid_dim_y,
       grid_dim_z,
       block_dim_x,
       block_dim_y,
       block_dim_z,
       shared_mem](RunContext rctx) {
        std::vector<void*> p_args;
        for (size_t i = 0; i < arg_types.size(); ++i) {
          if (arg_types[i].is_ndarray) {
            const auto& array = dmlc::get<NDArray>(args[i]);
            p_args.push_back(reinterpret_cast<void*>(const_cast<void**>(&array.data().dptr_)));
          } else {
            MSHADOW_TYPE_SWITCH(arg_types[i].dtype, DType, {
              const auto& number = dmlc::get<DType>(args[i]);
              p_args.push_back(const_cast<DType*>(&number));
            });
          }
        }

        mshadow::Stream<gpu>* s = rctx.get_stream<gpu>();
        CUDA_DRIVER_CALL(cuLaunchKernel(function,
                                        grid_dim_x,
                                        grid_dim_y,
                                        grid_dim_z,
                                        block_dim_x,
                                        block_dim_y,
                                        block_dim_z,
                                        shared_mem,
                                        s->stream_,
                                        p_args.data(),
                                        nullptr));
        CUDA_CALL(cudaStreamSynchronize(s->stream_));
      },
      ctx,
      read_vars,
      write_vars,
      FnProperty::kNormal,
      0,
      mangled_name_.c_str());
}

}  // namespace rtc
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
