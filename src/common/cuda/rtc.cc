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

#include "mxnet/base.h"

#if MXNET_USE_CUDA

#include <nvrtc.h>

#include <mutex>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "rtc.h"
#include "rtc/half-inl.h"
#include "rtc/util-inl.h"
#include "rtc/forward_functions-inl.h"
#include "rtc/backward_functions-inl.h"
#include "rtc/vectorization-inl.h"
#include "rtc/special_functions-inl.h"
#include "utils.h"


namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

std::mutex lock;

namespace util {

std::string to_string(OpReqType req) {
  switch (req) {
    case kNullOp:
      return "OpReqType::kNullOp";
    case kWriteTo:
    case kWriteInplace:
      return "OpReqType::kWriteTo";
    case kAddTo:
      return "OpReqType::kAddTo";
  }
  LOG(FATAL) << "Unrecognized req.";
}

TypeInfo mshadow_type_info(int type_flag) {
  using namespace mshadow;
  switch (type_flag) {
    case kFloat32:
      return TypeInfo("float32", sizeof(float));
    case kFloat64:
      return TypeInfo("float64", sizeof(double));
    case kFloat16:
      return TypeInfo("float16", 2);
    case kUint8:
      return TypeInfo("uint8", sizeof(uint8_t));
    case kInt32:
      return TypeInfo("int32", sizeof(int32_t));
    case kInt8:
      return TypeInfo("int8", sizeof(int8_t));
    case kInt64:
      return TypeInfo("int64", sizeof(int64_t));
    case kBool:
      return TypeInfo("bool", sizeof(bool));
    default:
      LOG(FATAL) << "Unknown type flag " << type_flag;
      return TypeInfo("INVALID", 1);
  }
}

}  // namespace util

namespace {

// Obtain compilation log from the program.
std::string GetCompileLog(nvrtcProgram program) {
  size_t log_size_including_null;
  NVRTC_CALL(nvrtcGetProgramLogSize(program, &log_size_including_null));
  // For most std::string implementations, this is probably 1 char bigger than needed.  OK though.
  std::string log(log_size_including_null, '\0');
  NVRTC_CALL(nvrtcGetProgramLog(program, &log[0]));
  // Make sure the string reflects the true size (so minus the null terminator).
  log.resize(log_size_including_null - 1);
  return log;
}

// Obtain compilation result (ptx assembly) from the program.
std::string GetPtx(nvrtcProgram program) {
  size_t ptx_size_including_null;
  NVRTC_CALL(nvrtcGetPTXSize(program, &ptx_size_including_null));
  // For most std::string implementations, this is probably 1 char bigger than needed.  OK though.
  std::string ptx(ptx_size_including_null, '\0');
  NVRTC_CALL(nvrtcGetPTX(program, &ptx[0]));
  // Make sure the string reflects the true size (so minus the null terminator).
  ptx.resize(ptx_size_including_null - 1);
  return ptx;
}

}  // namespace

CUfunction get_function(const std::string &code,
                        const std::string &kernel_name,
                        int dev_id) {
  constexpr int CACHESIZE_WARN_THRESHOLD = 10000;
  std::lock_guard<std::mutex> l(lock);
  // Local class for value type of compile cache
  struct KernelInfo {
    std::string mangled_name;
    std::string ptx;
    std::vector<CUfunction> functions;
  };
  // Maps from the cuda source code (minus header) to the ptx and jit-compiled CUfunctions.
  using KernelCache = std::unordered_map<std::string, KernelInfo>;
  // Per-gpu-architecture compiled kernel cache with jit-compiled function for each device context
  static std::unordered_map<int32_t, KernelCache> compiled_kernels;
  int sm_arch = SMArch(dev_id);
  KernelCache& compiled_kernels_this_arch = compiled_kernels[sm_arch];  // make null map as needed
  KernelInfo& kinfo = compiled_kernels_this_arch[code];                 // make KernelInfo as needed
  if (kinfo.ptx.size() == 0) {
    // It's the first time we've seen this kernel, so we need to generate the ptx and mangled_name.
    static std::string common_header =
        std::string(fp16_support_string) + "\n" +
        type_support_string + "\n" +
        util_string + "\n" +
        float_limits() +
        special_functions_definitions + '\n' +
        function_definitions + "\n" +
        backward_function_definitions + "\n" +
        vectorization_support_string + "\n";
    std::string code_with_header = common_header + code;
    // If verbose mode, output kernel source, though not including the common header
    if (dmlc::GetEnv("MXNET_RTC_VERBOSE", false)) {
      LOG(INFO) << "\n" << std::string(80, '-') << "\n" << code;
    }
    if (compiled_kernels_this_arch.size() == CACHESIZE_WARN_THRESHOLD + 1 &&
        dmlc::GetEnv("MXNET_RTC_SIZE_WARNING", true)) {
      LOG(WARNING) << "The number of different compiled kernels exceeds "
                   << CACHESIZE_WARN_THRESHOLD
                   << ".  Set MXNET_RTC_SIZE_WARNING=0 to quiet this warning.";
    }
    nvrtcProgram program;
    std::ofstream f("debug.log");
    f << code_with_header;
    f.close();

    NVRTC_CALL(nvrtcCreateProgram(&program,                                  // prog
                                  &code_with_header[0],                      // buffer
                                  (kernel_name + "_kernel.cu").c_str(),      // name
                                  0,                                         // num headers
                                  nullptr,                                   // headers
                                  nullptr));                                 // include names

    std::string gpu_arch_arg = "--gpu-architecture=compute_" + std::to_string(sm_arch);
    const char *opts[] = {gpu_arch_arg.c_str(),
                          "--std=c++11"};
    const std::string kernel_name_demangled = kernel_name;
    NVRTC_CALL(nvrtcAddNameExpression(program, (kernel_name_demangled).c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(program,  // prog
                                                    2,        // num options
                                                    opts);    // options
    CHECK_EQ(compileResult, NVRTC_SUCCESS)
        << "NVRTC Compilation failed. Please set environment variable MXNET_USE_FUSION to 0.\n"
        << GetCompileLog(program);

    kinfo.ptx = GetPtx(program);
    const char *mangled_name;
    NVRTC_CALL(nvrtcGetLoweredName(program,
                                   kernel_name_demangled.c_str(),
                                   &mangled_name));
    kinfo.mangled_name = mangled_name;
    // Destroy the program.
    NVRTC_CALL(nvrtcDestroyProgram(&program));
  }
  // Ensure function array is deep enough to index by dev_id
  while (kinfo.functions.size() <= static_cast<size_t>(dev_id))
    kinfo.functions.push_back(static_cast<CUfunction>(nullptr));
  // Jit-compile ptx for the device as needed
  if (kinfo.functions[dev_id] == static_cast<CUfunction>(nullptr)) {
    // Make sure driver context is set to the proper device
    CUdevice cu_device;
    CUcontext context;
    CUDA_DRIVER_CALL(cuDeviceGet(&cu_device, dev_id));
    CUDA_DRIVER_CALL(cuDevicePrimaryCtxRetain(&context, cu_device));
    // Jit-compile ptx for the driver's current context
    CUmodule module;
    CUDA_DRIVER_CALL(cuModuleLoadData(&module, kinfo.ptx.c_str()));
    CUDA_DRIVER_CALL(cuModuleGetFunction(&kinfo.functions[dev_id],
                                         module,
                                         kinfo.mangled_name.c_str()));
  }
  return kinfo.functions[dev_id];
}

void launch(CUfunction function,
            const dim3 grid_dim,
            const dim3 block_dim,
            unsigned int shared_mem_bytes,
            mshadow::Stream<gpu> *stream,
            std::vector<const void*> *args) {
  CHECK(args->size() != 0) <<
    "Empty argument list passed to a kernel.";
  CUDA_DRIVER_CALL(
      cuLaunchKernel(function,                    // function to launch
        grid_dim.x, grid_dim.y, grid_dim.z,       // grid dim
        block_dim.x, block_dim.y, block_dim.z,    // block dim
        shared_mem_bytes,                         // shared memory
        mshadow::Stream<gpu>::GetStream(stream),  // stream
        const_cast<void**>(args->data()),         // arguments
        nullptr));
}

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
