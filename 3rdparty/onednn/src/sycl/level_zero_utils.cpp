/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dnnl/dnnl_config.h"

#if defined(DNNL_WITH_LEVEL_ZERO)

#include <stdio.h>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

#if !defined(__SYCL_COMPILER_VERSION)
#error "Unsupported compiler"
#endif

#if (__SYCL_COMPILER_VERSION <= 20200630)
#if defined(__INTEL_CLANG_COMPILER)
#include <CL/sycl/backend/Intel_level0.hpp>
#else
#define LEVEL_ZERO_NOT_SUPPORTED
#endif
#else // __SYCL_COMPILER_VERSION <= 20200630
#if (__SYCL_COMPILER_VERSION >= 20200818)
#define USE_GOLD_LEVEL_ZERO_SYCL_INTEROP
#endif
#include <CL/sycl/backend/level_zero.hpp>
#endif // __SYCL_COMPILER_VERSION <= 20200630

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "sycl/sycl_gpu_engine.hpp"

#include "sycl/level_zero_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

#if defined(LEVEL_ZERO_NOT_SUPPORTED)

device_uuid_t get_device_uuid(const cl::sycl::device &) {
    return device_uuid_t(0, 0);
}

status_t sycl_create_kernel_with_level_zero(std::unique_ptr<cl::sycl::kernel> &,
        const sycl_gpu_engine_t *, const std::vector<unsigned char> &,
        const std::string &, void **) {
    return status::unimplemented;
}

status_t sycl_destroy_kernel_with_level_zero(
        std::unique_ptr<cl::sycl::kernel> &, void *) {
    return status::unimplemented;
}

#else

namespace {

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            if (get_verbose()) { \
                printf("dnnl_verbose,gpu,ze_error,%d\n", (int)(res_)); \
                fflush(0); \
            } \
            return status::runtime_error; \
        } \
    } while (false)

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__) && defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(__linux__)
    void *handle = dlopen("libze_loader.so", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    HMODULE handle = LoadLibraryA("ze_loader.dll");
#endif
    if (!handle) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find Level Zero loader "
                   "library\n");
        assert(!"not expected");
        return nullptr;
    }

#if defined(__linux__)
    void *f = dlsym(handle, symbol);
#elif defined(_WIN32)
    void *f = GetProcAddress(handle, symbol);
#endif
    if (!f) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find symbol: %s\n", symbol);
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

#if defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
status_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, hDevice, desc, phModule, phBuildLog));
    return status::success;
}
#else
status_t func_zeModuleCreate(ze_device_handle_t hDevice,
        const ze_module_desc_t *desc, ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, desc, phModule, phBuildLog));
    return status::success;
}
#endif

status_t func_zeModuleDestroy(ze_module_handle_t hModule) {
    static auto f
            = find_ze_symbol<decltype(&zeModuleDestroy)>("zeModuleDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule));
    return status::success;
}

status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetProperties)>(
            "zeDeviceGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

} // namespace

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
device_uuid_t get_device_uuid(const cl::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t ze_device_properties;
#if defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
    auto ze_device = dev.get_native<cl::sycl::backend::level_zero>();
#else
    auto ze_device = (ze_device_handle_t)dev.get();
#endif
    auto status = func_zeDeviceGetProperties(ze_device, &ze_device_properties);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)ze_device_id[i]) << shift);
    }
    return device_uuid_t(uuid[0], uuid[1]);
}

status_t sycl_create_kernel_with_level_zero(
        std::unique_ptr<cl::sycl::kernel> &sycl_kernel,
        const sycl_gpu_engine_t *sycl_engine,
        const std::vector<unsigned char> &binary,
        const std::string &kernel_name, void **handle_to_destroy) {
    auto desc = ze_module_desc_t();
#if defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
#else
    desc.version = ZE_MODULE_DESC_VERSION_CURRENT;
#endif
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

#if defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
    auto ze_device
            = sycl_engine->device().get_native<cl::sycl::backend::level_zero>();
    auto ze_ctx = sycl_engine->context()
                          .get_native<cl::sycl::backend::level_zero>();
    CHECK(func_zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    auto sycl_program = cl::sycl::level_zero::make<cl::sycl::program>(
            sycl_engine->context(), ze_module);
#else
    auto ze_device = (ze_device_handle_t)sycl_engine->device().get();
    CHECK(func_zeModuleCreate(ze_device, &desc, &ze_module, nullptr));
#if defined(__INTEL_CLANG_COMPILER)
    auto sycl_program = cl::sycl::level0::make<cl::sycl::program>(
            sycl_engine->context(), ze_module);
#else
    auto sycl_program = cl::sycl::level_zero::make<cl::sycl::program>(
            sycl_engine->context(), ze_module);
#endif
#endif

    sycl_kernel.reset(
            new cl::sycl::kernel(sycl_program.get_kernel(kernel_name)));

    *handle_to_destroy = ze_module;

    return status::success;
}

status_t sycl_destroy_kernel_with_level_zero(
        std::unique_ptr<cl::sycl::kernel> &sycl_kernel,
        void *handle_to_destroy) {
    sycl_kernel.reset();
    auto status = status::success;
#if !defined(USE_GOLD_LEVEL_ZERO_SYCL_INTEROP)
    // L0 module should be destroyed manually in case of using pre Gold API
    auto ze_module = reinterpret_cast<ze_module_handle_t>(handle_to_destroy);
    status = func_zeModuleDestroy(ze_module);
#endif
    return status;
}

#endif // defined(LEVEL_ZERO_NOT_SUPPORTED)

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // DNNL_WITH_LEVEL_ZERO
