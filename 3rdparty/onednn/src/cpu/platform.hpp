/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_PLATFORM_HPP
#define CPU_PLATFORM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#include "common/c_types_map.hpp"
#include "common/z_magic.hpp"

// Possible architectures:
// - DNNL_X64
// - DNNL_AARCH64
// - DNNL_PPC64
// - DNNL_S390X
// - DNNL_ARCH_GENERIC
// Target architecture macro is set to 1, others to 0. All macros are defined.

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_ARCH_GENERIC) \
        == 0
#if defined(__x86_64__) || defined(_M_X64)
#define DNNL_X64 1
#elif defined(__aarch64__)
#define DNNL_AARCH64 1
#elif defined(__powerpc64__) || defined(__PPC64__) || defined(_ARCH_PPC64)
#define DNNL_PPC64 1
#elif defined(__s390x__)
#define DNNL_S390X 1
#else
#define DNNL_ARCH_GENERIC 1
#endif
#endif // defined(DNNL_X64) + ... == 0

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_ARCH_GENERIC) \
        != 1
#error One and only one architecture should be defined at a time
#endif

#if !defined(DNNL_X64)
#define DNNL_X64 0
#endif
#if !defined(DNNL_AARCH64)
#define DNNL_AARCH64 0
#endif
#if !defined(DNNL_PPC64)
#define DNNL_PPC64 0
#endif
#if !defined(DNNL_S390X)
#define DNNL_S390X 0
#endif
#if !defined(DNNL_ARCH_GENERIC)
#define DNNL_ARCH_GENERIC 0
#endif

// Helper macros: expand the parameters only on the corresponding architecture.
// Equivalent to: #if DNNL_$ARCH ... #endif
#define DNNL_X64_ONLY(...) Z_CONDITIONAL_DO(DNNL_X64, __VA_ARGS__)
#define DNNL_PPC64_ONLY(...) Z_CONDITIONAL_DO(DNNL_PPC64_ONLY, __VA_ARGS__)
#define DNNL_S390X_ONLY(...) Z_CONDITIONAL_DO(DNNL_S390X_ONLY, __VA_ARGS__)
#define DNNL_AARCH64_ONLY(...) Z_CONDITIONAL_DO(DNNL_AARCH64, __VA_ARGS__)

// Using Arm Compute Library kernels is optional for AArch64 builds
// and can be enabled with the DNNL_AARCH64_USE_ACL CMake option
#if defined(DNNL_AARCH64) && defined(DNNL_AARCH64_USE_ACL)
#define DNNL_AARCH64_ACL_ONLY(...) __VA_ARGS__
#else
#define DNNL_AARCH64_ACL_ONLY(...)
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info();
dnnl_cpu_isa_t get_effective_cpu_isa();
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);

bool DNNL_API has_data_type_support(data_type_t data_type);
float s8s8_weights_scale_factor();

unsigned get_per_core_cache_size(int level);
unsigned get_num_cores();

constexpr int get_cache_line_size() {
    return 64;
}

int get_vector_register_size();

} // namespace platform

// XXX: find a better place for these values?
enum {
    PAGE_4K = 4096,
    PAGE_2M = 2097152,
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
