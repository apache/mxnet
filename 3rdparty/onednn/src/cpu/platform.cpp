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

#include "cpu/platform.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info() {
#if DNNL_X64
    return x64::get_isa_info();
#else
    return "Generic";
#endif
}

dnnl_cpu_isa_t get_effective_cpu_isa() {
#if DNNL_X64
    return x64::get_effective_cpu_isa();
#else
    return dnnl_cpu_isa_all;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
#if DNNL_X64
    return x64::set_max_cpu_isa(isa);
#else
    return status::unimplemented;
#endif
}

bool has_data_type_support(data_type_t data_type) {
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core);
#else
            return false;
#endif
        case data_type::f16: return false;
        default: return true;
    }
}

float s8s8_weights_scale_factor() {
#if DNNL_X64
    return x64::mayiuse(x64::avx512_core_vnni) ? 1.0f : 0.5f;
#else
    return 1.0f;
#endif
}

unsigned get_per_core_cache_size(int level) {
    auto guess = [](int level) {
        switch (level) {
            case 1: return 32U * 1024;
            case 2: return 512U * 1024;
            case 3: return 1024U * 1024;
            default: return 0U;
        }
    };

#if DNNL_X64
    using namespace x64;
    if (cpu().getDataCacheLevels() == 0) return guess(level);

    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
    } else
        return 0;
#else
    return guess(level);
#endif
}

unsigned get_num_cores() {
#if DNNL_X64
    return x64::cpu().getNumCores(Xbyak::util::CoreLevel);
#else
    return 1;
#endif
}

int get_vector_register_size() {
#if DNNL_X64
    using namespace x64;
    if (mayiuse(avx512_common)) return cpu_isa_traits<avx512_common>::vlen;
    if (mayiuse(avx)) return cpu_isa_traits<avx>::vlen;
    if (mayiuse(sse41)) return cpu_isa_traits<sse41>::vlen;
#endif
    return 0;
}

} // namespace platform
} // namespace cpu
} // namespace impl
} // namespace dnnl
