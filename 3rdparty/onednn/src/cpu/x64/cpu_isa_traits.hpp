/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_X64_CPU_ISA_TRAITS_HPP
#define CPU_X64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#define XBYAK_NO_EXCEPTION
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif
#include "cpu/x64/xbyak/xbyak.h"
#include "cpu/x64/xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

enum cpu_isa_bit_t : unsigned {
    sse41_bit = 1u << 0,
    avx_bit = 1u << 1,
    avx2_bit = 1u << 2,
    avx512_common_bit = 1u << 3,
    avx512_mic_bit = 1u << 4,
    avx512_mic_4ops_bit = 1u << 5,
    avx512_core_bit = 1u << 6,
    avx512_core_vnni_bit = 1u << 7,
    avx512_core_bf16_bit = 1u << 8,
    amx_tile_bit = 1u << 9,
    amx_int8_bit = 1u << 10,
    amx_bf16_bit = 1u << 11,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    sse41 = sse41_bit,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    avx512_common = avx512_common_bit | avx2,
    avx512_mic = avx512_mic_bit | avx512_common,
    avx512_mic_4ops = avx512_mic_4ops_bit | avx512_mic,
    avx512_core = avx512_core_bit | avx512_common,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    amx_tile = amx_tile_bit,
    amx_int8 = amx_int8_bit | amx_tile,
    amx_bf16 = amx_bf16_bit | amx_tile,
    avx512_core_bf16_amx_int8 = avx512_core_bf16 | amx_int8,
    avx512_core_bf16_amx_bf16 = avx512_core_bf16 | amx_bf16,
    avx512_core_amx = avx512_core_bf16 | amx_int8 | amx_bf16,
    // NOTE: Intel AMX is under initial support and turned off by default
    isa_all = ~0u & ~amx_tile_bit & ~amx_int8_bit & ~amx_bf16_bit,
};

const char *get_isa_info();

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

// pack struct so it can fit into a single 64-byte cache line
#pragma pack(push, 1)
struct palette_config_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
};
#pragma pack(pop)

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<sse41> {
    typedef Xbyak::Xmm Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sse41;
    static constexpr const char *user_option_env = "SSE41";
};

template <>
struct cpu_isa_traits<avx> {
    typedef Xbyak::Ymm Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx;
    static constexpr const char *user_option_env = "AVX";
};

template <>
struct cpu_isa_traits<avx2> : public cpu_isa_traits<avx> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "AVX2";
};

template <>
struct cpu_isa_traits<avx512_common> {
    typedef Xbyak::Zmm Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "AVX512_CORE";
};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_mic;
    static constexpr const char *user_option_env = "AVX512_MIC";
};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_mic> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_mic_4ops;
    static constexpr const char *user_option_env = "AVX512_MIC_4OPS";
};

template <>
struct cpu_isa_traits<avx512_core_vnni> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "AVX512_CORE_VNNI";
};

template <>
struct cpu_isa_traits<avx512_core_bf16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "AVX512_CORE_BF16";
};

template <>
struct cpu_isa_traits<avx512_core_amx> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_amx;
    static constexpr const char *user_option_env = "AVX512_CORE_AMX";
};

inline const Xbyak::util::Cpu &cpu() {
    const static Xbyak::util::Cpu cpu_;
    return cpu_;
}

namespace {

static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak::util;

    unsigned cpu_isa_mask = x64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case sse41: return cpu().has(Cpu::tSSE41);
        case avx: return cpu().has(Cpu::tAVX);
        case avx2: return cpu().has(Cpu::tAVX2);
        case avx512_common: return cpu().has(Cpu::tAVX512F);
        case avx512_core:
            return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW)
                    && cpu().has(Cpu::tAVX512VL) && cpu().has(Cpu::tAVX512DQ);
        case avx512_core_vnni:
            return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW)
                    && cpu().has(Cpu::tAVX512VL) && cpu().has(Cpu::tAVX512DQ)
                    && cpu().has(Cpu::tAVX512_VNNI);
        case avx512_mic:
            return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512CD)
                    && cpu().has(Cpu::tAVX512ER) && cpu().has(Cpu::tAVX512PF);
        case avx512_mic_4ops:
            return mayiuse(avx512_mic, soft) && cpu().has(Cpu::tAVX512_4FMAPS)
                    && cpu().has(Cpu::tAVX512_4VNNIW);
        case avx512_core_bf16:
            return mayiuse(avx512_core_vnni, soft)
                    && cpu().has(Cpu::tAVX512_BF16);
        case amx_tile: return cpu().has(Cpu::tAMX_TILE);
        case amx_int8:
            return mayiuse(amx_tile, soft) && cpu().has(Cpu::tAMX_INT8);
        case amx_bf16:
            return mayiuse(amx_tile, soft) && cpu().has(Cpu::tAMX_BF16);
        case avx512_core_bf16_amx_int8:
            return mayiuse(avx512_core_bf16, soft) && mayiuse(amx_int8, soft);
        case avx512_core_bf16_amx_bf16:
            return mayiuse(avx512_core_bf16, soft) && mayiuse(amx_bf16, soft);
        case avx512_core_amx:
            return mayiuse(avx512_core_bf16_amx_int8, soft)
                    && mayiuse(avx512_core_bf16_amx_bf16, soft);
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return isa == avx512_core_bf16;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == sse41 ? prefix STRINGIFY(sse41) : \
    ((isa) == avx ? prefix STRINGIFY(avx) : \
    ((isa) == avx2 ? prefix STRINGIFY(avx2) : \
    ((isa) == avx512_common ? prefix STRINGIFY(avx512_common) : \
    ((isa) == avx512_mic ? prefix STRINGIFY(avx512_mic) : \
    ((isa) == avx512_mic_4ops ? prefix STRINGIFY(avx512_mic_4ops) : \
    ((isa) == avx512_core ? prefix STRINGIFY(avx512_core) : \
    ((isa) == avx512_core_vnni ? prefix STRINGIFY(avx512_core_vnni) : \
    ((isa) == avx512_core_bf16 ? prefix STRINGIFY(avx512_core_bf16) : \
    ((isa) == avx512_core_bf16_amx_int8 ? prefix STRINGIFY(avx512_core_amx_int8) : \
    ((isa) == avx512_core_bf16_amx_bf16 ? prefix STRINGIFY(avx512_core_amx_bf16) : \
    prefix suffix_if_any))))))))))))
/* clang-format on */

namespace amx {

int get_max_palette();
int get_max_tiles(int palette);
int get_max_column_bytes(int palette);
int get_max_rows(int palette);

} // namespace amx

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
