/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <assert.h>
#include <map>
#include <vector>

#include "common/memory.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_reorder_pd.hpp"

#include "cpu/rnn/rnn_reorders.hpp"
#include "cpu/simple_reorder.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_reorder.hpp"
#include "cpu/x64/wino_reorder.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using rpd_create_f = dnnl::impl::engine_t::reorder_primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

struct reorder_impl_key_t {
    data_type_t src_dt;
    data_type_t dst_dt; // data_type::undef if arbitrary
    int ndims; // 0 if arbitrary

    bool operator<(const reorder_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    enum { MAX_DT_NUM = 10 };
    size_t value() const {
        return ((size_t)ndims * MAX_DT_NUM + (size_t)src_dt) * MAX_DT_NUM
                + (size_t)dst_dt;
    }
};

using impl_list_map_t = std::map<reorder_impl_key_t, std::vector<rpd_create_f>>;

#define REG_SR(idt, ifmt, odt, ofmt, ...) \
    simple_reorder_t<idt, ifmt, odt, ofmt, __VA_ARGS__>::pd_t::create

#define REG_SR_BIDIR(idt, ifmt, odt, ofmt) \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order::keep), \
            REG_SR(idt, ifmt, odt, ofmt, fmt_order::reverse)

#define REG_SR_DIRECT_COPY(idt, odt) \
    REG_SR(idt, any, odt, any, fmt_order::any, spec::direct_copy), \
            REG_SR(idt, any, odt, any, fmt_order::any, \
                    spec::direct_copy_except_dim_0)

#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && !defined(__clang__))
/* Direct copy for icc which is faster than jitted code;
 * Direct copy for gcc which might or might not be faster than jitted
 * code, but still worth it because doesn't require jitting, i.e. much
 * faster creation time. This is tentative solution and should be
 * removed later (when we will cache jitted code?...). */
#define REG_FAST_DIRECT_COPY_F32_F32_COMMA REG_SR_DIRECT_COPY(f32, f32),
#else
#define REG_FAST_DIRECT_COPY_F32_F32_COMMA
#endif

/* regular reorders */
#ifdef __INTEL_COMPILER
/* direct copy for icc, which is faster than jitted code */
#define REG_FAST_DIRECT_COPY_COMMA(sdt, ddt) REG_SR_DIRECT_COPY(sdt, ddt),
#else
#define REG_FAST_DIRECT_COPY_COMMA(sdt, ddt)
#endif

// clang-format off

const impl_list_map_t regular_impl_list_map {
    // f32 -> bf16
    {{f32, bf16, 0}, {
        rnn_weights_reorder_t<f32, bf16>::pd_t::create,

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, bf16, nChw16c),
        REG_SR_BIDIR(f32, any, bf16, nCdhw16c),

        REG_SR(f32, oihw, bf16, OIhw8i16o2i, fmt_order::keep),
        REG_SR(f32, goihw, bf16, gOIhw8i16o2i, fmt_order::keep),
        REG_SR(f32, oihw, bf16, OIhw8o16i2o, fmt_order::keep),
        REG_SR(f32, goihw, bf16, gOIhw8o16i2o, fmt_order::keep),
        REG_SR(f32, oihw, bf16, IOhw8o16i2o, fmt_order::keep),
        REG_SR(f32, goihw, bf16, gIOhw8o16i2o, fmt_order::keep),
        REG_SR(f32, oihw, bf16, OIhw16i16o, fmt_order::keep),
        REG_SR(f32, goihw, bf16, gOIhw16i16o, fmt_order::keep),

        REG_SR(f32, any, bf16, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f32 -> f16
    {{f32, f16, 0}, {
        REG_SR(f32, any, f16, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f32 -> f32
    {{f32, f32, 0}, {
        REG_FAST_DIRECT_COPY_F32_F32_COMMA

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},
    {{f32, f32, 3}, {
        REG_FAST_DIRECT_COPY_F32_F32_COMMA

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, f32, nCw16c),
        REG_SR_BIDIR(f32, any, f32, nCw8c),
        REG_SR_BIDIR(f32, any, f32, nCw4c),

        REG_SR_BIDIR(f32, nCw4c, f32, nCw16c),
        REG_SR_BIDIR(f32, nCw8c, f32, nCw16c),

        REG_SR_BIDIR(f32, any, f32, OIw4i4o),
        REG_SR_BIDIR(f32, any, f32, OIw4o4i),
        REG_SR_BIDIR(f32, any, f32, OIw8i8o),
        REG_SR_BIDIR(f32, any, f32, OIw8o8i),

        REG_SR_BIDIR(f32, any, f32, OIw16o16i),
        REG_SR_BIDIR(f32, any, f32, OIw16i16o),
        REG_SR_BIDIR(f32, any, f32, IOw16o16i),

        REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},
    {{f32, f32, 4}, {
        DNNL_X64_ONLY(x64::wino_reorder_t<f32, f32>::pd_t::create,)

        REG_FAST_DIRECT_COPY_F32_F32_COMMA

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, f32, nChw16c),
        REG_SR_BIDIR(f32, any, f32, nChw8c),
        REG_SR_BIDIR(f32, any, f32, nChw4c),

        REG_SR_BIDIR(f32, nChw4c, f32, nChw16c),
        REG_SR_BIDIR(f32, nChw8c, f32, nChw16c),

        REG_SR_BIDIR(f32, any, f32, gOIw4i4o),
        REG_SR_BIDIR(f32, any, f32, gOIw4o4i),
        REG_SR_BIDIR(f32, any, f32, gOIw8i8o),
        REG_SR_BIDIR(f32, any, f32, gOIw8o8i),

        REG_SR_BIDIR(f32, any, f32, gOIw16o16i),
        REG_SR_BIDIR(f32, any, f32, gOIw16i16o),
        REG_SR_BIDIR(f32, any, f32, gIOw16o16i),

        REG_SR_BIDIR(f32, any, f32, OIhw4i4o),
        REG_SR_BIDIR(f32, any, f32, OIhw4o4i),
        REG_SR_BIDIR(f32, any, f32, Ohwi8o),

        REG_SR_BIDIR(f32, any, f32, OIhw8i8o),
        REG_SR_BIDIR(f32, any, f32, OIhw8o8i),

        REG_SR_BIDIR(f32, any, f32, Oihw4o),
        REG_SR_BIDIR(f32, any, f32, Oihw16o),
        REG_SR_BIDIR(f32, any, f32, Ohwi4o),
        REG_SR_BIDIR(f32, any, f32, Ohwi16o),
        REG_SR_BIDIR(f32, any, f32, OIhw16o16i),
        REG_SR_BIDIR(f32, any, f32, OIhw16i16o),
        REG_SR_BIDIR(f32, any, f32, IOhw16o16i),

        REG_SR_BIDIR(f32, any, f32, OIhw4i16o4i),

        REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},
    {{f32, f32, 5}, {
        DNNL_X64_ONLY(x64::wino_reorder_t<f32, f32>::pd_t::create,)
        rnn_weights_reorder_t<f32, f32>::pd_t::create,

        REG_FAST_DIRECT_COPY_F32_F32_COMMA

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, f32, nCdhw16c),
        REG_SR_BIDIR(f32, any, f32, nCdhw8c),
        REG_SR_BIDIR(f32, any, f32, nCdhw4c),

        REG_SR_BIDIR(f32, nCdhw4c, f32, nCdhw16c),
        REG_SR_BIDIR(f32, nCdhw8c, f32, nCdhw16c),

        REG_SR_BIDIR(f32, any, f32, gOIhw4i4o),
        REG_SR_BIDIR(f32, any, f32, gOIhw4o4i),
        REG_SR_BIDIR(f32, any, f32, gOhwi8o),
        REG_SR_BIDIR(f32, any, f32, gOIhw8i8o),
        REG_SR_BIDIR(f32, any, f32, gOIhw8o8i),

        REG_SR_BIDIR(f32, any, f32, gOihw4o),
        REG_SR_BIDIR(f32, any, f32, gOihw16o),
        REG_SR_BIDIR(f32, any, f32, gOhwi4o),
        REG_SR_BIDIR(f32, any, f32, gOhwi16o),
        REG_SR_BIDIR(f32, any, f32, gOIhw16o16i),
        REG_SR_BIDIR(f32, any, f32, gOIhw16i16o),
        REG_SR_BIDIR(f32, any, f32, gIOhw16o16i),

        REG_SR_BIDIR(f32, any, f32, OIdhw4i4o),
        REG_SR_BIDIR(f32, any, f32, OIdhw4o4i),
        REG_SR_BIDIR(f32, any, f32, Odhwi8o),
        REG_SR_BIDIR(f32, any, f32, OIdhw8i8o),
        REG_SR_BIDIR(f32, any, f32, OIdhw8o8i),

        REG_SR_BIDIR(f32, any, f32, Oidhw4o),
        REG_SR_BIDIR(f32, any, f32, Oidhw16o),
        REG_SR_BIDIR(f32, any, f32, Odhwi16o),
        REG_SR_BIDIR(f32, any, f32, OIdhw16o16i),
        REG_SR_BIDIR(f32, any, f32, OIdhw16i16o),
        REG_SR_BIDIR(f32, any, f32, IOdhw16o16i),

        REG_SR_BIDIR(f32, any, f32, gOIhw4i16o4i),

        REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},
    {{f32, f32, 6}, {
        REG_FAST_DIRECT_COPY_F32_F32_COMMA

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, f32, gOIdhw4i4o),
        REG_SR_BIDIR(f32, any, f32, gOIdhw4o4i),
        REG_SR_BIDIR(f32, any, f32, gOdhwi8o),
        REG_SR_BIDIR(f32, any, f32, gOIdhw8i8o),
        REG_SR_BIDIR(f32, any, f32, gOIdhw8o8i),

        REG_SR_BIDIR(f32, any, f32, gOidhw4o),
        REG_SR_BIDIR(f32, any, f32, gOidhw16o),
        REG_SR_BIDIR(f32, any, f32, gOdhwi16o),
        REG_SR_BIDIR(f32, any, f32, gOIdhw16o16i),
        REG_SR_BIDIR(f32, any, f32, gOIdhw16i16o),
        REG_SR_BIDIR(f32, any, f32, gIOdhw16o16i),

        REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f32 -> s32
    {{f32, s32, 0}, {
        REG_FAST_DIRECT_COPY_COMMA(f32, s32)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, s32, nChw16c),

        REG_SR(f32, any, s32, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f32 -> s8
    {{f32, s8, 0}, {
        DNNL_X64_ONLY(x64::wino_reorder_t<f32, s8>::pd_t::create,)
        rnn_weights_reorder_s8_t<f32>::pd_t::create,

        REG_FAST_DIRECT_COPY_COMMA(f32, s8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, s8, nChw16c),
        REG_SR_BIDIR(f32, any, s8, OIhw4i16o4i),
        REG_SR_BIDIR(f32, any, s8, gOIhw4i16o4i),

        REG_SR(f32, any, s8, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f32 -> u8
    {{f32, u8, 0}, {
        rnn_data_reorder_t<f32, u8>::pd_t::create,

        REG_FAST_DIRECT_COPY_COMMA(f32, u8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(f32, any, u8, nChw16c),

        REG_SR(f32, any, u8, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // bf16 ->
    {{bf16, data_type::undef, 0}, {
        rnn_weights_reorder_t<bf16, bf16>::pd_t::create,

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(bf16, any, f32, nChw16c),
        REG_SR_BIDIR(bf16, any, f32, nCdhw16c),

        REG_SR_BIDIR(bf16, any, s8, nChw16c),
        REG_SR_BIDIR(bf16, any, s8, nCdhw16c),

        REG_SR_BIDIR(bf16, any, u8, nChw16c),
        REG_SR_BIDIR(bf16, any, u8, nCdhw16c),

        REG_SR_BIDIR(bf16, any, bf16, nChw16c),
        REG_SR_BIDIR(bf16, any, bf16, nCdhw16c),

        REG_SR_BIDIR(bf16, any, f32, OIdhw16o16i),
        REG_SR_BIDIR(bf16, any, f32, OIdhw16i16o),
    
        REG_SR_BIDIR(bf16, any, s8, OIdhw16o16i),
        REG_SR_BIDIR(bf16, any, s8, OIdhw16i16o),

        REG_SR_BIDIR(bf16, any, u8, OIdhw16o16i),
        REG_SR_BIDIR(bf16, any, u8, OIdhw16i16o),

        REG_SR(bf16, any, bf16, any, fmt_order::any, spec::reference),
        REG_SR(bf16, any, f32, any, fmt_order::any, spec::reference),
        REG_SR(bf16, any, s8, any, fmt_order::any, spec::reference),
        REG_SR(bf16, any, u8, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // f16 ->
    {{f16, data_type::undef, 0}, {
        REG_SR(f16, any, f16, any, fmt_order::any, spec::reference),
        REG_SR(f16, any, f32, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // s32 ->
    {{s32, data_type::undef, 0}, {
        REG_FAST_DIRECT_COPY_COMMA(s32, f32)
        REG_FAST_DIRECT_COPY_COMMA(s32, s32)
        REG_FAST_DIRECT_COPY_COMMA(s32, s8)
        REG_FAST_DIRECT_COPY_COMMA(s32, u8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(s32, any, f32, nChw16c),
        REG_SR_BIDIR(s32, any, s32, nChw16c),
        REG_SR_BIDIR(s32, any, s8, nChw16c),
        REG_SR_BIDIR(s32, any, u8, nChw16c),

        REG_SR(s32, any, f32, any, fmt_order::any, spec::reference),
        REG_SR(s32, any, s32, any, fmt_order::any, spec::reference),
        REG_SR(s32, any, s8, any, fmt_order::any, spec::reference),
        REG_SR(s32, any, u8, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // s8 ->
    {{s8, data_type::undef, 0}, {
        rnn_weights_reorder_s8_t<s8>::pd_t::create,

        REG_FAST_DIRECT_COPY_COMMA(s8, f32)
        REG_FAST_DIRECT_COPY_COMMA(s8, s32)
        REG_FAST_DIRECT_COPY_COMMA(s8, bf16)
        REG_FAST_DIRECT_COPY_COMMA(s8, s8)
        REG_FAST_DIRECT_COPY_COMMA(s8, u8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(s8, any, f32, nChw16c),
        REG_SR_BIDIR(s8, any, s32, nChw16c),
        REG_SR_BIDIR(s8, any, bf16, nChw16c),
        REG_SR_BIDIR(s8, any, s8, nChw16c),
        REG_SR_BIDIR(s8, any, u8, nChw16c),

        REG_SR_BIDIR(s8, any, f32, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, bf16, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, s8, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, f32, gOIhw4i16o4i),
        REG_SR_BIDIR(s8, any, bf16, gOIhw4i16o4i),
        REG_SR_BIDIR(s8, any, s8, gOIhw4i16o4i),

        REG_SR(s8, any, f32, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, s32, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, bf16, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, s8, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, u8, any, fmt_order::any, spec::reference),

        nullptr,
    }},

    // u8 ->
    {{u8, data_type::undef, 0}, {
        REG_FAST_DIRECT_COPY_COMMA(u8, f32)
        REG_FAST_DIRECT_COPY_COMMA(u8, s32)
        REG_FAST_DIRECT_COPY_COMMA(u8, bf16)
        REG_FAST_DIRECT_COPY_COMMA(u8, s8)
        REG_FAST_DIRECT_COPY_COMMA(u8, u8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)

        REG_SR_BIDIR(u8, any, f32, nChw16c),
        REG_SR_BIDIR(u8, any, s32, nChw16c),
        REG_SR_BIDIR(u8, any, bf16, nChw16c),
        REG_SR_BIDIR(u8, any, s8, nChw16c),
        REG_SR_BIDIR(u8, any, u8, nChw16c),

        REG_SR(u8, any, f32, any, fmt_order::any, spec::reference),
        REG_SR(u8, any, s32, any, fmt_order::any, spec::reference),
        REG_SR(u8, any, bf16, any, fmt_order::any, spec::reference),
        REG_SR(u8, any, u8, any, fmt_order::any, spec::reference),
        REG_SR(u8, any, s8, any, fmt_order::any, spec::reference),

        nullptr,
    }},
};

/* conv reorders w/ compensation */
const impl_list_map_t comp_s8s8_impl_list_map {
    // f32 -> s8
    {{f32, s8, 3}, {
        REG_SR(f32, any, s8, wio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oiw, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wio, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oiw, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wio, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oiw, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wio, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{f32, s8, 4}, {
        REG_SR(f32, any, s8, hwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, any, s8, wigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oihw, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwio, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwio, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oihw, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwio, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oihw, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goiw, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, wigo, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{f32, s8, 5}, {
        REG_SR(f32, any, s8, hwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, any, s8, dhwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oidhw, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, dhwio, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oidhw, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, dhwio, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, oidhw, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, dhwio, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goihw, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, hwigo, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{f32, s8, 6}, {
        REG_SR(f32, any, s8, dhwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goidhw, s8, gOIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goidhw, s8, gOIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(f32, goidhw, s8, gOIdhw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    // bf16 -> s8
    {{bf16, s8, 3}, {
        REG_SR(bf16, any, s8, wio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oiw, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wio, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oiw, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wio, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oiw, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wio, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{bf16, s8, 4}, {
        REG_SR(bf16, any, s8, hwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, any, s8, wigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oihw, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwio, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwio, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oihw, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwio, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oihw, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goiw, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, wigo, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{bf16, s8, 5}, {
        REG_SR(bf16, any, s8, hwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, any, s8, dhwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oidhw, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, dhwio, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oidhw, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, dhwio, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, oidhw, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, dhwio, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goihw, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, hwigo, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{bf16, s8, 6}, {
        REG_SR(bf16, any, s8, dhwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goidhw, s8, gOIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goidhw, s8, gOIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(bf16, goidhw, s8, gOIdhw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    // s8 -> s8
    {{s8, s8, 3}, {
        REG_SR(s8, any, s8, wio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oiw, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wio, s8, OIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oiw, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wio, s8, OIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oiw, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wio, s8, OIw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{s8, s8, 4}, {
        REG_SR(s8, any, s8, hwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, any, s8, wigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, gOIw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, gOIw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, gOIw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwio, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oihw, s8, OIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwio, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oihw, s8, OIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwio, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oihw, s8, OIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goiw, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, wigo, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{s8, s8, 5}, {
        REG_SR(s8, any, s8, hwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, any, s8, dhwio, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, gOIhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, gOIhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, gOIhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oidhw, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, dhwio, s8, OIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oidhw, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, dhwio, s8, OIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, oidhw, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, dhwio, s8, OIdhw4o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goihw, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, hwigo, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
    {{s8, s8, 6}, {
        REG_SR(s8, any, s8, dhwigo, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goidhw, s8, gOIdhw4i16o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goidhw, s8, gOIdhw2i8o4i, fmt_order::keep, spec::conv_req_comp),
        REG_SR(s8, goidhw, s8, gOIdhw4o4i, fmt_order::keep, spec::conv_req_comp),

        nullptr,
    }},
};

// clang-format on

} // namespace

const rpd_create_f *cpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    const impl_list_map_t &impl_list
            = (dst_md->extra.flags
                      & (memory_extra_flags::compensation_conv_s8s8
                              | memory_extra_flags::
                                      compensation_conv_asymmetric_src))
            ? comp_s8s8_impl_list_map
            : regular_impl_list_map;

    reorder_impl_key_t key {
            src_md->data_type, dst_md->data_type, src_md->ndims};

    {
        const auto it = impl_list.find(key);
        if (it != impl_list.cend()) return it->second.data();
    }

    {
        key.ndims = 0;
        const auto it = impl_list.find(key);
        if (it != impl_list.cend()) return it->second.data();
    }

    {
        key.dst_dt = data_type::undef;
        const auto it = impl_list.find(key);
        if (it != impl_list.cend()) return it->second.data();
    }

    static const rpd_create_f empty_list[] = {nullptr};
    return empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
