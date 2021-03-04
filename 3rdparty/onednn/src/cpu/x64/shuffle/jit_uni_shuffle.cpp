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

#include <assert.h>
#include <cmath>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace format_tag;

#define GET_OFF(field) offsetof(jit_shuffle_args_t, field)
struct jit_shuffle_args_t {
    jit_shuffle_args_t();
    const float *src;
    float *dst;
    const int32_t *input_off_ptr;
};

jit_shuffle_args_t::jit_shuffle_args_t()
    : src(nullptr), dst(nullptr), input_off_ptr(nullptr) {}

template <cpu_isa_t isa>
struct jit_uni_shuffle_base_kernel_t : public jit_generator {
    jit_uni_shuffle_base_kernel_t(const shuffle_pd_t *pd) : pd_(pd) {}

    void uni_pinsrd(Xmm xmm_reg, Reg64 load_reg, int data_size, int xmm_off) {
        vpinsrd(xmm_reg, xmm_reg, ptr[src_reg + load_reg * data_size], xmm_off);
    }

    void store(int dst_off, Xmm xmm_reg) {
        vmovups(ptr[dst_reg + dst_off], xmm_reg);
    }

protected:
    const shuffle_pd_t *pd_;

    const Reg64 src_reg = r9;
    const Reg64 dst_reg = r8;
    const Reg64 input_off_reg = r15;
};

template <>
void jit_uni_shuffle_base_kernel_t<sse41>::uni_pinsrd(
        Xmm xmm_reg, Reg64 load_reg, int data_size, int xmm_off) {
    pinsrd(xmm_reg, ptr[src_reg + load_reg * data_size], xmm_off);
}

template <>
void jit_uni_shuffle_base_kernel_t<sse41>::store(int dst_off, Xmm xmm_reg) {
    movups(ptr[dst_reg + dst_off], xmm_reg);
}

// jit kernels
namespace {

template <cpu_isa_t isa, int data_type_size, int group_size>
struct jit_uni_shuffle_kernel_t : public jit_uni_shuffle_base_kernel_t<isa> {};

template <cpu_isa_t isa>
struct jit_uni_shuffle_kernel_t<isa, 4, 3>
    : public jit_uni_shuffle_base_kernel_t<isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_kernel_t)

    jit_uni_shuffle_kernel_t(const shuffle_pd_t *pd)
        : jit_uni_shuffle_base_kernel_t<isa>(pd) {

        this->preamble();

        this->generate();

        this->postamble();
    }

private:
    void generate() override;
};

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa, 4, 3>::generate() {
    static constexpr int group_size = 3;
    static constexpr int data_type_size = 4;
    static constexpr int blk_size = 16;
    static constexpr int offset_data_type_size = 4;
    static constexpr int xmm_max_avx = cpu_isa_traits<avx>::n_vregs;
    // In case AVX512DQ is not available, due to usage of vpinsrd instruction,
    // xmm_max has to be limited to avx's xmm_max.
    static const int xmm_max = !mayiuse(avx512_core)
            ? xmm_max_avx
            : cpu_isa_traits<isa>::n_vregs;
    static constexpr int step_size = 4;
    static constexpr int unit_size = 4;
    const size_t C = this->pd_->C();
    const auto C_over_grps = utils::div_up(C, group_size);
    const auto stride = C_over_grps * data_type_size;

    const Reg32 load_registers_32[4]
            = {this->ebx, this->eax, this->edx, this->esi};
    const Reg64 load_registers_64[4]
            = {this->rbx, this->rax, this->rdx, this->rsi};

    for (int i = 0; i < 4; i++)
        this->xor_(load_registers_64[i], load_registers_64[i]);

    this->mov(this->input_off_reg,
            this->ptr[abi_param1 + GET_OFF(input_off_ptr)]);

    this->mov(this->src_reg, this->ptr[abi_param1 + GET_OFF(src)]);
    this->mov(this->dst_reg, this->ptr[abi_param1 + GET_OFF(dst)]);

    const auto SP = this->pd_->H() * this->pd_->W();

    std::vector<std::pair<size_t, int>> stores;
    auto xmm_id = 0;
    const int group_elems = C / group_size;
    const auto stride_mod = SP - 1;

    auto shuffle_one_by_one = [&](int elem, int gr, int num_elements) {
        const auto elem_blks = elem / blk_size;
        const auto current_4_elems = (elem - gr * group_elems) / unit_size;
        const auto tmp = std::div(current_4_elems, unit_size);
        const auto output_off = (tmp.quot * blk_size + tmp.rem * unit_size
                                        + elem_blks * stride_mod * blk_size)
                        * data_type_size
                + gr * stride;
        for (int s = 0; s < num_elements; s++) {
            this->mov(load_registers_32[0],
                    this->ptr[this->input_off_reg
                            + (elem + s) * offset_data_type_size]);
            this->mov(load_registers_32[1],
                    this->ptr[this->src_reg
                            + load_registers_64[0] * data_type_size]);
            const auto elem_blks_mod = (elem + s) / blk_size - elem_blks;
            this->mov(this->ptr[this->dst_reg + output_off + s * data_type_size
                              + elem_blks_mod * stride_mod * blk_size
                                      * data_type_size],
                    load_registers_32[1]);
        }
    };

    for (int gr = 0; gr < group_size; ++gr)
        for (int elem = gr * group_elems; elem < group_elems * (gr + 1);
                elem += step_size) {
            if (group_elems * (gr + 1) - elem < step_size) {
                // tail
                shuffle_one_by_one(elem, gr, group_elems * (gr + 1) - elem);
            } else if (elem / blk_size != (elem + step_size - 1) / blk_size) {
                // skip
                shuffle_one_by_one(elem, gr, step_size);
            } else {
                for (int i = 0; i < step_size; i++)
                    this->mov(load_registers_32[i],
                            this->ptr[this->input_off_reg
                                    + (elem + i) * offset_data_type_size]);
                for (int i = 0; i < step_size; i++)
                    this->uni_pinsrd(Xmm(xmm_id), load_registers_64[i],
                            data_type_size, i);
                const auto elem_blks = elem / blk_size;
                const auto current_4_elems
                        = (elem - gr * group_elems) / unit_size;
                const auto tmp = std::div(current_4_elems, unit_size);
                const auto output_off
                        = (tmp.quot * blk_size + tmp.rem * unit_size
                                  + elem_blks * stride_mod * blk_size)
                                * data_type_size
                        + gr * stride;

                stores.emplace_back(std::make_pair(output_off, xmm_id));
                xmm_id++;
            }
            const bool last_step = (gr + 1 == group_size
                    && elem + step_size >= group_elems * (gr + 1));
            if (xmm_id == xmm_max || last_step) {
                // store
                for (auto const &value : stores)
                    this->store(value.first, Xmm(value.second));
                stores.clear();
                xmm_id = 0;
            }
        }
}

template struct jit_uni_shuffle_kernel_t<sse41, 4, 3>;
template struct jit_uni_shuffle_kernel_t<avx, 4, 3>;
template struct jit_uni_shuffle_kernel_t<avx512_common, 4, 3>;

#undef GET_OFF
} // namespace

template <cpu_isa_t isa, int data_type_size>
status_t jit_uni_shuffle_t<isa, data_type_size>::init(engine_t *engine) {
    kernel_ = utils::make_unique<
            jit_uni_shuffle_kernel_t<isa, data_type_size, 3>>(pd());
    CHECK(kernel_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa, int data_type_size>
inline jit_uni_shuffle_t<isa, data_type_size>::jit_uni_shuffle_t(
        const pd_t *apd)
    : primitive_t(apd) {
    const int axis_size = pd()->axis_size();
    const int group_size = pd()->group_size();
    const int transpose_row
            = pd()->is_fwd() ? group_size : axis_size / group_size;
    const int transpose_col
            = pd()->is_fwd() ? axis_size / group_size : group_size;
    std::vector<int> rev_transposed_(axis_size);

    parallel_nd(transpose_col, transpose_row, [&](int i, int j) {
        rev_transposed_[j * transpose_col + i] = i * transpose_row + j;
    });

    const int C = pd()->C();
    const auto tmp = std::div(C, blk_size);
    const int C_16 = tmp.rem > 0 ? tmp.quot + 1 : tmp.quot;
    const int SP = pd()->H() * pd()->W();
    input_off_
            = (int *)malloc(C * sizeof(int), platform::get_cache_line_size());

    parallel_nd(C_16, [&](int c_16) {
        PRAGMA_OMP_SIMD()
        for (int cc = 0; cc < nstl::min(blk_size, C - c_16 * blk_size); ++cc) {
            const int &input_c = rev_transposed_[c_16 * blk_size + cc];
            input_off_[c_16 * blk_size + cc]
                    = input_c / blk_size * SP * blk_size + input_c % blk_size;
        }
    });
}

template <cpu_isa_t isa, int data_type_size>
constexpr int jit_uni_shuffle_t<isa, data_type_size>::blk_size;

template <cpu_isa_t isa, int data_type_size>
jit_uni_shuffle_t<isa, data_type_size>::~jit_uni_shuffle_t() {
    free(this->input_off_);
}

template <cpu_isa_t isa, int data_type_size>
template <dnnl_format_tag_t tag>
void jit_uni_shuffle_t<isa, data_type_size>::execute_(
        const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;

    const memory_desc_wrapper data_d(pd()->data_md());

    const auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const data_t *, i_arg);
    auto output = CTX_OUT_MEM(data_t *, o_arg);

    const int MB = pd()->MB();
    const int SP = pd()->H() * pd()->W();
    const size_t stride_mb = data_d.blocking_desc().strides[0];

    parallel_nd(MB, SP, [&](int mb, int sp) {
        const size_t group_size = pd()->group_size();

        const auto c_over_blks = 0;
        const auto grps_over_blks
                = std::div(c_over_blks * group_size, blk_size);
        const auto out_off = mb * stride_mb
                + grps_over_blks.quot * SP * blk_size + grps_over_blks.rem
                + sp * blk_size;

        const int in_off
                = mb * stride_mb + c_over_blks * SP * blk_size + sp * blk_size;

        jit_shuffle_args_t args;
        args.src = input + in_off;
        args.dst = output + out_off;

        args.input_off_ptr = this->input_off_;

        (*kernel_)(&args);
    });
}

template struct jit_uni_shuffle_t<sse41, 4>;
template struct jit_uni_shuffle_t<avx, 4>;
template struct jit_uni_shuffle_t<avx512_common, 4>;

template void jit_uni_shuffle_t<sse41, 4>::execute_<nChw16c>(
        const exec_ctx_t &ctx) const;
template void jit_uni_shuffle_t<avx, 4>::execute_<nChw16c>(
        const exec_ctx_t &ctx) const;
template void jit_uni_shuffle_t<avx512_common, 4>::execute_<nChw16c>(
        const exec_ctx_t &ctx) const;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
