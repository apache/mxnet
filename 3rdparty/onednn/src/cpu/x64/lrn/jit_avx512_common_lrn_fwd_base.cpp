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

#include <numeric>
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_base.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

static constexpr int acc_size = sizeof(acc_data_t);
static constexpr int acc_bf_16_size = sizeof(acc_data_bf16_t);

template <data_type_t d_type>
const int32_t
        jit_avx512_common_lrn_kernel_fwd_t<d_type>::jit_args_fwd_t::mask[48]
        = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0};

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_t<d_type>::jit_args_fwd_t::jit_args_fwd_t()
    : src(nullptr)
    , dst(nullptr)
    , ws0(nullptr)
    , ws1(nullptr)
    , mask_ptr(&mask[16]) {}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<f32>::load_data(
        Xmm reg, const Address p, bool from_stack) {
    this->vmovups(reg, p);
}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<bf16>::load_data(
        Xmm reg, const Address p, bool from_stack) {
    if (!from_stack) {
        this->vpmovzxwd(reg, p);
        this->vpslld(reg, reg, 0x10);
    } else
        this->vmovups(reg, p);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_t<d_type>::load_tail(int tail_value,
        Reg64 src, int src_mem_offset, int dst_stack_offset,
        int tmp_load_to_stack_idx_tail) {

    static constexpr auto src_size = sizeof(data_t);

    const auto load_tail_simd = [&](Xmm tmp_reg, int vlen) {
        this->load_data(tmp_reg, this->EVEX_compress_addr(src, src_mem_offset));
        this->vmovups(this->EVEX_compress_addr(rsp, dst_stack_offset), tmp_reg);
        dst_stack_offset += vlen * acc_size;
        src_mem_offset += vlen * src_size;
        tail_value -= vlen;
    };

    if (tail_value >= 8)
        load_tail_simd(this->yreg(0, tmp_load_to_stack_idx_tail), 8);
    if (tail_value >= 4)
        load_tail_simd(this->xreg(0, tmp_load_to_stack_idx_tail), 4);

    for (int i = 0; i < tail_value; ++i) {
        if (d_type == bf16) {
            this->movzx(this->imm_addr64_, word[src + src_mem_offset]);
            this->vmovq(this->xreg(0, tmp_load_to_stack_idx_tail),
                    this->imm_addr64_);
            this->vpslld(this->xreg(0, tmp_load_to_stack_idx_tail),
                    this->xreg(0, tmp_load_to_stack_idx_tail), 0x10);
        } else
            this->vmovss(this->xreg(0, tmp_load_to_stack_idx_tail),
                    this->EVEX_compress_addr(src, src_mem_offset));

        this->vmovss(ptr[rsp + dst_stack_offset],
                this->xreg(0, tmp_load_to_stack_idx_tail));

        dst_stack_offset += acc_size;
        src_mem_offset += src_size;
    }
}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<bf16>::store_data(
        const Address addr, Zmm zr, Ymm yr) {
    if (emulateBfloat_)
        this->bf16_emu_->vcvtneps2bf16(yr, zr);
    else
        this->vcvtneps2bf16(yr, zr);

    this->vmovdqu16(addr, yr);
}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<f32>::store_data(
        const Address addr, Zmm zr, Ymm yr) {
    this->vmovups(addr, zr);
}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<f32>::store_tail(int tail_value,
        Zmm src, Reg64 dst, int dst_mem_offset, int tmp_stack_offset,
        int tmp_idx) {

    this->store_data(this->EVEX_compress_addr(rsp, tmp_stack_offset), src,
            this->yreg(0, tmp_idx));

    const auto store_tail_simd = [&](Xmm tmp_reg, int vlen) {
        this->vmovups(tmp_reg, this->EVEX_compress_addr(rsp, tmp_stack_offset));
        this->vmovups(this->EVEX_compress_addr(dst, dst_mem_offset), tmp_reg);
        tmp_stack_offset += vlen * acc_size;
        dst_mem_offset += vlen * acc_size;
        tail_value -= vlen;
    };

    if (tail_value >= 8) store_tail_simd(this->yreg(0, tmp_idx), 8);
    if (tail_value >= 4) store_tail_simd(this->xreg(0, tmp_idx), 4);

    for (int i = 0; i < tail_value;
            ++i, tmp_stack_offset += acc_size, dst_mem_offset += acc_size) {
        this->vmovss(this->xreg(0, tmp_idx),
                this->EVEX_compress_addr(rsp, tmp_stack_offset));
        this->vmovss(this->EVEX_compress_addr(dst, dst_mem_offset),
                this->xreg(0, tmp_idx));
    }
}

template <>
void jit_avx512_common_lrn_kernel_fwd_t<bf16>::store_tail(int tail_value,
        Zmm src, Reg64 dst, int dst_mem_offset, int tmp_stack_offset,
        int tmp_idx) {

    this->store_data(this->EVEX_compress_addr(rsp, tmp_stack_offset), src,
            this->yreg(0, tmp_idx));
    const auto res = std::div(tail_value, 4);

    for (int i = 0; i < res.quot; ++i, tmp_stack_offset += 4 * acc_bf_16_size,
             dst_mem_offset += 4 * acc_bf_16_size) {
        this->mov(this->imm_addr64_, qword[rsp + tmp_stack_offset]);
        this->mov(qword[dst + dst_mem_offset], this->imm_addr64_);
    }

    for (int i = 0; i < res.rem; ++i, tmp_stack_offset += acc_bf_16_size,
             dst_mem_offset += acc_bf_16_size) {
        this->mov(this->imm_addr16_, word[rsp + tmp_stack_offset]);
        this->mov(word[dst + dst_mem_offset], this->imm_addr16_);
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_t<d_type>::jit_avx512_common_lrn_kernel_fwd_t(
        prop_kind_t prop_kind, float alpha, float beta, float k, int local_size,
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , pk_(prop_kind)
    , alpha_(alpha)
    , beta_(beta)
    , k_(k)
    , local_size_ {local_size - !(local_size % 2)}
    , z_prev_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3);
        return v;
    }()}
    , z_next_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3 + this->local_size_ / 2);
        return v;
    }()}
    , zsum_ {std::max(local_size_ + 2, 6)}
    , emulateBfloat_(d_type == bf16 && !mayiuse(avx512_core_bf16))
    , regs_used_per_block_ {std::max(this->local_size_ + 2, 6)}
    , reg_block_ {[this]() {
        const int max_possible_reg_block
                = (emulateBfloat_ ? 26 : 30) / this->regs_used_per_block_;
        return mayiuse(avx512_core) ? max_possible_reg_block
                                    : std::min(max_possible_reg_block, 2);
    }()} {
    if (emulateBfloat_) {
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                bf16_emu_scratch_, bf16_emu_reserv_4_);
        bf16_emu_->init_vcvtneps2bf16();
    }
}

template <data_type_t d_type>
Zmm jit_avx512_common_lrn_kernel_fwd_t<d_type>::zreg(int irb, int i) const {
    return Zmm(irb * regs_used_per_block_ + i);
}

template <data_type_t d_type>
Ymm jit_avx512_common_lrn_kernel_fwd_t<d_type>::yreg(int irb, int i) const {
    return Ymm(irb * regs_used_per_block_ + i);
}

template <data_type_t d_type>
Xmm jit_avx512_common_lrn_kernel_fwd_t<d_type>::xreg(int irb, int i) const {
    return Xmm(irb * regs_used_per_block_ + i);
}

template class jit_avx512_common_lrn_kernel_fwd_t<f32>;
template class jit_avx512_common_lrn_kernel_fwd_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
