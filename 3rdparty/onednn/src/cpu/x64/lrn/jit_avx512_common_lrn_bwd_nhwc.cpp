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
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_nhwc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_nhwc_t<
        d_type>::jit_avx512_common_lrn_kernel_bwd_nhwc_t(unsigned C,
        float alpha, float beta, int local_size, void *code_ptr,
        size_t code_size)
    : jit_avx512_common_lrn_kernel_bwd_t<d_type>(
            alpha, beta, local_size, code_ptr, code_size)
    , tmp_mask_prev_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), this->zdiffsrc_ + 2);
        return v;
    }()}
    , tmp_mask_next_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(),
                this->zdiffsrc_ + 2 + this->local_size_ / 2);
        return v;
    }()}
    , half_ls_ {(local_size - 1) / 2}
    , C_(C) {}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::generate() {

    const auto res = std::div(C_, 16);
    const auto &C_tail = res.rem;
    const auto &num_full_16c_blocks = res.quot;
    static const auto stack_space = zmm_size_ * 9;

    this->preamble();
    if (C_tail) reserve_stack_space(stack_space);
    this->set_up_ker_params();
    this->execute_compute_loop(num_full_16c_blocks, C_tail);
    if (C_tail) unreserve_stack_space(stack_space);

    this->postamble();
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::reserve_stack_space(
        std::size_t space) {
    const unsigned maxCounter = (space / zmm_size_) - 1;
    this->sub(rsp, space);
    this->uni_vpxor(zmm4, zmm4, zmm4);
    for (unsigned i = 0; i < maxCounter; ++i)
        this->vmovups(ptr[rsp + i * zmm_size_], zmm4);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::unreserve_stack_space(
        std::size_t space) {
    this->add(rsp, space);
}

template <data_type_t d_type>
int jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::get_stack_offset(
        const Reg64 reg, tail_mode tail_proc) {

    int stack_postion = 0;
    if (reg == this->diffdst_)
        stack_postion = 1;
    else if (reg == this->workspace1_)
        stack_postion = 3;
    else if (reg == this->workspace0_)
        stack_postion = 4;
    else if (reg == this->src_)
        stack_postion = 5;

    return zmm_size_
            * (stack_postion + (tail_proc == tail_mode::NextTail ? -1 : 0));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::load_data_to_stack(
        unsigned C_tail, across_version version, tail_mode tail_proc) {

    if (version != across_version::Single) {
        const int previousChunkOffset
                = tail_proc == tail_mode::NextTail ? 0 : -1 * this->vlen_;
        this->load_data(this->zreg(0, tmp_load_to_stack_idx_prev_),
                this->EVEX_compress_addr(this->diffdst_, previousChunkOffset));
        this->vmovups(
                this->EVEX_compress_addr(rsp,
                        get_stack_offset(this->diffdst_, tail_mode::NextTail)),
                this->zreg(0, tmp_load_to_stack_idx_prev_));

        this->load_data(this->zreg(0, tmp_load_to_stack_idx_prev_),
                this->EVEX_compress_addr(
                        this->workspace1_, previousChunkOffset));
        this->vmovups(this->EVEX_compress_addr(rsp,
                              get_stack_offset(
                                      this->workspace1_, tail_mode::NextTail)),
                this->zreg(0, tmp_load_to_stack_idx_prev_));
    }

    const int tail_src_mem_offset
            = tail_proc == tail_mode::NextTail ? this->vlen_ : 0;
    this->load_tail(C_tail, this->diffdst_, tail_src_mem_offset,
            get_stack_offset(this->diffdst_, tail_mode::CurrentTail),
            this->tmp_load_to_stack_idx_tail_);
    this->load_tail(C_tail, this->workspace0_, tail_src_mem_offset,
            get_stack_offset(this->workspace0_, tail_mode::CurrentTail),
            this->tmp_load_to_stack_idx_tail_);
    this->load_tail(C_tail, this->workspace1_, tail_src_mem_offset,
            get_stack_offset(this->workspace1_, tail_mode::CurrentTail),
            this->tmp_load_to_stack_idx_tail_);
    this->load_tail(C_tail, this->src_, tail_src_mem_offset,
            get_stack_offset(this->src_, tail_mode::CurrentTail),
            this->tmp_load_to_stack_idx_tail_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::set_up_ker_params() {
#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_bwd_t< \
                     d_type>::jit_args_bwd_t, \
            field)
    this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
    this->mov(this->diffdst_, ptr[this->param_ + GET_OFF(diff_dst)]);
    this->mov(this->workspace0_, ptr[this->param_ + GET_OFF(ws0)]);
    this->mov(this->workspace1_, ptr[this->param_ + GET_OFF(ws1)]);
    this->mov(this->diffsrc_, ptr[this->param_ + GET_OFF(diff_src)]);

    this->mov(this->mask_, ptr[this->param_ + GET_OFF(mask_ptr)]);
#undef GET_OFF

    this->mov(this->imm_addr64_, float2int(this->nalphabeta_));
    this->vmovq(this->xnalphabeta_, this->imm_addr64_);
    this->vbroadcastss(this->znalphabeta_, this->xnalphabeta_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::execute_compute_loop(
        unsigned num_full_16c_blocks, unsigned C_tail) {

    if ((num_full_16c_blocks == 1u && !C_tail)
            || (num_full_16c_blocks == 0u && C_tail)) {
        const auto tail_proc
                = C_tail ? tail_mode::CurrentTail : tail_mode::NoTail;
        compute_loop(across_version::Single, tail_proc, C_tail);
    } else {
        const int begin_end = C_tail ? 1 : 2;
        int middle_16_c_blocks = num_full_16c_blocks == 1
                ? 0
                : num_full_16c_blocks - begin_end;
        int LTAIL = 0;
        if (C_tail && middle_16_c_blocks) {
            middle_16_c_blocks -= 1;
            LTAIL = 1;
        }

        const int LSREST = middle_16_c_blocks % this->reg_block_;
        const int LS = middle_16_c_blocks - LSREST;

        if (LS > 0) this->mov(this->blockC_, LS);
        const auto first_tail_proc = num_full_16c_blocks == 1
                ? tail_mode::NextTail
                : tail_mode::NoTail;
        compute_loop(across_version::First, first_tail_proc, C_tail);
        increment_loop_params(this->vlen_);

        Label lrn_loop;

        if (LS > 0) {

            this->L(lrn_loop);
            {
                compute_loop(across_version::Middle, tail_mode::NoTail, C_tail,
                        this->reg_block_);
                increment_loop_params(this->reg_block_ * this->vlen_);
                this->sub(this->blockC_, this->reg_block_);
                this->cmp(this->blockC_, 0);
                this->jne(lrn_loop, this->T_NEAR);
            }
        }

        if (LSREST > 0) {
            compute_loop(
                    across_version::Middle, tail_mode::NoTail, C_tail, LSREST);
            increment_loop_params(LSREST * this->vlen_);
        }

        if (LTAIL) {
            compute_loop(
                    across_version::Middle, tail_mode::NextTail, C_tail, LTAIL);
            increment_loop_params(LTAIL * this->vlen_);
        }

        const auto last_tail_proc
                = C_tail ? tail_mode::CurrentTail : tail_mode::NoTail;
        compute_loop(across_version::Last, last_tail_proc, C_tail);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::compute_loop(
        across_version version, tail_mode tail_proc, unsigned C_tail,
        int loop_size_param) {

    if (tail_proc != tail_mode::NoTail)
        load_data_to_stack(C_tail, version, tail_proc);
    load_compute_data(version, tail_proc, loop_size_param);
    compute(loop_size_param, tail_proc);
    store_compute_data(loop_size_param, tail_proc, C_tail);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::compute(
        int loop_size, tail_mode tail_proc) {

    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->z_prev_[0])));
    assert(this->zsrc_ == this->z_prev_[0]);

    if (tail_proc == tail_mode::CurrentTail)
        this->load_data(this->zreg(0, this->zsrc_),
                this->EVEX_compress_addr(rsp,
                        get_stack_offset(this->src_, tail_mode::CurrentTail)),
                true);
    else
        IRB_LOOP(this->load_data(this->zreg(irb, this->zsrc_),
                this->EVEX_compress_addr(this->src_, irb * this->vlen_)));

    for (unsigned regIdx = 1; regIdx < this->z_prev_.size(); ++regIdx)
        IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
                this->zreg(irb, this->zdiffsrc_),
                this->zreg(irb, this->z_prev_[regIdx])));
    for (const auto reg : this->z_next_)
        IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
                this->zreg(irb, this->zdiffsrc_), this->zreg(irb, reg)));

    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsrc_),
            this->zreg(irb, this->zsrc_), this->znalphabeta_));

    if (tail_proc == tail_mode::CurrentTail) {
        this->load_data(this->zreg(0, this->zws0_),
                this->EVEX_compress_addr(rsp,
                        get_stack_offset(
                                this->workspace0_, tail_mode::CurrentTail)),
                true);
    } else {
        IRB_LOOP(this->load_data(this->zreg(irb, this->zws0_),
                this->EVEX_compress_addr(
                        this->workspace0_, irb * this->vlen_)));
    }

    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdiffdst_),
            this->zreg(irb, this->zdiffdst_), this->zreg(irb, this->zws0_)));
    IRB_LOOP(this->vfmadd213ps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zdiffdst_)));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::increment_loop_params(
        std::size_t offset) {
    this->add(this->src_, offset);
    this->add(this->diffsrc_, offset);
    this->add(this->diffdst_, offset);
    this->add(this->workspace0_, offset);
    this->add(this->workspace1_, offset);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::load_compute_data(
        across_version version, tail_mode tail_proc, int loop_size_param) {

    const int loop_size = loop_size_param;
    static constexpr int mask_shift = sizeof(int32_t);
    static constexpr int acc_size = d_type == bf16 ? 2 : 4;
    const auto load_shifted_padded_with_zeros
            = [this](int dstIdx, int srcIdx, int maskTmpIdx, int offset) {
                  this->uni_vpxor(this->zreg(0, dstIdx), this->zreg(0, dstIdx),
                          this->zreg(0, dstIdx));
                  this->load_data(this->zreg(0, maskTmpIdx),
                          this->EVEX_compress_addr(this->mask_, offset), true);
                  this->vpermt2ps(this->zreg(0, dstIdx),
                          this->zreg(0, maskTmpIdx), this->zreg(0, srcIdx));
              };

    const auto load_ws_diffdst = [&, this](int dstIdx, int offset,
                                         tail_mode tail_proc) {
        if (tail_proc == tail_mode::NoTail) {
            IRB_LOOP(this->load_data(this->zreg(irb, dstIdx),
                    this->EVEX_compress_addr(
                            this->workspace1_, (irb * this->vlen_) + offset)));
        } else
            this->load_data(this->zreg(0, dstIdx),
                    this->EVEX_compress_addr(this->rsp,
                            get_stack_offset(this->workspace1_, tail_proc)
                                    + offset),
                    true);

        if (d_type == bf16 || tail_proc != tail_mode::NoTail) {
            if (tail_proc == tail_mode::NoTail) {
                IRB_LOOP(this->load_data(this->zreg(irb, this->z_tmp_),
                        this->EVEX_compress_addr(
                                this->diffdst_, (irb * this->vlen_) + offset)));
            } else
                this->load_data(this->zreg(0, this->z_tmp_),
                        this->EVEX_compress_addr(this->rsp,
                                get_stack_offset(this->diffdst_, tail_proc)
                                        + offset),
                        true);

            IRB_LOOP(this->vmulps(this->zreg(irb, dstIdx),
                    this->zreg(irb, this->z_tmp_), this->zreg(irb, dstIdx)));
        } else {
            IRB_LOOP(this->vmulps(this->zreg(irb, dstIdx),
                    this->zreg(irb, dstIdx),
                    this->EVEX_compress_addr(
                            this->diffdst_, (irb * this->vlen_) + offset)));
        }
    };

    if (tail_proc == tail_mode::CurrentTail) {
        this->load_data(this->zreg(0, this->zdiffsrc_),
                this->EVEX_compress_addr(
                        rsp, get_stack_offset(this->workspace1_, tail_proc)),
                true);
        this->load_data(this->zreg(0, this->zdiffdst_),
                this->EVEX_compress_addr(
                        rsp, get_stack_offset(this->diffdst_, tail_proc)),
                true);
    } else {
        IRB_LOOP(this->load_data(this->zreg(irb, this->zdiffsrc_),
                this->EVEX_compress_addr(
                        this->workspace1_, irb * this->vlen_)));
        IRB_LOOP(this->load_data(this->zreg(irb, this->zdiffdst_),
                this->EVEX_compress_addr(this->diffdst_, irb * this->vlen_)));
    }

    IRB_LOOP(this->vmulps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffdst_),
            this->zreg(irb, this->zdiffsrc_)));

    int reg, mask, pos;
    std::vector<std::tuple<int, int, int>> prev_v;
    prev_v.reserve(this->half_ls_);
    for (int pos = 0; pos < this->half_ls_; ++pos) {
        prev_v.emplace_back(this->z_prev_[pos], this->tmp_mask_prev_[pos],
                this->half_ls_ - pos);
    };
    if (version == across_version::First || version == across_version::Single) {
        for (const auto &reg_mask_pos : prev_v) {
            std::tie(reg, mask, pos) = reg_mask_pos;
            load_shifted_padded_with_zeros(
                    reg, this->zdiffsrc_, mask, -1 * pos * mask_shift);
        }
    } else {
        for (const auto &reg_mask_pos : prev_v) {
            std::tie(reg, mask, pos) = reg_mask_pos;
            IRB_LOOP(load_ws_diffdst(reg, -1 * pos * acc_size,
                    tail_proc == tail_mode::CurrentTail ? tail_mode::CurrentTail
                                                        : tail_mode::NoTail));
        }
    }

    std::vector<std::tuple<int, int, int>> next_v;
    next_v.reserve(this->half_ls_);
    for (int pos = 0; pos < this->half_ls_; ++pos) {
        next_v.emplace_back(
                this->z_next_[pos], this->tmp_mask_next_[pos], pos + 1);
    }
    if (version == across_version::Last || version == across_version::Single) {
        for (const auto &reg_mask_pos : next_v) {
            std::tie(reg, mask, pos) = reg_mask_pos;
            load_shifted_padded_with_zeros(
                    reg, this->zdiffsrc_, mask, pos * mask_shift);
        }
    } else {
        for (const auto &reg_mask_pos : next_v) {
            std::tie(reg, mask, pos) = reg_mask_pos;
            IRB_LOOP(load_ws_diffdst(reg, pos * acc_size,
                    tail_proc == tail_mode::NextTail ? tail_mode::NextTail
                                                     : tail_mode::NoTail));
        }
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::store_compute_data(
        int loop_size_param, tail_mode tail_proc, unsigned C_tail) {
    const int loop_size = loop_size_param;

    if (tail_proc == tail_mode::CurrentTail) {
        this->store_tail(C_tail, this->zreg(0, this->zdiffsrc_), this->diffsrc_,
                0, 8 * zmm_size_, tmp_store_from_stack_idx_tail_);
    } else {
        Label unaligned_store, end_store;
        this->test(this->diffsrc_, this->vlen_ - 1);
        this->jnz(unaligned_store, this->T_NEAR);
        IRB_LOOP(this->store_data(true,
                this->EVEX_compress_addr(this->diffsrc_, irb * this->vlen_),
                this->zreg(irb, this->zdiffsrc_)));
        this->jmp(end_store, this->T_NEAR);
        this->L(unaligned_store);
        {
            IRB_LOOP(this->store_data(false,
                    this->EVEX_compress_addr(this->diffsrc_, irb * this->vlen_),
                    this->zreg(irb, this->zdiffsrc_)));
        }
        this->L(end_store);
    }
}

template class jit_avx512_common_lrn_kernel_bwd_nhwc_t<f32>;
template class jit_avx512_common_lrn_kernel_bwd_nhwc_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
