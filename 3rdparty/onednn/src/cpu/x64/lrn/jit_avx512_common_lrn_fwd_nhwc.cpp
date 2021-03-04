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
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_nhwc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_nhwc_t<
        d_type>::jit_avx512_common_lrn_kernel_fwd_nhwc_t(unsigned C,
        prop_kind_t prop_kind, float alpha, float beta, float k, int local_size,
        void *code_ptr, size_t code_size)
    : jit_avx512_common_lrn_kernel_fwd_t<d_type>(
            prop_kind, alpha, beta, k, local_size, code_ptr, code_size)
    , tmp_mask_prev_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), this->zc_ + 2);
        return v;
    }()}
    , tmp_mask_next_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), this->zc_ + 2 + this->local_size_ / 2);
        return v;
    }()}
    , half_ls_ {(local_size - 1) / 2}
    , C(C) {}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::generate() {

    const auto res = std::div(C, 16);
    const auto &C_tail = res.rem;
    const auto &num_full_16c_blocks = res.quot;
    static const auto stack_space = zmm_size * 3;

    this->preamble();
    if (C_tail) reserve_stack_space(stack_space);
    this->set_up_ker_params();
    this->execute_compute_loop(num_full_16c_blocks, C_tail);
    if (C_tail) unreserve_stack_space(stack_space);
    this->postamble();
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::reserve_stack_space(
        std::size_t space) {
    this->sub(rsp, space);
    this->uni_vpxor(zmm4, zmm4, zmm4);
    for (unsigned i = 0; i < 2u; ++i)
        this->vmovups(ptr[rsp + i * zmm_size], zmm4);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::unreserve_stack_space(
        std::size_t space) {
    this->add(rsp, space);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::set_up_ker_params() {

#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_fwd_t< \
                     d_type>::jit_args_fwd_t, \
            field)
    this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
    this->mov(this->dst_, ptr[this->param_ + GET_OFF(dst)]);
    if (this->pk_ != prop_kind::forward_inference) {
        this->mov(this->ws0_, ptr[this->param_ + GET_OFF(ws0)]);
        this->mov(this->ws1_, ptr[this->param_ + GET_OFF(ws1)]);
    }
    this->mov(this->mask_, ptr[this->param_ + GET_OFF(mask_ptr)]);
#undef GET_OFF

    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(this->xalpha_, this->imm_addr64_);
    this->vbroadcastss(this->zalpha_, this->xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(this->xk_, this->imm_addr64_);
    this->vbroadcastss(this->zk_, this->xk_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::execute_compute_loop(
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
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::increment_loop_params(
        std::size_t offset) {

    this->add(this->src_, offset);
    this->add(this->dst_, offset);
    if (this->pk_ != prop_kind::forward_inference) {
        this->add(this->ws0_, offset);
        this->add(this->ws1_, offset);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::compute_loop(
        across_version version, tail_mode tail_proc, unsigned C_tail,
        int loop_size_param) {

    if (tail_proc != tail_mode::NoTail)
        load_data_to_stack(C_tail, version, tail_proc);
    load_compute_data(version, tail_proc, loop_size_param);
    compute(loop_size_param);
    store_compute_data(loop_size_param, tail_proc, C_tail);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::load_data_to_stack(
        unsigned C_tail, across_version version, tail_mode tail_proc) {
    if (version != across_version::Single) {
        const int previousChunkOffset
                = tail_proc == tail_mode::NextTail ? 0 : -1 * this->vlen_;
        this->load_data(this->zreg(0, tmp_load_to_stack_idx_prev_),
                this->EVEX_compress_addr(this->src_, previousChunkOffset));
        this->vmovups(this->EVEX_compress_addr(rsp, 0),
                this->zreg(0, tmp_load_to_stack_idx_prev_));
    }

    const int tail_src_mem_offset
            = tail_proc == tail_mode::NextTail ? this->vlen_ : 0;
    static constexpr int tail_dst_stack_offset = zmm_size;
    this->load_tail(C_tail, this->src_, tail_src_mem_offset,
            tail_dst_stack_offset, this->tmp_load_to_stack_idx_tail_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::load_compute_data(
        across_version version, tail_mode tail_proc, int loop_size_param) {

    static constexpr int acc_size
            = d_type == bf16 ? sizeof(acc_data_bf16_t) : sizeof(acc_data_t);

    const int loop_size = loop_size_param;
    static constexpr int mask_shift = sizeof(int32_t);
    const auto load_shifted_padded_with_zeros
            = [&](int dstIdx, int srcIdx, int maskTmpIdx, int offset) {
                  this->uni_vpxor(this->zreg(0, dstIdx), this->zreg(0, dstIdx),
                          this->zreg(0, dstIdx));
                  this->load_data(this->zreg(0, maskTmpIdx),
                          this->EVEX_compress_addr(this->mask_, offset), true);
                  this->vpermt2ps(this->zreg(0, dstIdx),
                          this->zreg(0, maskTmpIdx), this->zreg(0, srcIdx));
              };

    if (tail_proc == tail_mode::CurrentTail) {
        this->load_data(this->zreg(0, this->zc_),
                this->EVEX_compress_addr(rsp, zmm_size), true);
    } else {
        IRB_LOOP(this->load_data(this->zreg(irb, this->zc_),
                this->EVEX_compress_addr(this->src_, irb * this->vlen_)));
    }

    struct entry_t {
        int reg, mask, pos;
        entry_t(int reg, int mask, int pos)
            : reg {reg}, mask {mask}, pos {pos} {}
    };
    std::vector<entry_t> prev_v;
    prev_v.reserve(this->half_ls_);
    for (int pos = 0; pos < this->half_ls_; ++pos) {
        prev_v.emplace_back(this->z_prev_[pos], this->tmp_mask_prev_[pos],
                this->half_ls_ - pos);
    }
    if (version == across_version::First || version == across_version::Single) {
        for (const auto &entry : prev_v) {
            load_shifted_padded_with_zeros(entry.reg, this->zc_, entry.mask,
                    -1 * entry.pos * mask_shift);
        }
    } else {
        if (tail_proc == tail_mode::CurrentTail) {
            for (const auto &entry : prev_v) {
                this->load_data(this->zreg(0, entry.reg),
                        this->EVEX_compress_addr(rsp,
                                zmm_size - 1 * entry.pos * sizeof(acc_data_t)),
                        true);
            }
        } else {
            for (const auto &entry : prev_v) {
                IRB_LOOP(this->load_data(this->zreg(irb, entry.reg),
                        this->EVEX_compress_addr(this->src_,
                                (irb * this->vlen_)
                                        - 1 * entry.pos * acc_size)));
            }
        }
    }

    std::vector<entry_t> next_v;
    next_v.reserve(this->half_ls_);
    for (int pos = 0; pos < this->half_ls_; ++pos) {
        next_v.emplace_back(
                this->z_next_[pos], this->tmp_mask_next_[pos], pos + 1);
    }
    if (version == across_version::Last || version == across_version::Single) {
        for (const auto &entry : next_v) {
            load_shifted_padded_with_zeros(
                    entry.reg, this->zc_, entry.mask, entry.pos * mask_shift);
        }
    } else {
        if (tail_proc == tail_mode::NextTail) {
            for (const auto &entry : next_v) {
                this->load_data(this->zreg(0, entry.reg),
                        this->EVEX_compress_addr(
                                rsp, entry.pos * sizeof(acc_data_t)),
                        true);
            }
        } else {
            for (const auto &entry : next_v) {
                IRB_LOOP(this->load_data(this->zreg(irb, entry.reg),
                        this->EVEX_compress_addr(this->src_,
                                (irb * this->vlen_) + entry.pos * acc_size)));
            }
        }
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::compute(
        int loop_size_param) {

    const int loop_size = loop_size_param;

    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->zc_), this->zreg(irb, this->zc_)));

    for (const auto reg : this->z_prev_)
        IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
                this->zreg(irb, reg), this->zreg(irb, reg)));
    for (const auto reg : this->z_next_)
        IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
                this->zreg(irb, reg), this->zreg(irb, reg)));

    IRB_LOOP(this->vfmadd132ps(
            this->zreg(irb, this->zsum_), this->zk_, this->zalpha_));
    IRB_LOOP(this->vmovaps(
            this->zreg(irb, this->zbase_), this->zreg(irb, this->zsum_)));

    if (this->beta_ != 1) {
        IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum2_),
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));
        IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum2_)));

        for (unsigned i = 0; i < 2; ++i)
            IRB_LOOP(this->vsqrtps(this->zreg(irb, this->zsum_),
                    this->zreg(irb, this->zsum_)));
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::store_compute_data(
        int loop_size_param, tail_mode tail_proc, unsigned C_tail) {

    const int loop_size = loop_size_param;
    static const int ytmp = this->zsum2_;

    if (this->pk_ != prop_kind::forward_inference) {
        // save intermediate results for lrn backward
        if (tail_proc == tail_mode::CurrentTail)
            this->store_tail(C_tail, this->zreg(0, this->zsum_), this->ws0_, 0,
                    2 * zmm_size, tmp_store_from_stack_idx_tail_);
        else
            IRB_LOOP(this->store_data(
                    this->EVEX_compress_addr(this->ws0_, irb * this->vlen_),
                    this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdst_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zsum_)));
    // storing to dst
    if (tail_proc == tail_mode::CurrentTail)
        this->store_tail(C_tail, this->zreg(0, this->zdst_), this->dst_, 0,
                2 * zmm_size, tmp_store_from_stack_idx_tail_);
    else
        IRB_LOOP(this->store_data(
                this->EVEX_compress_addr(this->dst_, irb * this->vlen_),
                this->zreg(irb, this->zdst_), this->yreg(irb, ytmp)));

    if (this->pk_ != prop_kind::forward_inference) {
        // calculate and save more intermediate results for lrn backward
        /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */
        IRB_LOOP(this->vdivps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zdst_), this->zreg(irb, this->zbase_)));

        if (tail_proc == tail_mode::CurrentTail)
            this->store_tail(C_tail, this->zreg(0, this->zsum_), this->ws1_, 0,
                    2 * zmm_size, tmp_store_from_stack_idx_tail_);
        else
            IRB_LOOP(this->store_data(
                    this->EVEX_compress_addr(this->ws1_, irb * this->vlen_),
                    this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
}

template class jit_avx512_common_lrn_kernel_fwd_nhwc_t<f32>;
template class jit_avx512_common_lrn_kernel_fwd_nhwc_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
