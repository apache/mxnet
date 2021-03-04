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
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_blocked.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>::
        jit_avx512_common_lrn_kernel_fwd_blocked_t(
                const struct nChw16c_across_t &J, prop_kind_t prop_kind,
                int use_h_parallel, float alpha, float beta, float k,
                int local_size, void *code_ptr, size_t code_size)
    : jit_avx512_common_lrn_kernel_fwd_t<d_type>(
            prop_kind, alpha, beta, k, local_size, code_ptr, code_size)
    , use_h_parallelism_(use_h_parallel) {
    // some registers needed for conversion from bf16 to f32
    src_prev_offset_ = this->vlen_ - 4 * sizeof(data_t);
    version_ = J.version;
    W_ = J.W;
    HW_ = J.W * J.H;
    xmm_size_ = 4 * sizeof(acc_data_t);
    zmm_size_ = 64;
    buffer_block_ = xmm_size_ + zmm_size_ + xmm_size_;
    buffer_nest_offset_ = xmm_size_ + zmm_size_;
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>::generate() {
    this->preamble();

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
#undef GET_OFF

    int LSB = use_h_parallelism_ ? W_ : HW_;

    this->sub(t_, this->reg_block_ * buffer_block_);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(this->xalpha_, this->imm_addr64_);
    this->vbroadcastss(this->zalpha_, this->xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(this->xk_, this->imm_addr64_);
    this->vbroadcastss(this->zk_, this->xk_);

    if (version_ == across_version::First
            || version_ == across_version::Single) {
        this->uni_vpxor(xmm2, xmm2, xmm2);
        for (int irb = 0; irb < this->reg_block_; irb++) {
            this->vmovups(ptr[t_ + irb * buffer_block_], xmm2);
        }
    }
    if (version_ == across_version::Last
            || version_ == across_version::Single) {
        this->uni_vpxor(xmm2, xmm2, xmm2);
        for (int irb = 0; irb < this->reg_block_; irb++) {
            this->vmovups(
                    ptr[t_ + irb * buffer_block_ + buffer_nest_offset_], xmm2);
        }
    }

    const int LSREST = LSB % this->reg_block_;
    const int LS = LSB - LSREST;

    Label lrn_loop;

    if (LS > 0) {
        this->mov(hw_, LS);

        this->L(lrn_loop);
        {
            compute_loop(this->reg_block_);

            this->add(this->src_, this->reg_block_ * this->vlen_);
            this->add(this->dst_, this->reg_block_ * this->vlen_);
            if (this->pk_ != prop_kind::forward_inference) {
                this->add(this->ws0_, this->reg_block_ * this->vlen_);
                this->add(this->ws1_, this->reg_block_ * this->vlen_);
            }

            for (int irb = 0; irb < this->reg_block_; irb++)
                this->dec(hw_);
            this->cmp(hw_, 0);
            this->jne(lrn_loop, this->T_NEAR);
        }
    }

    compute_loop(LSREST);

    this->add(t_, this->reg_block_ * buffer_block_);
    this->postamble();
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_blocked_t<d_type>::compute_loop(
        int loop_size_param) {
    // loop_size - param for IRB_LOOP macro
    int loop_size = loop_size_param;

    const int prf0_offt = 1 * loop_size;
    const int prf2_offt = 8 * loop_size;

    if (version_ != across_version::First
            && version_ != across_version::Single) {
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->src_ + (irb + prf0_offt - HW_) * this->vlen_]));
        IRB_LOOP(this->mic_prefetcht2(
                ptr[this->src_ + (irb + prf2_offt - HW_) * this->vlen_]));
    }
    IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
            this->src_, (irb + prf0_offt) * this->vlen_)));
    IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
            this->src_, (irb + prf2_offt) * this->vlen_)));
    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->src_ + (irb + prf0_offt + HW_) * this->vlen_]));
        IRB_LOOP(this->mic_prefetcht2(
                ptr[this->src_ + (irb + prf2_offt + HW_) * this->vlen_]));
    }
    if (this->pk_ != prop_kind::forward_inference) {
        IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                this->ws0_, (irb + prf0_offt) * this->vlen_)));
        IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                this->ws0_, (irb + prf2_offt) * this->vlen_)));
    }
    IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
            this->dst_, (irb + prf0_offt) * this->vlen_)));
    IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
            this->dst_, (irb + prf2_offt) * this->vlen_)));
    if (this->pk_ != prop_kind::forward_inference) {
        IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                this->ws1_, (irb + prf0_offt) * this->vlen_)));
        IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                this->ws1_, (irb + prf2_offt) * this->vlen_)));
    }

    if (loop_size == 0) return;

    // --- loading source data to special buffer to form convenient data layout
    // for ACROSS lrn ---
    if (version_ != across_version::First
            && version_ != across_version::Single) {
        IRB_LOOP(this->load_data(this->xreg(irb, xsrc_prev_),
                ptr[this->src_ + (irb - HW_) * this->vlen_
                        + src_prev_offset_]));
    }
    IRB_LOOP(this->load_data(this->zreg(irb, this->zsrc_),
            this->EVEX_compress_addr(this->src_, irb * this->vlen_)));
    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        IRB_LOOP(this->load_data(this->xreg(irb, xsrc_next_),
                ptr[this->src_ + (irb + HW_) * this->vlen_]));
    }

    if (version_ != across_version::First
            && version_ != across_version::Single) {
        IRB_LOOP(this->vmovups(
                ptr[t_ + irb * buffer_block_], this->xreg(irb, xsrc_prev_)));
    }
    IRB_LOOP(this->vmovups(
            this->EVEX_compress_addr(t_, irb * buffer_block_ + xmm_size_),
            this->zreg(irb, this->zsrc_)));
    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        IRB_LOOP(this->vmovups(
                ptr[t_ + irb * buffer_block_ + buffer_nest_offset_],
                this->xreg(irb, xsrc_next_)));
    }

    // --- perform ACROSS lrn ---
    const size_t acc_size = sizeof(acc_data_t);
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_prev_[0]),
            this->EVEX_compress_addr(
                    t_, irb * buffer_block_ + xmm_size_ - 2 * acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_prev_[1]),
            this->EVEX_compress_addr(
                    t_, irb * buffer_block_ + xmm_size_ - acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_next_[0]),
            this->EVEX_compress_addr(
                    t_, irb * buffer_block_ + xmm_size_ + acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_next_[1]),
            this->EVEX_compress_addr(
                    t_, irb * buffer_block_ + xmm_size_ + 2 * acc_size)));

    assert(this->zc_ == this->zsrc_);
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->zc_), this->zreg(irb, this->zc_)));

    IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->z_prev_[0]),
            this->zreg(irb, this->z_prev_[0])));
    IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->z_prev_[1]),
            this->zreg(irb, this->z_prev_[1])));
    IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->z_next_[0]),
            this->zreg(irb, this->z_next_[0])));
    IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->z_next_[1]),
            this->zreg(irb, this->z_next_[1])));

    IRB_LOOP(this->vfmadd132ps(
            this->zreg(irb, this->zsum_), this->zk_, this->zalpha_));

    IRB_LOOP(this->vmovaps(
            this->zreg(irb, this->zbase_), this->zreg(irb, this->zsum_)));

    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum2_),
            this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));

    if (this->beta_ != 1) {
        IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum2_)));

        IRB_LOOP(this->vsqrtps(
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));
        IRB_LOOP(this->vsqrtps(
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));
    }

    const int ytmp = this->zsum2_; // temporary ymm for f32->bf16 conversion
    if (this->pk_ != prop_kind::forward_inference) {
        // save intermediate results for lrn backward
        IRB_LOOP(this->store_data(
                this->EVEX_compress_addr(this->ws0_, irb * this->vlen_),
                this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdst_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zsum_)));
    // storing to dst
    IRB_LOOP(this->store_data(
            this->EVEX_compress_addr(this->dst_, irb * this->vlen_),
            this->zreg(irb, this->zdst_), this->yreg(irb, ytmp)));
    if (this->pk_ != prop_kind::forward_inference) {
        // calculate and save more intermediate results for lrn backward
        /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */
        IRB_LOOP(this->vdivps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zdst_), this->zreg(irb, this->zbase_)));
        IRB_LOOP(this->store_data(
                this->EVEX_compress_addr(this->ws1_, irb * this->vlen_),
                this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
}

template class jit_avx512_common_lrn_kernel_fwd_blocked_t<f32>;
template class jit_avx512_common_lrn_kernel_fwd_blocked_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
