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
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_blocked.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>::
        jit_avx512_common_lrn_kernel_bwd_blocked_t(
                const struct nChw16c_across_t &J, float alpha, float beta,
                int local_size, int use_h_parallel, void *code_ptr,
                size_t code_size)
    : jit_avx512_common_lrn_kernel_bwd_t<d_type>(
            alpha, beta, local_size, code_ptr, code_size)
    , xmm_size_ {4 * sizeof(acc_data_t)}
    , zmm_size_ {64}
    , buffer_block_ {xmm_size_ + zmm_size_ + xmm_size_}
    , buffer_nest_offset_ {xmm_size_ + zmm_size_}
    , src_prev_offset_ {static_cast<int>(this->vlen_ - 4 * sizeof(data_t))}
    , use_h_parallelism_(use_h_parallel) {
    W_ = J.W;
    HW_ = J.H * J.W;
    version_ = J.version;
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>::generate() {

    this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
    this->mov(this->diffdst_, ptr[this->param_ + GET_OFF(diff_dst)]);
    this->mov(this->workspace0_, ptr[this->param_ + GET_OFF(ws0)]);
    this->mov(this->workspace1_, ptr[this->param_ + GET_OFF(ws1)]);
    this->mov(this->diffsrc_, ptr[this->param_ + GET_OFF(diff_src)]);
#undef GET_OFF

    int LSB = this->use_h_parallelism_ ? W_ : HW_;

    this->sub(this->rsp, this->reg_block_ * buffer_block_);
    this->mov(this->imm_addr64_, float2int(this->nalphabeta_));
    this->vmovq(this->xnalphabeta_, this->imm_addr64_);
    this->vbroadcastss(this->znalphabeta_, this->xnalphabeta_);

    if (version_ == across_version::First
            || version_ == across_version::Single) {
        this->uni_vpxor(xmm1, xmm1, xmm1);
        for (int irb = 0; irb < this->reg_block_; irb++) {
            this->vmovups(ptr[this->rsp + irb * buffer_block_], xmm1);
        }
    }
    if (version_ == across_version::Last
            || version_ == across_version::Single) {
        this->uni_vpxor(xmm1, xmm1, xmm1);
        for (int irb = 0; irb < this->reg_block_; irb++) {
            this->vmovups(
                    ptr[this->rsp + irb * buffer_block_ + buffer_nest_offset_],
                    xmm1);
        }
    }

    int LSREST = LSB % this->reg_block_;
    int LS = LSB - LSREST;

    Label lrn_loop;

    if (LS > 0) {
        this->mov(hw_, LS);

        this->L(lrn_loop);
        {
            compute_loop(this->reg_block_, 1, 1);

            this->add(this->src_, this->reg_block_ * this->vlen_);
            this->add(this->diffsrc_, this->reg_block_ * this->vlen_);
            this->add(this->diffdst_, this->reg_block_ * this->vlen_);
            this->add(this->workspace0_, this->reg_block_ * this->vlen_);
            this->add(this->workspace1_, this->reg_block_ * this->vlen_);

            for (int irb = 0; irb < this->reg_block_; irb++)
                this->dec(hw_);
            this->cmp(hw_, 0);
            this->jne(lrn_loop, this->T_NEAR);
        }
    }

    compute_loop(LSREST, 1, this->use_h_parallelism_ ? 0 : 1);

    this->add(this->rsp, this->reg_block_ * buffer_block_);
    this->postamble();
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>::compute_loop(
        int loop_size_param, int prefetchL1, int prefetchL2) {
    // loop_size - this->param_ for IRB_LOOP macro
    int loop_size = loop_size_param;
    const int prf0_offt = 1 * this->reg_block_;
    const int prf2_offt = 8 * this->reg_block_;

    // ---- prefetching -------------------------------------------
    if (version_ != across_version::First
            && version_ != across_version::Single) {
        if (prefetchL1)
            IRB_LOOP(this->mic_prefetcht0(ptr[this->workspace1_
                    + (irb + prf0_offt - 2 * HW_) * this->vlen_]));
        if (prefetchL1)
            IRB_LOOP(this->mic_prefetcht0(ptr[this->diffdst_
                    + (irb + prf0_offt - HW_) * this->vlen_]));
    }

    if (prefetchL1)
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->src_ + (irb + prf0_offt) * this->vlen_]));
    if (prefetchL2)
        IRB_LOOP(this->mic_prefetcht2(
                ptr[this->src_ + (irb + prf2_offt) * this->vlen_]));

    if (prefetchL1)
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->workspace1_ + (irb + prf0_offt) * this->vlen_]));

    if (prefetchL1)
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->diffdst_ + (irb + prf0_offt) * this->vlen_]));

    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        if (prefetchL1)
            IRB_LOOP(this->mic_prefetcht0(ptr[this->workspace1_
                    + (irb + prf0_offt + 2 * HW_) * this->vlen_]));
        if (prefetchL2)
            IRB_LOOP(this->mic_prefetcht2(ptr[this->workspace1_
                    + (irb + prf2_offt + 2 * HW_) * this->vlen_]));

        if (prefetchL1)
            IRB_LOOP(this->mic_prefetcht0(ptr[this->diffdst_
                    + (irb + prf0_offt + HW_) * this->vlen_]));
        if (prefetchL2)
            IRB_LOOP(this->mic_prefetcht2(ptr[this->diffdst_
                    + (irb + prf2_offt + HW_) * this->vlen_]));
    }
    if (prefetchL1)
        IRB_LOOP(this->mic_prefetcht0(
                ptr[this->workspace0_ + (irb + prf0_offt) * this->vlen_]));
    if (prefetchL2)
        IRB_LOOP(this->mic_prefetcht2(
                ptr[this->workspace0_ + (irb + prf2_offt) * this->vlen_]));
    // -----------------------------------------------------------

    if (loop_size_param == 0) return;

    if (version_ != across_version::First
            && version_ != across_version::Single) {
        IRB_LOOP(this->load_data(this->xreg(irb, xws1_prev_),
                ptr[this->workspace1_ + (irb - 2 * HW_) * this->vlen_
                        + src_prev_offset_]));
        IRB_LOOP(this->load_data(this->xreg(irb, xdiffdst_prev_),
                ptr[this->diffdst_ + (irb - HW_) * this->vlen_
                        + src_prev_offset_]));
        IRB_LOOP(this->vmulps(this->xreg(irb, xdiffdst_prev_),
                this->xreg(irb, xdiffdst_prev_), this->xreg(irb, xws1_prev_)));
    }

    IRB_LOOP(this->load_data(this->zreg(irb, zws1_),
            this->EVEX_compress_addr(this->workspace1_, irb * this->vlen_)));
    IRB_LOOP(this->load_data(this->zreg(irb, this->zdiffdst_),
            this->EVEX_compress_addr(this->diffdst_, irb * this->vlen_)));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffdst_), this->zreg(irb, zws1_)));

    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        IRB_LOOP(this->load_data(this->xreg(irb, xws1_next_),
                ptr[this->workspace1_ + (irb + 2 * HW_) * this->vlen_]));
        IRB_LOOP(this->load_data(this->xreg(irb, xdiffdst_next_),
                ptr[this->diffdst_ + (irb + HW_) * this->vlen_]));
        IRB_LOOP(this->vmulps(this->xreg(irb, xdiffdst_next_),
                this->xreg(irb, xdiffdst_next_), this->xreg(irb, xws1_next_)));
    }

    if (version_ != across_version::First
            && version_ != across_version::Single) {
        IRB_LOOP(this->vmovups(ptr[this->rsp + irb * buffer_block_],
                this->xreg(irb, xdiffdst_prev_)));
    }
    IRB_LOOP(this->vmovups(this->EVEX_compress_addr(
                                   this->rsp, irb * buffer_block_ + xmm_size_),
            this->zreg(irb, this->zdiffsrc_)));
    if (version_ != across_version::Last
            && version_ != across_version::Single) {
        IRB_LOOP(this->vmovups(
                ptr[this->rsp + irb * buffer_block_ + buffer_nest_offset_],
                this->xreg(irb, xdiffdst_next_)));
    }
    size_t acc_size = sizeof(acc_data_t);
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_prev_[0]),
            this->EVEX_compress_addr(this->rsp,
                    irb * buffer_block_ + xmm_size_ - 2 * acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_prev_[1]),
            this->EVEX_compress_addr(this->rsp,
                    irb * buffer_block_ + xmm_size_ - 1 * acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_next_[0]),
            this->EVEX_compress_addr(this->rsp,
                    irb * buffer_block_ + xmm_size_ + 1 * acc_size)));
    IRB_LOOP(this->vmovups(this->zreg(irb, this->z_next_[1]),
            this->EVEX_compress_addr(this->rsp,
                    irb * buffer_block_ + xmm_size_ + 2 * acc_size)));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->z_prev_[0])));
    assert(this->zsrc_ == this->z_prev_[0]);
    IRB_LOOP(this->load_data(this->zreg(irb, this->zsrc_),
            this->EVEX_compress_addr(this->src_, irb * this->vlen_)));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->z_prev_[1])));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->z_next_[0])));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->z_next_[1])));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsrc_),
            this->zreg(irb, this->zsrc_), this->znalphabeta_));

    IRB_LOOP(this->load_data(this->zreg(irb, this->zws0_),
            this->EVEX_compress_addr(this->workspace0_, irb * this->vlen_)));
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdiffdst_),
            this->zreg(irb, this->zdiffdst_), this->zreg(irb, this->zws0_)));
    IRB_LOOP(this->vfmadd213ps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zdiffdst_)));

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

template class jit_avx512_common_lrn_kernel_bwd_blocked_t<f32>;
template class jit_avx512_common_lrn_kernel_bwd_blocked_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
