/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include <array>
#include <cmath>
#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/lrn/jit_uni_lrn_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;

#define IRB_LOOP(statement) \
    if (1 == reg_block) { \
        const int irb_off = 0; \
        const int irb = this->reg_block_idx_ % vsum.size(); \
        statement; \
        MAYBE_UNUSED(irb_off); \
    } else { \
        for (int irb = 0; irb < reg_block; irb++) { \
            const int irb_off = irb * this->single_pixel_offset_; \
            statement; \
            MAYBE_UNUSED(irb_off); \
        } \
    }

using namespace Xbyak;

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::jit_uni_lrn_kernel_t(
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , emulate_bfloat_(isa == avx512_common
              && d_type == dnnl::impl::data_type::bf16
              && !mayiuse(avx512_core_bf16))
    , bf16_emu_(
              emulate_bfloat_ ? utils::make_unique<bf16_emulation_t>(this,
                      bf16_emu_reserv_1_, bf16_emu_reserv_2_,
                      bf16_emu_reserv_3_, bf16_emu_scratch_, bf16_emu_reserv_4_)
                              : nullptr) {

    if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::jit_uni_lrn_kernel_t(
        const within_config_t &config, void *code_ptr, size_t code_size)
    : jit_uni_lrn_kernel_t(code_ptr, code_size) {
    if (config.dat_tag == nhwc)
        single_pixel_offset_
                = config.C * sizeof(typename prec_traits<d_type>::type);
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::~jit_uni_lrn_kernel_t() = default;

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::within_loop(
        const within_config_t &config, int max_reg_blocks, prop_kind_t pk) {
    const auto derived_ptr = static_cast<Derived<isa, d_type> *>(this);

    const int lower_bound = (config.size - 1) / 2,
              upper_bound = config.size - lower_bound - 1;

    int pixel_count = 0;

    for (int i = 0; i < lower_bound; ++i) {
        pixel_count = 0;
        for (int j = 0; j < lower_bound; ++j)
            derived_ptr->within_body(-i, upper_bound, -j, upper_bound, config.W,
                    pk, 1, pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks, -i,
                upper_bound, -lower_bound, upper_bound, config.W, pk);

        pixel_count = 0;
        for (int j = config.W - upper_bound; j < config.W; ++j)
            derived_ptr->within_body(-i, upper_bound, -lower_bound,
                    config.W - 1 - j, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);
    }

    this->mov(h_, config.H - config.size + 1);
    Label lrn_loop_h;
    this->L(lrn_loop_h);
    pixel_count = 0;
    for (int j = 0; j < lower_bound; ++j)
        derived_ptr->within_body(-lower_bound, upper_bound, -j, upper_bound,
                config.W, pk, 1, pixel_count++ * this->single_pixel_offset_);
    derived_ptr->move_data_pointers(pixel_count, pk);

    within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks,
            -lower_bound, upper_bound, -lower_bound, upper_bound, config.W, pk);

    pixel_count = 0;
    for (int j = config.W - upper_bound; j < config.W; ++j)
        derived_ptr->within_body(-lower_bound, upper_bound, -lower_bound,
                config.W - 1 - j, config.W, pk, 1,
                pixel_count++ * this->single_pixel_offset_);
    derived_ptr->move_data_pointers(pixel_count, pk);

    this->dec(h_);
    this->cmp(h_, 0);
    this->jne(lrn_loop_h, this->T_NEAR);

    for (int i = config.H - upper_bound; i < config.H; ++i) {
        pixel_count = 0;
        for (int j = 0; j < lower_bound; ++j)
            derived_ptr->within_body(-lower_bound, config.H - 1 - i, -j,
                    upper_bound, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks,
                -lower_bound, config.H - 1 - i, -lower_bound, upper_bound,
                config.W, pk);

        pixel_count = 0;
        for (int j = config.W - upper_bound; j < config.W; ++j)
            derived_ptr->within_body(-lower_bound, config.H - 1 - i,
                    -lower_bound, config.W - 1 - j, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);
    }
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::within_body_reg_blocked(
        int loop_count, int max_reg_blocks, int hoff, int Hoff, int woff,
        int Woff, int stride, prop_kind_t pk) {

    const auto derived_ptr = static_cast<Derived<isa, d_type> *>(this);
    Label reg_block_compute_loop;

    const auto res = std::div(loop_count, max_reg_blocks);
    if (res.quot) {
        this->mov(this->w_, res.quot);
        this->L(reg_block_compute_loop);
        derived_ptr->within_body(
                hoff, Hoff, woff, Woff, stride, pk, max_reg_blocks, 0);
        derived_ptr->move_data_pointers(max_reg_blocks, pk);
        this->dec(this->w_);
        this->cmp(this->w_, 0);
        this->jne(reg_block_compute_loop, this->T_NEAR);
    }
    if (res.rem) {
        derived_ptr->within_body(
                hoff, Hoff, woff, Woff, stride, pk, res.rem, 0);
        derived_ptr->move_data_pointers(res.rem, pk);
    }
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::load_data(
        const Vmm &reg, const Xbyak::Address &p) {
    this->uni_vmovups(reg, p);
}

template <typename Gen, typename Reg, typename Addr>
void load_bf16_data(Gen generator, const Reg &reg, const Addr &p) {
    generator->vpmovzxwd(reg, p);
    generator->vpslld(reg, reg, 0x10);
}

template <>
void jit_uni_lrn_kernel_t<jit_uni_lrn_fwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>>::load_data(const Vmm &reg,
        const Xbyak::Address &p) {
    load_bf16_data(this, reg, p);
}

template <>
void jit_uni_lrn_kernel_t<jit_uni_lrn_bwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>>::load_data(const Vmm &reg,
        const Xbyak::Address &p) {
    load_bf16_data(this, reg, p);
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::store_data(
        const Xbyak::Address &addr, const Vmm &reg) {
    this->uni_vmovups(addr, reg);
}

template <typename Gen, typename Bf16Emu>
void store_bf16_data(
        Gen generator, Bf16Emu emu, const Xbyak::Address &addr, const Zmm &zr) {
    const Ymm yr = Ymm(zr.getIdx());
    if (mayiuse(avx512_core_bf16))
        generator->vcvtneps2bf16(yr, zr);
    else
        emu->vcvtneps2bf16(yr, zr);
    generator->vmovdqu16(addr, yr);
}

template <>
void jit_uni_lrn_kernel_t<jit_uni_lrn_fwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>>::store_data(const Xbyak::Address &addr,
        const Zmm &zr) {
    store_bf16_data(this, bf16_emu_.get(), addr, zr);
}

template <>
void jit_uni_lrn_kernel_t<jit_uni_lrn_bwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>>::store_data(const Xbyak::Address &addr,
        const Zmm &zr) {
    store_bf16_data(this, bf16_emu_.get(), addr, zr);
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::load_constant(
        float constant, const Vmm &v_constant, const Xbyak::Xmm &x_constant) {
    this->mov(this->imm_addr64_, float2int(constant));
    this->uni_vmovq(x_constant, this->imm_addr64_);
    this->vbroadcastss(v_constant, x_constant);
}

template <>
void jit_uni_lrn_kernel_t<jit_uni_lrn_fwd_kernel_t<sse41,
        dnnl::impl::data_type::f32>>::load_constant(float constant,
        const Vmm &v_constant, const Xbyak::Xmm &x_constant) {
    this->mov(this->imm_addr64_, float2int(constant));
    this->uni_vmovq(x_constant, this->imm_addr64_);
    this->shufps(x_constant, x_constant, 0);
}

//////////////////////////////////////////////////////////////////////////////
// forward kernel
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::within_body(int hoff, int Hoff,
        int woff, int Woff, int stride, prop_kind_t pk, const int reg_block,
        int pixel_offset) {

    static const std::array<Vmm, 3> vsum {{Vmm(2), Vmm(11), Vmm(20)}};
    static const std::array<Vmm, 3> vsum2 {{Vmm(3), Vmm(12), Vmm(21)}};
    static const std::array<Vmm, 3> vdst {{Vmm(4), Vmm(13), Vmm(22)}};
    static const std::array<std::array<Vmm, 6u>, 3u> vtmp {
            {{{Vmm(5), Vmm(6), Vmm(7), Vmm(8), Vmm(9), Vmm(14)}},
                    {{Vmm(18), Vmm(15), Vmm(16), Vmm(17), Vmm(29), Vmm(30)}},
                    {{Vmm(23), Vmm(24), Vmm(25), Vmm(26), Vmm(28), Vmm(31)}}}};
    static const std::array<Vmm, 3> vscratch = {{Vmm(10), Vmm(19), Vmm(27)}};
    static const std::size_t used_tmp_regs
            = this->emulate_bfloat_ ? vtmp[0].size() - 2 : vtmp[0].size();

    IRB_LOOP(this->uni_vxorps(vsum[irb], vsum[irb], vsum[irb]));
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            if (i == 0 && j == 0) {
                IRB_LOOP(this->load_data(
                        vdst[irb], this->ptr[src_ + pixel_offset + irb_off]));
                IRB_LOOP(this->vfmadd231ps(vsum[irb], vdst[irb], vdst[irb]));
            } else {
                const auto idx = this->tempIdx_ % used_tmp_regs;
                IRB_LOOP(this->load_data(vtmp[irb][idx],
                        this->ptr[(src_ + pixel_offset + irb_off)
                                + (i * stride + j)
                                        * this->single_pixel_offset_]));
                IRB_LOOP(this->vfmadd231ps(
                        vsum[irb], vtmp[irb][idx], vtmp[irb][idx]));
                ++(this->tempIdx_);
            }
        }
    }

    this->tempIdx_ = this->tempIdx_ % used_tmp_regs;

    IRB_LOOP(this->vfmadd132ps(
            vsum[irb], vk_, valpha_)); // ysum <- ysum*valpha_+yk_
    IRB_LOOP(this->vmovaps(vscratch[irb], vsum[irb]));

    IRB_LOOP(this->vmulps(vsum2[irb], vsum[irb], vsum[irb]));
    IRB_LOOP(this->vmulps(
            vsum[irb], vsum[irb], vsum2[irb])); // ysum = (ysum*valpha_+yk_)^3;
    IRB_LOOP(this->vsqrtps(vsum[irb], vsum[irb]));
    IRB_LOOP(this->vsqrtps(
            vsum[irb], vsum[irb])); // ysum = (ysum*valpha_+yk_)^0.75
    IRB_LOOP(this->vdivps(
            vdst[irb], vdst[irb], vsum[irb])); // ydst <- ydst / ysum

    if (pk_ != prop_kind::forward_inference) {
        IRB_LOOP(this->store_data(
                this->ptr[scratch_ + pixel_offset + irb_off], vsum[irb]));
        IRB_LOOP(this->vdivps(vscratch[irb], vdst[irb], vscratch[irb]));
        IRB_LOOP(this->store_data(
                this->ptr[bwd_intermediate_res_ + pixel_offset + irb_off],
                vscratch[irb]));
    }

    IRB_LOOP(this->store_data(
            this->ptr[dst_ + pixel_offset + irb_off], vdst[irb]));

    if (isa == avx512_common)
        this->reg_block_idx_ = (this->reg_block_idx_ % vsum.size()) + 1;
}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::within_body(
        int hoff, int Hoff, int woff, int Woff, int stride, prop_kind_t pk,
        int reg_block, int pixel_offset) {

    const Xbyak::Xmm &xtmp_lo = this->xmm2;
    const Xbyak::Xmm &xtmp_hi = this->xmm3;
    const Xbyak::Xmm &xsum_lo = this->xmm4;
    const Xbyak::Xmm &xsum_hi = this->xmm5;
    const Xbyak::Xmm &xdst_lo = this->xmm6;
    const Xbyak::Xmm &xdst_hi = this->xmm7;
    const Xbyak::Xmm &xsum2_lo = this->xmm8;
    const Xbyak::Xmm &xsum2_hi = this->xmm9;

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            if (i == 0 && j == 0) {
                movups(xdst_lo, ptr[src_ + pixel_offset]);
                movups(xdst_hi, ptr[src_ + pixel_offset + 4 * sizeof(float)]);
                mulps(xdst_lo, xdst_lo);
                mulps(xdst_hi, xdst_hi);
                addps(xsum_lo, xdst_lo);
                addps(xsum_hi, xdst_hi);
            } else {
                movups(xtmp_lo,
                        ptr[src_ + pixel_offset
                                + (i * stride + j) * single_pixel_offset_]);
                movups(xtmp_hi,
                        ptr[src_ + pixel_offset
                                + (i * stride + j) * single_pixel_offset_
                                + 4 * sizeof(float)]);
                this->mulps(xtmp_lo, xtmp_lo);
                this->mulps(xtmp_hi, xtmp_hi);
                this->addps(xsum_lo, xtmp_lo);
                this->addps(xsum_hi, xtmp_hi);
            }
        }
    }
    this->mulps(xsum_lo, xalpha_);
    this->mulps(xsum_hi, xalpha_);
    this->addps(xsum_lo, xk_);
    this->addps(xsum_hi, xk_); // xsum <- xsum*xalpha_+xk_
    this->movaps(xtmp_lo, xsum_lo);
    this->movaps(xtmp_hi, xsum_hi);
    if (pk_ != prop_kind::forward_inference) {
        this->movups(this->ptr[scratch_ + pixel_offset], xtmp_lo);
        this->movups(this->ptr[scratch_ + pixel_offset + 4 * sizeof(float)],
                xtmp_hi);
    }
    this->movaps(xsum2_lo, xsum_lo);
    this->movaps(xsum2_hi, xsum_hi);
    this->mulps(xsum2_lo, xsum_lo);
    this->mulps(xsum2_hi, xsum_hi);
    this->mulps(xsum_lo, xsum2_lo);
    this->mulps(xsum_hi, xsum2_hi); // xsum = (xsum*xalpha_+xk_)^3;

    this->sqrtps(xsum_lo, xsum_lo);
    this->sqrtps(xsum_hi, xsum_hi);
    this->sqrtps(xsum_lo, xsum_lo);
    this->sqrtps(xsum_hi, xsum_hi); // xsum = (xsum*xalpha_+xk_)^0.75

    this->movups(xdst_lo, this->ptr[src_ + pixel_offset]);
    this->movups(xdst_hi, this->ptr[src_ + pixel_offset + 4 * sizeof(float)]);
    this->divps(xdst_lo, xsum_lo);
    this->divps(xdst_hi, xsum_hi); // xdst <- xdst / xsum

    this->movups(this->ptr[dst_ + pixel_offset], xdst_lo);
    this->movups(this->ptr[dst_ + pixel_offset + 4 * sizeof(float)], xdst_hi);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::move_data_pointers(
        int pixel_count, prop_kind_t pk) {

    const int pixel_offset = this->single_pixel_offset_ * pixel_count;
    this->add(src_, pixel_offset);
    this->add(dst_, pixel_offset);
    if (pk_ != prop_kind::forward_inference) {
        this->add(scratch_, pixel_offset);
        this->add(bwd_intermediate_res_, pixel_offset);
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const within_config_t &config, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : Base(config, code_ptr, code_size)
    , config_(lrn_config_t::within_config)
    , within_config_(config)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(
        const within_config_t &config) {
    this->preamble();

#define GET_OFF(field) offsetof(jit_args_fwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(dst_, this->ptr[this->param1 + GET_OFF(dst)]);
    if (pk_ != prop_kind::forward_inference) {
        this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
        this->mov(bwd_intermediate_res_,
                this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    }
#undef GET_OFF

    this->load_constant(alpha_, valpha_, xalpha_);
    this->load_constant(k_, vk_, xk_);

    static const int max_reg_blocks = isa == avx512_common ? 3 : 1;
    this->within_loop(config, max_reg_blocks, pk_);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const struct nchw8c_across_t &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nchw8c_across)
    , nchw8c_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nchw8c_across_t &J) {
    const Xbyak::Reg64 &t = this->rsp;
    const Xbyak::Reg64 &hw = this->r9;
    const Xbyak::Xmm &xsrc_prev = this->xmm2;
    const Xbyak::Ymm &ysrc = this->ymm3;
    const Xbyak::Ymm &yc = this->ymm3;
    const Xbyak::Xmm &xsrc_next = this->xmm4;
    const Xbyak::Ymm &ya = this->ymm5;
    const Xbyak::Ymm &yb = this->ymm6;
    const Xbyak::Ymm &yd = this->ymm7;
    const Xbyak::Ymm &ye = this->ymm8;
    const Xbyak::Ymm &ysum = this->ymm9;
    const Xbyak::Ymm &ysum2 = this->ymm10;
    const Xbyak::Ymm &ydst = this->ymm11;
    const Xbyak::Ymm &ybase = this->ymm12;

    this->preamble();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);
    this->sub(t, 64);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    if (J.version == -1) {
        this->vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        this->vmovups(this->ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1) {
        this->vxorps(xsrc_next, xsrc_next, xsrc_next);
        this->vmovups(this->ptr[t + 48], xsrc_next);
    }

    this->mov(hw, J.H * J.W);

    Label lrn_loop;
    this->L(lrn_loop);

    if (J.version != -1)
        this->vmovups(xsrc_prev, this->ptr[src_ - J.H * J.W * 32 + 16]);
    this->vmovups(ysrc, this->ptr[src_]);
    if (J.version != +1)
        this->vmovups(xsrc_next, this->ptr[src_ + J.H * J.W * 32]);

    if (J.version != -1) this->vmovups(this->ptr[t + 0], xsrc_prev);
    this->vmovups(this->ptr[t + 16], ysrc);
    if (J.version != +1) this->vmovups(this->ptr[t + 48], xsrc_next);

    this->vmovups(ya, this->ptr[t + 16 - 8]);
    this->vmovups(yb, this->ptr[t + 16 - 4]);
    this->vmovups(yd, this->ptr[t + 16 + 4]);
    this->vmovups(ye, this->ptr[t + 16 + 8]);
    this->vmulps(ysum, yc, yc);
    this->vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
    this->vfmadd231ps(ysum, yb, yb);
    this->vfmadd231ps(ysum, yd, yd);
    this->vfmadd231ps(ysum, ye, ye);
    this->vfmadd132ps(ysum, yk_, valpha_); // ysum <- ysum*valpha_+yk_

    this->vmovaps(ybase, ysum);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ysum2, ysum, ysum);
    this->vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
    this->vsqrtps(ysum, ysum);
    this->vsqrtps(ysum, ysum); // ysum = ybase^0.75
    this->vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
    this->vmovups(this->ptr[dst_], ydst);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, 32);
    this->dec(hw);
    this->cmp(hw, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->add(t, 64);
    this->postamble();
}

template <>
jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::
        jit_uni_lrn_fwd_kernel_t(const struct nchw8c_across_t &J, float A,
                float K, prop_kind_t pk, void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nchw8c_across)
    , nchw8c_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::generate(
        const nchw8c_across_t &J) {

    const Xbyak::Reg64 &t = this->rsp;
    const Xbyak::Reg64 &hw = this->r9;
    const Xbyak::Xmm &xsrc_lo = this->xmm2;
    const Xbyak::Xmm &xsrc_hi = this->xmm3;
    const Xbyak::Xmm &xc_lo = this->xmm4;
    const Xbyak::Xmm &xc_hi = this->xmm5;
    const Xbyak::Xmm &xsum_lo = xc_lo;
    const Xbyak::Xmm &xsum_hi = xc_hi;
    const Xbyak::Xmm &xsrc_prev = this->xmm6;
    const Xbyak::Xmm &xsrc_next = this->xmm7;
    const Xbyak::Xmm &xa_lo = this->xmm8;
    const Xbyak::Xmm &xa_hi = this->xmm9;
    const Xbyak::Xmm &xb_lo = this->xmm10;
    const Xbyak::Xmm &xb_hi = this->xmm11;
    const Xbyak::Xmm &xd_lo = this->xmm12;
    const Xbyak::Xmm &xd_hi = this->xmm13;
    const Xbyak::Xmm &xe_lo = this->xmm14;
    const Xbyak::Xmm &xe_hi = this->xmm15;
    const Xbyak::Xmm &xbase_lo = this->xmm14;
    const Xbyak::Xmm &xbase_hi = this->xmm15;

    this->preamble();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);
    this->sub(t, 64);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(xalpha_, this->imm_addr64_);
    this->shufps(xalpha_, xalpha_, 0);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(xk_, this->imm_addr64_);
    this->shufps(xk_, xk_, 0);

    if (J.version == -1) {
        this->xorps(xsrc_prev, xsrc_prev);
        this->movups(this->ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1) {
        this->xorps(xsrc_next, xsrc_next);
        this->movups(this->ptr[t + 48], xsrc_next);
    }

    this->mov(hw, J.H * J.W);
    Label lrn_loop;
    L(lrn_loop);

    if (J.version != -1)
        this->movups(xsrc_prev, this->ptr[src_ - J.H * J.W * 32 + 16]);
    this->movups(xsrc_lo, this->ptr[src_]);
    this->movups(xsrc_hi, this->ptr[src_ + 4 * sizeof(float)]);
    if (J.version != +1)
        this->movups(xsrc_next, this->ptr[src_ + J.H * J.W * 32]);

    if (J.version != -1) this->movups(this->ptr[t + 0], xsrc_prev);
    this->movups(this->ptr[t + 16], xsrc_lo);
    this->movups(this->ptr[t + 16 + 4 * sizeof(float)], xsrc_hi);
    if (J.version != +1) this->movups(this->ptr[t + 48], xsrc_next);

    this->movups(xa_lo, this->ptr[t + 16 - 8]);
    this->movups(xa_hi, this->ptr[t + 16 - 8 + 4 * sizeof(float)]);
    this->movups(xb_lo, this->ptr[t + 16 - 4]);
    this->movups(xb_hi, this->ptr[t + 16 - 4 + 4 * sizeof(float)]);
    this->movups(xd_lo, this->ptr[t + 16 + 4]);
    this->movups(xd_hi, this->ptr[t + 16 + 4 + 4 * sizeof(float)]);
    this->movups(xe_lo, this->ptr[t + 16 + 8]);
    this->movups(xe_hi, this->ptr[t + 16 + 8 + 4 * sizeof(float)]);
    this->movaps(xc_lo, xsrc_lo);
    this->movaps(xc_hi, xsrc_hi);
    this->mulps(xsum_lo, xc_lo);
    this->mulps(xsum_hi, xc_hi);
    this->mulps(xa_lo, xa_lo);
    this->mulps(xa_hi, xa_hi);
    this->addps(xsum_lo, xa_lo);
    this->addps(xsum_hi, xa_hi); // xsum <- xsum + xa*xa
    this->mulps(xb_lo, xb_lo);
    this->mulps(xb_hi, xb_hi);
    this->addps(xsum_lo, xb_lo);
    this->addps(xsum_hi, xb_hi);
    this->mulps(xd_lo, xd_lo);
    this->mulps(xd_hi, xd_hi);
    this->addps(xsum_lo, xd_lo);
    this->addps(xsum_hi, xd_hi);
    this->mulps(xe_lo, xe_lo);
    this->mulps(xe_hi, xe_hi);
    this->addps(xsum_lo, xe_lo);
    this->addps(xsum_hi, xe_hi);

    this->mulps(xsum_lo, xalpha_);
    this->mulps(xsum_hi, xalpha_);
    this->addps(xsum_lo, xk_);
    this->addps(xsum_hi, xk_); // xsum <- xsum*xalpha_+xk_

    this->movaps(xbase_lo, xsum_lo);
    this->movaps(xbase_hi, xsum_hi);
    if (pk_ != prop_kind::forward_inference) {
        this->movups(this->ptr[scratch_], xbase_lo);
        this->movups(this->ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    this->mulps(xsum_lo, xsum_lo);
    this->mulps(xsum_hi, xsum_hi);
    this->mulps(xsum_lo, xbase_lo);
    this->mulps(xsum_hi, xbase_hi); // xsum = xbase^3;
    this->sqrtps(xsum_lo, xsum_lo);
    this->sqrtps(xsum_hi, xsum_hi);
    this->sqrtps(xsum_lo, xsum_lo);
    this->sqrtps(xsum_hi, xsum_hi); // xsum = xbase^0.75
    this->divps(xsrc_lo, xsum_lo);
    this->divps(xsrc_hi, xsum_hi); // xdst = xsrc / xsum
    this->movups(this->ptr[dst_], xsrc_lo);
    this->movups(this->ptr[dst_ + 4 * sizeof(float)], xsrc_hi);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) add(scratch_, 32);
    this->dec(hw);
    this->cmp(hw, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->add(t, 64);
    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const struct nhwc_across_t &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nhwc_across)
    , nhwc_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nhwc_across_t &J) {
    static const uint32_t mask[] = {0, 0, 0x80000000, 0x80000000, 0x80000000,
            0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0};

    const Xbyak::Reg64 &c = this->r9;
    const Xbyak::Ymm &ya = this->ymm2;
    const Xbyak::Ymm &yb = this->ymm3;
    const Xbyak::Ymm &yc = this->ymm4;
    const Xbyak::Ymm &yd = this->ymm5;
    const Xbyak::Ymm &ye = this->ymm6;
    const Xbyak::Ymm &ysum = this->ymm7;
    const Xbyak::Ymm &ydst = this->ymm8;
    const Xbyak::Ymm &ybase = this->ymm9;
    const Xbyak::Ymm &ymask = this->ymm10;

    this->preamble();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    this->vxorps(ysum, ysum, ysum);

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[0]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(ya, ymask, this->ptr[src_ - 8]);
    this->vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[1]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(yb, ymask, this->ptr[src_ - 4]);
    this->vfmadd231ps(ysum, yb, yb);

    this->mov(c, J.C / 8 - 1);
    Label lrn_loop;
    this->L(lrn_loop);

    this->vmovups(yc, this->ptr[src_]);
    this->vmovups(yd, this->ptr[src_ + 4]);
    this->vmovups(ye, this->ptr[src_ + 8]);
    this->vfmadd231ps(ysum, yc, yc);
    this->vfmadd231ps(ysum, yd, yd);
    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75

    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75
    this->vmovups(this->ptr[dst_], ydst);

    this->vxorps(ysum, ysum, ysum);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, 32);

    this->vmovups(ya, this->ptr[src_ - 8]);
    this->vfmadd231ps(ysum, ya, ya);
    this->vmovups(yb, this->ptr[src_ - 4]);
    this->vfmadd231ps(ysum, yb, yb);

    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->vmovups(yc, this->ptr[src_]);
    this->vfmadd231ps(ysum, yc, yc);

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[2]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(yd, ymask, this->ptr[src_ + 4]);
    this->vfmadd231ps(ysum, yd, yd); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[3]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(ye, ymask, this->ptr[src_ + 8]);
    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    this->vmovups(this->ptr[dst_], ydst);

    this->postamble();
}

template <>
jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::
        jit_uni_lrn_fwd_kernel_t(const struct nhwc_across_t &J, float A,
                float K, prop_kind_t pk, void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nhwc_across)
    , nhwc_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::generate(
        const nhwc_across_t &J) {
    static uint32_t store[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const Xbyak::Reg64 c = this->r9;

    const Xbyak::Xmm &xdst_lo = this->xmm0;
    const Xbyak::Xmm &xdst_hi = this->xmm1;
    const Xbyak::Xmm &xa_lo = this->xmm2;
    const Xbyak::Xmm &xa_hi = this->xmm3;
    const Xbyak::Xmm &xb_lo = this->xmm2;
    const Xbyak::Xmm &xb_hi = this->xmm3;
    const Xbyak::Xmm &xc_lo = this->xmm4;
    const Xbyak::Xmm &xc_hi = this->xmm5;
    const Xbyak::Xmm &xd_lo = this->xmm6;
    const Xbyak::Xmm &xd_hi = this->xmm7;
    const Xbyak::Xmm &xe_lo = this->xmm8;
    const Xbyak::Xmm &xe_hi = this->xmm9;
    const Xbyak::Xmm &xsum_lo = this->xmm10;
    const Xbyak::Xmm &xsum_hi = this->xmm11;
    // unused: xmm12, xmm13;
    const Xbyak::Xmm &xbase_lo = this->xmm14;
    const Xbyak::Xmm &xbase_hi = this->xmm15;

    this->preamble();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        mov(scratch_, this->ptr[this->param1 + 16]);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(xalpha_, this->imm_addr64_);
    this->shufps(xalpha_, xalpha_, 0);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(xk_, this->imm_addr64_);
    this->shufps(xk_, xk_, 0);

    this->mov(store_addr_, reinterpret_cast<size_t>(&store[0]));
    this->and_(store_addr_, -15);
    this->movups(this->ptr[store_addr_], xalpha_);
    this->movups(this->ptr[store_addr_ + 4 * sizeof(float)], xk_);

    this->xorps(xsum_lo, xsum_lo);
    this->xorps(xsum_hi, xsum_hi);

    /* load the 2 first blocks of channels
     * block:         | -- low -- | -- hi --  |
     * C:             [c1,c2,c3,c4,c5,c6,c7,c8]
     * xa_lo << 2 [0,0,c1,c2]
     * xa_hi                [c3,c4,c5,c6]
     * xb_lo << 1   [0,c1,c2,c3]
     * xb_hi                   [c4,c5,c6,c7]
     *                | --  data  --     (...)
     *                ^ memory boundary
     */
    this->movups(xa_lo, this->ptr[src_]);
    this->movups(xa_hi, this->ptr[src_ + 2 * sizeof(float)]);
    this->pslldq(xa_lo, 2 * sizeof(float));
    this->mulps(xa_lo, xa_lo);
    this->mulps(xa_hi, xa_hi);
    this->addps(xsum_lo, xa_lo);
    this->addps(xsum_hi, xa_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    this->movups(xb_lo, this->ptr[src_]);
    this->movups(xb_hi, this->ptr[src_ + 3 * sizeof(float)]);
    this->pslldq(xb_lo, 1 * sizeof(float));
    this->mulps(xb_lo, xb_lo);
    this->mulps(xb_hi, xb_hi);
    this->addps(xsum_lo, xb_lo);
    this->addps(xsum_hi, xb_hi);

    this->mov(c, J.C / 8 - 1);
    Label lrn_loop;
    this->L(lrn_loop);

    this->movups(xc_lo, this->ptr[src_]);
    this->movups(xc_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->movups(xd_lo, this->ptr[src_ + 4]);
    this->movups(xd_hi, this->ptr[src_ + 4 + 4 * sizeof(float)]);
    this->movups(xe_lo, this->ptr[src_ + 8]);
    this->movups(xe_hi, this->ptr[src_ + 8 + 4 * sizeof(float)]);
    this->mulps(xc_lo, xc_lo);
    this->mulps(xc_hi, xc_hi);
    this->addps(xsum_lo, xc_lo);
    this->addps(xsum_hi, xc_hi);
    this->mulps(xd_lo, xd_lo);
    this->mulps(xd_hi, xd_hi);
    this->addps(xsum_lo, xd_lo);
    this->addps(xsum_hi, xd_hi);
    this->mulps(xe_lo, xe_lo);
    this->mulps(xe_hi, xe_hi);
    this->addps(xsum_lo, xe_lo);
    this->addps(xsum_hi, xe_hi);

    this->movaps(xdst_lo, xsum_lo);
    this->movaps(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha_+xk_
    this->mulps(xdst_lo, this->ptr[store_addr_]);
    this->mulps(xdst_hi, this->ptr[store_addr_]);
    this->addps(xdst_lo, this->ptr[store_addr_ + 4 * sizeof(float)]);
    this->addps(xdst_hi, this->ptr[store_addr_ + 4 * sizeof(float)]);

    this->movaps(xbase_lo, xdst_lo);
    this->movaps(xbase_hi, xdst_hi);
    if (pk_ != prop_kind::forward_inference) {
        this->movups(this->ptr[scratch_], xbase_lo);
        this->movups(this->ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    this->mulps(xdst_lo, xdst_lo);
    this->mulps(xdst_hi, xdst_hi);
    this->mulps(xdst_lo, xbase_lo);
    this->mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi);
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75

    this->movups(xc_lo, this->ptr[src_]);
    this->movups(xc_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->divps(xc_lo, xdst_lo);
    this->divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75
    this->movups(this->ptr[dst_], xc_lo);
    this->movups(this->ptr[dst_ + 4 * sizeof(float)], xc_hi);

    this->xorps(xsum_lo, xsum_lo);
    this->xorps(xsum_hi, xsum_hi);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, 32);

    this->movups(xa_lo, this->ptr[src_ - 8]);
    this->movups(xa_hi, this->ptr[src_ - 8 + 4 * sizeof(float)]);
    this->mulps(xa_lo, xa_lo);
    this->mulps(xa_hi, xa_hi);
    this->addps(xsum_lo, xa_lo);
    this->addps(xsum_hi, xa_hi);
    this->movups(xb_lo, this->ptr[src_ - 4]);
    this->movups(xb_hi, this->ptr[src_ - 4 + 4 * sizeof(float)]);
    this->mulps(xb_lo, xb_lo);
    this->mulps(xb_hi, xb_hi);
    this->addps(xsum_lo, xb_lo);
    this->addps(xsum_hi, xb_hi);

    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, this->T_NEAR);

    /* compute last 3 blocks of channels:
     * block:       | -- low -- | -- hi --  |
     * C:           [c1,c2,c3,c4,c5,c6,c7,c8]
     * xc_lo|xc_hi  [c1,c2,c3,c4|c5,c6,c7,c8]
     * xd_lo           [c2,c3,c4,c5]
     * xd_hi >> 1                  [c6,c7,c8, 0]
     * xe_lo              [c3,c4,c5,c6]
     * xe_hi >> 2                     [c7,c8, 0, 0]
     *                  (...) --  data  --  | -- illegal reading -- (...)
     *                                      ^ memory boundary
     */
    this->movups(xc_lo, this->ptr[src_]);
    this->movups(xc_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->mulps(xc_lo, xc_lo);
    this->mulps(xc_hi, xc_hi);
    this->addps(xsum_lo, xc_lo);
    this->addps(xsum_hi, xc_hi);

    this->movups(xd_lo, this->ptr[src_ + 1 * sizeof(float)]);
    this->movups(xd_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->psrldq(xd_hi, 1 * sizeof(float));
    this->mulps(xd_lo, xd_lo);
    this->mulps(xd_hi, xd_hi);
    this->addps(xsum_lo, xd_lo);
    this->addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    this->movups(xe_lo, this->ptr[src_ + 2 * sizeof(float)]);
    this->movups(xe_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->psrldq(xe_hi, 2 * sizeof(float));
    this->mulps(xe_lo, xe_lo);
    this->mulps(xe_hi, xe_hi);
    this->addps(xsum_lo, xe_lo);
    this->addps(xsum_hi, xe_hi);

    this->movups(xdst_lo, xsum_lo);
    this->movups(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha_+xk_
    this->mulps(xdst_lo, this->ptr[store_addr_]);
    this->mulps(xdst_hi, this->ptr[store_addr_]);
    this->addps(xdst_lo, this->ptr[store_addr_ + 4 * sizeof(float)]);
    this->addps(xdst_hi, this->ptr[store_addr_ + 4 * sizeof(float)]);

    this->movaps(xbase_lo, xdst_lo);
    this->movaps(xbase_hi, xdst_hi);
    if (pk_ != prop_kind::forward_inference) {
        this->movups(this->ptr[scratch_], xbase_lo);
        this->movups(this->ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    this->mulps(xdst_lo, xdst_lo);
    this->mulps(xdst_hi, xdst_hi);
    this->mulps(xdst_lo, xbase_lo);
    this->mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi);
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75
    this->movups(xc_lo, this->ptr[src_]);
    this->movups(xc_hi, this->ptr[src_ + 4 * sizeof(float)]);
    this->divps(xc_lo, xdst_lo);
    this->divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75

    this->movups(this->ptr[dst_], xc_lo);
    this->movups(this->ptr[dst_ + 4 * sizeof(float)], xc_hi);

    this->postamble();
}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::nchw_body(
        int tail, int HW, prop_kind_t pk, Xbyak::Ymm ymask, Xbyak::Ymm ya,
        Xbyak::Ymm yb, Xbyak::Ymm yc, Xbyak::Ymm yd, Xbyak::Ymm ye,
        Xbyak::Ymm ysum) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_body(int tail, int HW,
        prop_kind_t pk, Xbyak::Ymm ymask, Xbyak::Ymm ya, Xbyak::Ymm yb,
        Xbyak::Ymm yc, Xbyak::Ymm yd, Xbyak::Ymm ye, Xbyak::Ymm ysum) {
    const Xbyak::Ymm &ydst = this->ymm14;
    const Xbyak::Ymm &ybase = this->ymm15;

    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference) {
        if (tail != 0)
            this->vmaskmovps(this->ptr[scratch_], ymask, ybase);
        else
            this->vmovups(this->ptr[scratch_], ybase);
    }
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    if (tail != 0)
        this->vmaskmovps(this->ptr[dst_], ymask, ydst);
    else
        this->vmovups(this->ptr[dst_], ydst);

    this->vfnmadd231ps(ysum, ya, ya);
    this->vmovups(ya, yb);
    this->vmovups(yb, yc);
    this->vmovups(yc, yd);
    this->vmovups(yd, ye);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_tail_sse41(int tail,
        Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi) {}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41,
        dnnl::impl::data_type::f32>::nchw_tail_sse41(int tail,
        Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi) {
    Xbyak::Xmm xmm_tmp = xmm10;
    this->movaps(xmm_tmp, xtail_hi);

    if (tail > 3) {
        /* Store upper-half directly */
        this->movups(this->ptr[reg_dst + (tail - 4) * sizeof(float)], xtail_hi);
        this->movaps(xmm_tmp, xtail_lo);
        tail -= 4;
    }
    if (tail > 0) {
        /* Store on a single-element basis when 'tail' overlaps
         * with 'src_' */
        this->psrldq(xmm_tmp, (4 - tail) * sizeof(float));
        this->movss(this->ptr[reg_dst], xmm_tmp);

        for (int i = 1; i < tail; i++) {
            this->psrldq(xmm_tmp, sizeof(float));
            this->movss(this->ptr[reg_dst + i * sizeof(float)], xmm_tmp);
        }
    }
}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41,
        dnnl::impl::data_type::f32>::nchw_body_sse41(int tail, int HW,
        prop_kind_t pk, Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo,
        Xbyak::Xmm xsum_hi) {
    const Xbyak::Xmm &xdst_lo = this->xmm0;
    const Xbyak::Xmm &xdst_hi = this->xmm1;
    const Xbyak::Xmm &xbase_lo = this->xmm6;
    const Xbyak::Xmm &xbase_hi = this->xmm7;
    const Xbyak::Xmm &xtmp_lo = this->xmm8;
    const Xbyak::Xmm &xtmp_hi = this->xmm9;
    const Xbyak::Xmm &xa_lo = this->xmm6;
    const Xbyak::Xmm &xa_hi = this->xmm7;
    const Xbyak::Xmm &xb_lo = this->xmm8;
    const Xbyak::Xmm &xb_hi = this->xmm9;
    const Xbyak::Xmm &xc_lo = this->xmm10;
    const Xbyak::Xmm &xc_hi = this->xmm11;
    const Xbyak::Xmm &xd_lo = this->xmm12;
    const Xbyak::Xmm &xd_hi = this->xmm13;

    // store xe
    this->movaps(this->ptr[store_addr_ + 10 * 4 * sizeof(float)], xe_lo);
    this->movaps(this->ptr[store_addr_ + 11 * 4 * sizeof(float)], xe_hi);

    this->mulps(xe_lo, xe_lo);
    this->mulps(xe_hi, xe_hi);
    this->addps(xsum_lo, xe_lo);
    this->addps(xsum_hi, xe_hi);

    // xdst <- xsum*xalpha_+xk_
    this->movaps(xdst_lo, xsum_lo);
    this->movaps(xdst_hi, xsum_hi);
    this->mulps(xdst_lo, this->ptr[store_addr_ + 0 * 4 * sizeof(float)]);
    this->mulps(xdst_hi, this->ptr[store_addr_ + 0 * 4 * sizeof(float)]);
    this->addps(xdst_lo, this->ptr[store_addr_ + 1 * 4 * sizeof(float)]);
    this->addps(xdst_hi, this->ptr[store_addr_ + 1 * 4 * sizeof(float)]);

    this->movaps(xbase_lo, xdst_lo);
    this->movaps(xbase_hi, xdst_hi);
    if (pk_ != prop_kind::forward_inference) {
        if (tail != 0) {
            nchw_tail_sse41(tail, scratch_, xbase_lo, xbase_hi);
        } else {
            this->movups(this->ptr[scratch_], xbase_lo);
            this->movups(this->ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
        }
    }
    this->mulps(xdst_lo, xdst_lo);
    this->mulps(xdst_hi, xdst_hi);
    this->mulps(xdst_lo, xbase_lo);
    this->mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi);
    this->sqrtps(xdst_lo, xdst_lo);
    this->sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75
    this->movaps(xtmp_lo, this->ptr[store_addr_ + 6 * 4 * sizeof(float)]);
    this->movaps(xtmp_hi, this->ptr[store_addr_ + 7 * 4 * sizeof(float)]);
    this->divps(xtmp_lo, xdst_lo);
    this->divps(xtmp_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75
    this->movaps(xdst_lo, xtmp_lo);
    this->movaps(xdst_hi, xtmp_hi);

    if (tail != 0) {
        nchw_tail_sse41(tail, dst_, xdst_lo, xdst_hi);
    } else {
        this->movups(this->ptr[dst_], xdst_lo);
        this->movups(this->ptr[dst_ + 4 * sizeof(float)], xdst_hi);
    }

    this->movaps(xa_lo, this->ptr[store_addr_ + 2 * 4 * sizeof(float)]);
    this->movaps(xa_hi, this->ptr[store_addr_ + 3 * 4 * sizeof(float)]);
    this->mulps(xa_lo, xa_lo);
    this->mulps(xa_hi, xa_hi);
    this->subps(xsum_lo, xa_lo);
    this->subps(xsum_hi, xa_hi);

    // xa <- xb
    this->movaps(xb_lo, this->ptr[store_addr_ + 4 * 4 * sizeof(float)]);
    this->movaps(xb_hi, this->ptr[store_addr_ + 5 * 4 * sizeof(float)]);
    this->movaps(this->ptr[store_addr_ + 2 * 4 * sizeof(float)], xb_lo);
    this->movaps(this->ptr[store_addr_ + 3 * 4 * sizeof(float)], xb_hi);

    // xb <- xc
    this->movaps(xc_lo, this->ptr[store_addr_ + 6 * 4 * sizeof(float)]);
    this->movaps(xc_hi, this->ptr[store_addr_ + 7 * 4 * sizeof(float)]);
    this->movaps(this->ptr[store_addr_ + 4 * 4 * sizeof(float)], xc_lo);
    this->movaps(this->ptr[store_addr_ + 5 * 4 * sizeof(float)], xc_hi);

    // xc <- xd
    this->movaps(xd_lo, this->ptr[store_addr_ + 8 * 4 * sizeof(float)]);
    this->movaps(xd_hi, this->ptr[store_addr_ + 9 * 4 * sizeof(float)]);
    this->movaps(this->ptr[store_addr_ + 6 * 4 * sizeof(float)], xd_lo);
    this->movaps(this->ptr[store_addr_ + 7 * 4 * sizeof(float)], xd_hi);

    // xd <- xe
    this->movaps(xe_lo, this->ptr[store_addr_ + 10 * 4 * sizeof(float)]);
    this->movaps(xe_hi, this->ptr[store_addr_ + 11 * 4 * sizeof(float)]);
    this->movaps(this->ptr[store_addr_ + 8 * 4 * sizeof(float)], xe_lo);
    this->movaps(this->ptr[store_addr_ + 9 * 4 * sizeof(float)], xe_hi);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_body_sse41(int tail, int HW,
        prop_kind_t pk, Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo,
        Xbyak::Xmm xsum_hi) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const nchw_across_t &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nchw_across)
    , nchw_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nchw_across_t &J) {
    static const uint32_t mask[]
            = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                    0x80000000, 0x80000000, 0, 0, 0, 0, 0, 0, 0};
    const Xbyak::Reg64 &c = this->r10;
    const Xbyak::Ymm &ymask = this->ymm2;
    const Xbyak::Ymm &ye = this->ymm3;
    const Xbyak::Ymm &ya = this->ymm4;
    const Xbyak::Ymm &yb = this->ymm5;
    const Xbyak::Ymm &yc = this->ymm6;
    const Xbyak::Ymm &yd = this->ymm7;
    const Xbyak::Ymm &ysum = this->ymm8;

    this->preamble();

    if (J.tail != 0) {
        this->mov(
                this->imm_addr64_, reinterpret_cast<size_t>(&mask[7 - J.tail]));
        this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    }
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);

    this->vxorps(ya, ya, ya);
    this->vxorps(yb, yb, yb);
    if (J.tail != 0)
        this->vmaskmovps(yc, ymask, this->ptr[src_ + J.HW * 0]);
    else
        this->vmovups(yc, this->ptr[src_ + J.HW * 0]);
    if (J.tail != 0)
        this->vmaskmovps(yd, ymask, this->ptr[src_ + J.HW * 4]);
    else
        this->vmovups(yd, this->ptr[src_ + J.HW * 4]);

    this->vxorps(ysum, ysum, ysum);
    this->vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
    this->vfmadd231ps(ysum, yd, yd);

    this->mov(c, J.C - 2);
    Label lrn_loop;
    this->L(lrn_loop);

    if (J.tail != 0)
        this->vmaskmovps(ye, ymask, this->ptr[src_ + J.HW * 8]);
    else
        this->vmovups(ye, this->ptr[src_ + J.HW * 8]);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);

    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, J.HW * 4);
    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->vxorps(ye, ye, ye);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);
    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, J.HW * 4);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::~jit_uni_lrn_fwd_kernel_t() = default;

template <>
jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::
        jit_uni_lrn_fwd_kernel_t(const nchw_across_t &J, float A, float K,
                prop_kind_t pk, void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nchw_across)
    , nchw_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <>
void jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>::generate(
        const nchw_across_t &J) {

    /* Load from within the memory boundary of 'src_' and apply a zero-mask to
     * the 'x_hi' register:
     *  block:       src_  |tail = 3
     *  src_:      [x,x,x,x|a,b,c]
     *  x_hi:           [x,a,b,c]
     *  mask:           [0,1,1,1]
     *      (...) --  data  --  | -- illegal reading -- (...)
     *                          ^ memory boundary
     *
     * 'x_lo' is loaded with the elements between 'src_' and 'x_hi' when
     * tail.size is between [5:7]. The register is then left-shifted to
     * clear the overlapping elements with 'x_hi'.
     *  block: - src_ - |  tail = 7
     *  src_:  (...) [x,|a,b,c,d,e,f,g]
     *  x_hi                 [d,e,f,g]
     *  x_lo           [a,b,c,d]
     *    x_lo >> 1: [0,a,b,c]
     *           (...) --  data  --  | -- illegal reading -- (...)
     *                               ^ memory boundary
     *
     *  - seg-fault happens if read occurs anywhere outside the
     *  memory boundary.
     * */
    static const uint32_t mask[]
            = {0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    assert(J.HW > 3);

    const Xbyak::Reg64 &c = r10;

    // unused: xmm2
    const Xbyak::Xmm &xmask_hi = this->xmm3;
    const Xbyak::Xmm &xsum_lo = this->xmm4;
    const Xbyak::Xmm &xsum_hi = this->xmm5;
    const Xbyak::Xmm &xa_lo = this->xmm6;
    const Xbyak::Xmm &xa_hi = this->xmm7;
    const Xbyak::Xmm &xb_lo = this->xmm8;
    const Xbyak::Xmm &xb_hi = this->xmm9;
    const Xbyak::Xmm &xc_lo = this->xmm10;
    const Xbyak::Xmm &xc_hi = this->xmm11;
    const Xbyak::Xmm &xd_lo = this->xmm12;
    const Xbyak::Xmm &xd_hi = this->xmm13;
    const Xbyak::Xmm &xe_lo = this->xmm14;
    const Xbyak::Xmm &xe_hi = this->xmm15;

    const int vlen = cpu_isa_traits<sse41>::vlen / sizeof(float);

    bool compute_tail = J.tail != 0;
    bool load_lo = J.tail == 0 || J.tail > 4;

    size_t h_offset = vlen;
    size_t l_shift = 0;

    this->preamble();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);

    this->sub(rsp, stack_space_needed_);
    this->mov(store_addr_, rsp);
    this->and_(store_addr_, -15);

    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(xalpha_, this->imm_addr64_);
    this->shufps(xalpha_, xalpha_, 0);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(xk_, this->imm_addr64_);
    this->shufps(xk_, xk_, 0);

    // put alpha_ and k_ into store (free up regs)
    this->movaps(this->ptr[store_addr_ + 0 * 4 * sizeof(float)], xalpha_);
    this->movaps(this->ptr[store_addr_ + 1 * 4 * sizeof(float)], xk_);

    if (compute_tail) {
        assert(J.tail > 0 && J.tail < 2 * vlen);
        h_offset = J.tail - vlen;
        l_shift = nstl::min(2 * vlen - J.tail, vlen);

        /* if 'tail' is between [1:3], need to zero-mask for underflow */
        size_t m_off = nstl::min(J.tail - 1, 3);
        this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[m_off]));
        this->movups(xmask_hi, this->ptr[this->imm_addr64_]);
    }
    // init xa, xb
    this->xorps(xa_lo, xa_lo);
    this->xorps(xa_hi, xa_hi);
    this->xorps(xb_lo, xb_lo);
    this->xorps(xb_hi, xb_hi);

    // read xc, xd
    if (load_lo) this->movups(xc_lo, this->ptr[src_ + J.HW * 0]);
    this->movups(xc_hi, this->ptr[src_ + J.HW * 0 + h_offset * sizeof(float)]);
    if (compute_tail) {
        this->pslldq(xc_lo, l_shift * sizeof(float));
        this->andps(xc_hi, xmask_hi);
    }

    if (load_lo) this->movups(xd_lo, this->ptr[src_ + J.HW * 4]);
    this->movups(xd_hi, this->ptr[src_ + J.HW * 4 + h_offset * sizeof(float)]);
    if (compute_tail) {
        this->pslldq(xd_lo, l_shift * sizeof(float));
        this->andps(xd_hi, xmask_hi);
    }

    // put xa, xb, xc, xd into store to free-up regs
    this->movaps(this->ptr[store_addr_ + 2 * 4 * sizeof(float)], xa_lo);
    this->movaps(this->ptr[store_addr_ + 3 * 4 * sizeof(float)], xa_hi);
    this->movaps(this->ptr[store_addr_ + 4 * 4 * sizeof(float)], xb_lo);
    this->movaps(this->ptr[store_addr_ + 5 * 4 * sizeof(float)], xb_hi);
    this->movaps(this->ptr[store_addr_ + 6 * 4 * sizeof(float)], xc_lo);
    this->movaps(this->ptr[store_addr_ + 7 * 4 * sizeof(float)], xc_hi);
    this->movaps(this->ptr[store_addr_ + 8 * 4 * sizeof(float)], xd_lo);
    this->movaps(this->ptr[store_addr_ + 9 * 4 * sizeof(float)], xd_hi);

    this->xorps(xsum_lo, xsum_lo);
    this->xorps(xsum_hi, xsum_hi);
    this->mulps(xc_lo, xc_lo);
    this->mulps(xc_hi, xc_hi);
    this->addps(xsum_lo, xc_lo);
    this->addps(xsum_hi, xc_hi);
    this->mulps(xd_lo, xd_lo);
    this->mulps(xd_hi, xd_hi);
    this->addps(xsum_lo, xd_lo);
    this->addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    this->mov(c, J.C - 2);
    Label lrn_loop;
    this->L(lrn_loop);

    if (load_lo) this->movups(xe_lo, this->ptr[src_ + J.HW * 8]);
    this->movups(xe_hi, this->ptr[src_ + J.HW * 8 + h_offset * sizeof(float)]);
    if (compute_tail) {
        this->pslldq(xe_lo, l_shift * sizeof(float));
        this->andps(xe_hi, xmask_hi);
    }

    nchw_body_sse41(J.tail, J.HW, pk_, xe_lo, xe_hi, xsum_lo, xsum_hi);

    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) add(scratch_, J.HW * 4);
    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->xorps(xe_lo, xe_lo);
    this->xorps(xe_hi, xe_hi);

    nchw_body_sse41(J.tail, J.HW, pk_, xe_lo, xe_hi, xsum_lo, xsum_hi);
    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) add(scratch_, J.HW * 4);

    nchw_body_sse41(J.tail, J.HW, pk_, xe_lo, xe_hi, xsum_lo, xsum_hi);

    this->add(rsp, stack_space_needed_);

    this->postamble();
}

//////////////////////////////////////////////////////////////////////////////
// backward kernel
template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_kernel_t<isa, d_type>::jit_uni_lrn_bwd_kernel_t(
        const nchw8c_across_t &J, float A, float B, int use_h_parallel,
        void *code_ptr, size_t code_size)
    : Base(code_ptr, code_size)
    , config_(lrn_config_t::nchw8c_across)
    , nchw8c_across_(J)
    , nalphabeta_(-2 * A * B)
    , use_h_parallelizm_(use_h_parallel) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::generate(const nchw8c_across_t &J) {

    const Xbyak::Reg64 &t = this->rsp;
    const Xbyak::Reg64 &hw = this->r10;
    const Xbyak::Xmm &xsrc_prev = this->xmm1;
    const Xbyak::Xmm &xws_prev = this->xmm2;
    const Xbyak::Xmm &xdiffdst_prev = this->xmm3;
    const Xbyak::Ymm &ysrc = this->ymm4;
    const Xbyak::Ymm &yws = this->ymm5;
    const Xbyak::Ymm &ydiffdst = this->ymm6;
    const Xbyak::Xmm &xsrc_next = this->xmm7;
    const Xbyak::Xmm &xws_next = this->xmm8;
    const Xbyak::Xmm &xdiffdst_next = this->xmm9;
    const Xbyak::Ymm &ya = this->ymm10;
    const Xbyak::Xmm &xa = this->xmm10;
    const Xbyak::Ymm &yb = this->ymm11;
    const Xbyak::Ymm &yd = this->ymm12;
    const Xbyak::Ymm &ye = this->ymm13;
    const Xbyak::Ymm &ysum = this->ymm14;
    const Xbyak::Ymm &ydiffsrc = this->ymm15;

    this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(diffdst_, this->ptr[this->param1 + GET_OFF(diff_dst)]);
    this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
    this->mov(bwd_intermediate_res_,
            this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    this->mov(diffsrc_, this->ptr[this->param1 + GET_OFF(diff_src)]);
#undef GET_OFF

    this->sub(t, 64);
    this->mov(this->imm_addr64_, float2int(this->nalphabeta_));
    this->vmovq(xnalphabeta_, this->imm_addr64_);
    this->vbroadcastss(vnalphabeta_, xnalphabeta_);

    bool is_single = J.version == 3;
    bool is_first = J.version == -1 || J.version == -2;
    bool is_last = J.version == +1 || J.version == -2;

    if (is_first || is_single) {
        this->vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        this->vmovups(this->ptr[t + 0], xsrc_prev);
    }
    if (is_last || is_single) {
        this->vxorps(xsrc_next, xsrc_next, xsrc_next);
        this->vmovups(this->ptr[t + 48], xsrc_next);
    }
    this->mov(hw, this->use_h_parallelizm_ ? J.W : J.H * J.W);
    Label lrn_loop;
    this->L(lrn_loop);
    {
        if (!is_first && !is_single) {
            this->vmovups(xws_prev, this->ptr[scratch_ - J.H * J.W * 32 + 16]);
            this->vmovups(xsrc_prev, this->ptr[src_ - J.H * J.W * 32 + 16]);
            this->vmovups(
                    xdiffdst_prev, this->ptr[diffdst_ - J.H * J.W * 32 + 16]);
            this->vmulps(xa, xws_prev, xws_prev);
            this->vmulps(xa, xa, xws_prev);
            this->vsqrtps(xa, xa);
            this->vsqrtps(xa, xa);
            this->vmulps(xa, xa, xws_prev);
            this->vdivps(xsrc_prev, xsrc_prev, xa);
            this->vmulps(xdiffdst_prev, xdiffdst_prev, xsrc_prev);
        }

        this->vmovups(ysrc, this->ptr[src_]);
        this->vmovups(yws, this->ptr[scratch_]);
        this->vmovups(ydiffdst, this->ptr[diffdst_]);
        this->vmulps(ya, yws, yws);
        this->vmulps(ya, ya, yws);
        this->vsqrtps(ya, ya);
        this->vsqrtps(ya, ya);
        this->vdivps(ydiffsrc, ydiffdst, ya);
        this->vdivps(ysum, ydiffsrc, yws);
        this->vmulps(ysum, ysum, ysrc);

        if (!is_last && !is_single) {
            this->vmovups(xws_next, this->ptr[scratch_ + J.H * J.W * 32]);
            this->vmovups(xsrc_next, this->ptr[src_ + J.H * J.W * 32]);
            this->vmovups(xdiffdst_next, this->ptr[diffdst_ + J.H * J.W * 32]);
            this->vmulps(xa, xws_next, xws_next);
            this->vmulps(xa, xa, xws_next);
            this->vsqrtps(xa, xa);
            this->vsqrtps(xa, xa);
            this->vmulps(xa, xa, xws_next);
            this->vdivps(xsrc_next, xsrc_next, xa);
            this->vmulps(xdiffdst_next, xdiffdst_next, xsrc_next);
        }

        if (!is_first && !is_single)
            this->vmovups(this->ptr[t + 0], xdiffdst_prev);
        this->vmovups(this->ptr[t + 16], ysum);
        if (!is_last && !is_single)
            this->vmovups(this->ptr[t + 48], xdiffdst_next);

        this->vmovups(ya, this->ptr[t + 16 - 8]);
        this->vmovups(yb, this->ptr[t + 16 - 4]);
        this->vaddps(ysum, ysum, ya);
        this->vmulps(ysrc, ysrc, vnalphabeta_);
        this->vaddps(ysum, ysum, yb);

        this->vmovups(yd, this->ptr[t + 16 + 4]);
        this->vmovups(ye, this->ptr[t + 16 + 8]);
        this->vaddps(ysum, ysum, yd);
        this->vaddps(ysum, ysum, ye);

        this->vfmadd231ps(ydiffsrc, ysum, ysrc);

        this->vmovups(this->ptr[diffsrc_], ydiffsrc);

        this->add(src_, 32);
        this->add(diffsrc_, 32);
        this->add(diffdst_, 32);
        this->add(scratch_, 32);

        this->dec(hw);
        this->cmp(hw, 0);
        this->jne(lrn_loop, this->T_NEAR);
    }

    this->add(t, 64);
    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_kernel_t<isa, d_type>::jit_uni_lrn_bwd_kernel_t(
        const within_config_t &config, float A, float B, void *code_ptr,
        size_t code_size)
    : Base(config, code_ptr, code_size)
    , config_(lrn_config_t::within_config)
    , within_config_(config)
    , nalphabeta_(-2.0f * A * B) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::generate(
        const within_config_t &config) {

    this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(diffdst_, this->ptr[this->param1 + GET_OFF(diff_dst)]);
    this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
    this->mov(bwd_intermediate_res_,
            this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    this->mov(diffsrc_, this->ptr[this->param1 + GET_OFF(diff_src)]);
#undef GET_OFF
    this->load_constant(nalphabeta_, vnalphabeta_, xnalphabeta_);

    static const int max_reg_blocks = isa == avx512_common ? 3 : 1;
    this->within_loop(config, max_reg_blocks, prop_kind::backward);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::within_body(int hoff, int Hoff,
        int woff, int Woff, int stride, prop_kind_t pk, const int reg_block,
        int pixel_offset) {

    static const std::array<Vmm, 3> vsum {{Vmm(1), Vmm(9), Vmm(18)}};
    static const std::array<std::array<Vmm, 3>, 3> diff_dst {{
            {{Vmm(2), Vmm(3), Vmm(6)}},
            {{Vmm(10), Vmm(11), Vmm(23)}},
            {{Vmm(19), Vmm(20), Vmm(26)}},
    }};
    static const std::array<std::array<Vmm, 3>, 3> ws1 {{
            {{Vmm(4), Vmm(5), Vmm(15)}},
            {{Vmm(12), Vmm(13), Vmm(27)}},
            {{Vmm(21), Vmm(22), Vmm(28)}},
    }};
    static const std::array<Vmm, 3> ws0 = !this->emulate_bfloat_
            ? std::array<Vmm, 3> {{Vmm(29), Vmm(30), Vmm(31)}}
            : std::array<Vmm, 3> {{Vmm(6), Vmm(15), Vmm(23)}};
    static const std::array<Vmm, 3> src {{Vmm(7), Vmm(16), Vmm(24)}};
    static const std::array<Vmm, 3> a {{Vmm(8), Vmm(17), Vmm(25)}};

    static const std::size_t used_tmp_regs
            = this->emulate_bfloat_ ? ws1[0].size() - 1 : ws1[0].size();

    IRB_LOOP(this->uni_vxorps(vsum[irb], vsum[irb], vsum[irb]));
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            const auto idx = this->tempIdx_ % used_tmp_regs;
            IRB_LOOP(this->load_data(diff_dst[irb][idx],
                    this->ptr[(diffdst_ + pixel_offset + irb_off)
                            + (i * stride + j) * this->single_pixel_offset_]));
            IRB_LOOP(this->load_data(ws1[irb][idx],
                    this->ptr[(bwd_intermediate_res_ + pixel_offset + irb_off)
                            + (i * stride + j) * this->single_pixel_offset_]));

            if (i == 0 && j == 0) {
                if (d_type == dnnl::impl::data_type::bf16) {
                    IRB_LOOP(this->load_data(ws0[irb],
                            this->ptr[(scratch_ + pixel_offset + irb_off)]));
                    IRB_LOOP(
                            this->vdivps(a[irb], diff_dst[irb][idx], ws0[irb]));
                } else {
                    IRB_LOOP(this->vdivps(a[irb], diff_dst[irb][idx],
                            this->ptr[(scratch_ + pixel_offset + irb_off)]));
                }
            }

            IRB_LOOP(this->vfmadd231ps(
                    vsum[irb], ws1[irb][idx], diff_dst[irb][idx]));
            ++(this->tempIdx_);
        }
    }

    this->tempIdx_ = this->tempIdx_ % used_tmp_regs;

    if (d_type == dnnl::impl::data_type::bf16) {
        IRB_LOOP(this->load_data(
                src[irb], this->ptr[(src_ + pixel_offset + irb_off)]));
        IRB_LOOP(this->vmulps(src[irb], this->vnalphabeta_, src[irb]));
    } else {
        IRB_LOOP(this->vmulps(src[irb], this->vnalphabeta_,
                this->ptr[(src_ + pixel_offset + irb_off)]));
    }

    IRB_LOOP(this->vfmadd231ps(a[irb], src[irb], vsum[irb]));

    IRB_LOOP(this->store_data(
            this->ptr[diffsrc_ + pixel_offset + irb_off], a[irb]));

    if (isa == avx512_common)
        this->reg_block_idx_ = (this->reg_block_idx_ % vsum.size()) + 1;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::move_data_pointers(
        int pixel_count, prop_kind_t pk) {
    const int pixel_offset = this->single_pixel_offset_ * pixel_count;
    this->add(src_, pixel_offset);
    this->add(diffsrc_, pixel_offset);
    this->add(diffdst_, pixel_offset);
    this->add(scratch_, pixel_offset);
    this->add(bwd_intermediate_res_, pixel_offset);
}

template class jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>;
template class jit_uni_lrn_fwd_kernel_t<avx2, dnnl::impl::data_type::f32>;
template class jit_uni_lrn_fwd_kernel_t<avx512_common,
        dnnl::impl::data_type::f32>;
template class jit_uni_lrn_fwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>;

template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<sse41, dnnl::impl::data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx2, dnnl::impl::data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx512_common, dnnl::impl::data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx512_common, dnnl::impl::data_type::bf16>>;

template class jit_uni_lrn_bwd_kernel_t<avx512_common,
        dnnl::impl::data_type::f32>;
template class jit_uni_lrn_bwd_kernel_t<avx512_common,
        dnnl::impl::data_type::bf16>;
template class jit_uni_lrn_bwd_kernel_t<avx2, dnnl::impl::data_type::f32>;

template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx2, dnnl::impl::data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx512_common, dnnl::impl::data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx512_common, dnnl::impl::data_type::bf16>>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
