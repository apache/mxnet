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

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace gemm_x8s8s32x_convolution_utils {

using namespace dnnl::impl::cpu::gemm_x8s8s32x_convolution_utils;

struct jit_pp_ker_t : pp_ker_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);

    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp) {
        if (do_eltwise_)
            eltwise_injector_.reset(
                    new jit_uni_eltwise_injector_f32<avx512_common>(this,
                            eltwise_, true, Xbyak::util::rax,
                            Xbyak::Opmask(2)));

        if (bias_data_type_ != data_type::undef)
            bias_data_type_size_ = types::data_type_size(bias_data_type_);
        dst_data_type_size_ = types::data_type_size(dst_data_type_);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(void *void_dst, const acc_data_t *acc, const char *bias,
            const float *scales, float nslope, float sum_scale,
            float signed_scale, int g, size_t start,
            size_t end) const override {

        if (end <= start) return;

        char *dst = (char *)void_dst;

        ker_args_t args;
        size_t oc_offset = start % OC_;
        size_t os_offset = start / OC_;
        args.acc = acc + start;
        args.dst = dst
                + (os_offset * dst_os_stride_ + oc_offset)
                        * dst_data_type_size_;
        args.bias = bias + (g * jcp_.oc + oc_offset) * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * (g * jcp_.oc + oc_offset);
        args.nslope = nslope;
        args.sum_scale = sum_scale;
        args.signed_scale = signed_scale;
        args.len = end - start;
        args.oc_offset = oc_offset;
        jit_generator::operator()(&args);
    }

private:
    void generate() override;

    struct ker_args_t {
        char *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float nslope;
        float sum_scale;
        float signed_scale;
        size_t len;
        size_t oc_offset;
    };

    std::unique_ptr<jit_uni_eltwise_injector_f32<avx512_common>>
            eltwise_injector_;

    size_t bias_data_type_size_ = 0;
    size_t dst_data_type_size_ = 0;
};

void jit_pp_ker_t::generate() {
    using namespace Xbyak;
    using namespace utils;

    // TODO: clean-up
    Reg64 reg_param = abi_param1;
    Reg64 reg_dst = rdx;
    Reg64 reg_acc = rax;
    Reg64 reg_bias = rbx;
    Reg64 reg_scales = rsi;

    Reg64 reg_len = r8;
    Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Reg64 reg_oc_offset = r9;
    Reg64 reg_rem_mask_short = r10;
    Reg64 reg_rem_mask_vlen = r11;
    Reg64 reg_tmp_comp = r12; // used to broadcast scalar values to vreg
    Opmask kreg_rem_mask_short = k1;
    Opmask kreg_rem_mask_vlen = k3;

    size_t vlen = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    for (; vlen >= 1 && (OC_ % vlen != 0); --vlen) {}

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_scale = Zmm(1);
    Zmm vreg_nslope = Zmm(2);
    Zmm vreg_sum_scale = Zmm(3);
    Zmm vreg_signed_scale = Zmm(4);
    Zmm vreg_saturation_ubound = Zmm(5);

    size_t def_unroll = 4;
    size_t max_unroll = 12;
    size_t zmm_step = 2;
    if (do_sum_) {
        max_unroll = 8;
        zmm_step = 3;
    }

    auto vreg_dst = [&](int idx) { return Zmm(6 + idx * zmm_step + 0); };
    auto vreg_bias = [&](int idx) { return Zmm(6 + idx * zmm_step + 1); };
    auto vreg_prev_dst = [&](int idx) { return Zmm(6 + idx * zmm_step + 2); };

    preamble();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    vbroadcastss(vreg_nslope, ptr[reg_param + PARAM_OFF(nslope)]);
    vbroadcastss(vreg_sum_scale, ptr[reg_param + PARAM_OFF(sum_scale)]);
    vbroadcastss(vreg_signed_scale, ptr[reg_param + PARAM_OFF(signed_scale)]);
    if (scale_idx_mult_ == 0) vbroadcastss(vreg_scale, dword[reg_scales]);

#undef PARAM_OFF

    mov(reg_rem_mask_vlen, 1);
    shl(reg_rem_mask_vlen, vlen);
    sub(reg_rem_mask_vlen, 1);
    kmovq(kreg_rem_mask_vlen, reg_rem_mask_vlen);

    if (do_eltwise_) vxorps(vreg_zero, vreg_zero, vreg_zero);
    init_saturate_f32(vreg_zero, vreg_saturation_ubound, reg_tmp_comp,
            data_type::f32, dst_data_type_);

    // Load accumulated value, convert to float, apply sum (if any),
    // bias (if any), scaling, and relu (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        if (scale_idx_mult_ > 0) {
            assert(scale_idx_mult_ == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            if (apply_mask)
                vreg_scale_ = vreg_scale_ | kreg_rem_mask_short;
            else
                vreg_scale_ = vreg_scale_ | kreg_rem_mask_vlen;
            vmovups(vreg_scale_, scale_addr);
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (apply_mask)
            vreg_dst_ = vreg_dst_ | kreg_rem_mask_short;
        else
            vreg_dst_ = vreg_dst_ | kreg_rem_mask_vlen;
        vcvtdq2ps(vreg_dst_, acc_addr);

        if (do_signed_scaling_)
            vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_short;
            else
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_vlen;

            switch (bias_data_type_) {
                case data_type::s8: vpmovsxbd(vreg_bias_, bias_addr); break;
                case data_type::u8: vpmovzxbd(vreg_bias_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: vmovups(vreg_bias_, bias_addr); break;
                default: assert(!"unimplemented");
            }
            if (bias_data_type_ != data_type::f32)
                vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];

        if (do_sum_) {
            auto vreg_prev_dst_ = vreg_prev_dst(idx);
            if (apply_mask)
                vreg_prev_dst_ = vreg_prev_dst_ | kreg_rem_mask_short;
            else
                vreg_prev_dst_ = vreg_prev_dst_ | kreg_rem_mask_vlen;

            switch (dst_data_type_) {
                case data_type::f32:
                case data_type::s32: vmovups(vreg_prev_dst_, dst_addr); break;
                case data_type::s8: vpmovsxbd(vreg_prev_dst_, dst_addr); break;
                case data_type::u8: vpmovzxbd(vreg_prev_dst_, dst_addr); break;
                default: assert(!"unsupported data type");
            }
            if (dst_data_type_ != data_type::f32)
                vcvtdq2ps(vreg_prev_dst(idx), vreg_prev_dst(idx));

            vfmadd231ps(vreg_dst(idx), vreg_prev_dst(idx), vreg_sum_scale);
        }

        if (do_eltwise_)
            eltwise_injector_->compute_vector(vreg_dst(idx).getIdx());

        if (one_of(dst_data_type_, data_type::u8, data_type::s8,
                    data_type::s32)) {
            saturate_f32(vreg_dst(idx), vreg_zero, vreg_saturation_ubound,
                    dst_data_type_);
            vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
        }

        switch (dst_data_type_) {
            case data_type::s8: vpmovsdb(dst_addr, vreg_dst_); break;
            case data_type::u8: vpmovusdb(dst_addr, vreg_dst_); break;
            case data_type::f32:
            case data_type::s32: vmovups(dst_addr, vreg_dst_); break;
            default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * dst_data_type_size_);
        add(reg_acc, offset * sizeof(acc_data_t));
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            add(reg_scales, offset * sizeof(float));
        }
        if (do_bias_) add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * dst_data_type_size_]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_) sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
        add(reg_dst, (dst_os_stride_ - OC_) * dst_data_type_size_);
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    Label prologue_end;
    cmp(reg_oc_offset, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp, OC_);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp, vlen);
        jle(prologue_loop_tail, T_NEAR);
        L(prologue_loop);
        {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        mov(reg_rem_mask_short, 1);
        // cl == reg_tmp because reg_tmp <= vlen here
        shl(reg_rem_mask_short, cl);
        sub(reg_rem_mask_short, 1);
        jz(prologue_loop_end, T_NEAR);

        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, OC_);
        jle(main_loop_end, T_NEAR);

        Label main_loop;
        L(main_loop);
        {
            size_t OC_loop, OC_tail;
            if (OC_ < max_unroll * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = OC_;
            } else {
                OC_loop = vlen * def_unroll;
                OC_tail = OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            if (OC_tail % vlen) {
                int vlen_tail = OC_tail % vlen;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
            }

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label oc_loop;
                L(oc_loop);
                {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                advance_ptrs_imm(OC_tail);
            }

            rewind_ptrs();
            sub(reg_len, OC_);
            cmp(reg_len, OC_);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        cmp(reg_len, vlen);
        jle(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop);
        {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        mov(reg_rem_mask_short, 1);
        shl(reg_rem_mask_short, cl); // reg_tmp == rcx and reg_tail < vlen
        sub(reg_rem_mask_short, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    if (do_eltwise_) eltwise_injector_->prepare_table();
}

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
    if (!mayiuse(avx512_core)) return nullptr;
    return new jit_pp_ker_t(pd, jcp);
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
