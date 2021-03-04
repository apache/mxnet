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

#include "cpu/x64/jit_uni_layer_normalization_kernels.hpp"
#include "common/bfloat16.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lnorm_utils {

using namespace dnnl::impl::cpu::lnorm_utils;
using namespace dnnl::impl::cpu::x64;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t>
struct jit_transfer_t;

template <>
struct jit_transfer_t<f32> {
    jit_transfer_t(jit_generator &gen, const int simd_w = 8);

    template <data_type_t load_data_type>
    void load(Ymm &vmm_src, Reg64 reg_src, int nelems, size_t offt_elems);

    template <data_type_t store_data_type>
    void store(Ymm &vmm_dst, Reg64 reg_dst, int nelems, size_t offt_elems);

protected:
    jit_generator &gen_;
    const int simd_w_;
};

jit_transfer_t<f32>::jit_transfer_t(jit_generator &gen, const int simd_w)
    : gen_(gen), simd_w_ {simd_w} {}

template <>
struct jit_transfer_t<bf16> : jit_transfer_t<f32> {
    jit_transfer_t(jit_generator &gen);

    template <data_type_t load_data_type>
    void load(Zmm &zmm_src, Reg64 reg_src, int nelems, size_t offt_elems);

    template <data_type_t store_data_type>
    void store(Zmm &zmm_dst, Reg64 reg_dst, int nelems, size_t offt_elems);

private:
    const bool emulate_bf16_;
    const Reg64 reg_tmp_ = r15;
    const Zmm bf16_emu_reserv_1_ = Zmm(28);
    const Zmm bf16_emu_reserv_2_ = Zmm(29);
    const Reg64 bf16_emu_scratch_ = rax;
    const Zmm bf16_emu_reserv_3_ = Zmm(30);
    const Zmm bf16_emu_reserv_4_ = Zmm(31);
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

jit_transfer_t<bf16>::jit_transfer_t(jit_generator &gen)
    : jit_transfer_t<f32>(gen, 16 /* simd_w_ */)
    , emulate_bf16_ {!mayiuse(avx512_core_bf16)} {
    if (emulate_bf16_) {
        this->bf16_emu_ = utils::make_unique<bf16_emulation_t>(&this->gen_,
                this->bf16_emu_reserv_1_, this->bf16_emu_reserv_2_,
                this->bf16_emu_reserv_3_, this->bf16_emu_scratch_,
                this->bf16_emu_reserv_4_);
        this->bf16_emu_->init_vcvtneps2bf16();
    }
}

template <>
void jit_transfer_t<f32>::load<f32>(
        Ymm &vmm_src, Reg64 reg_src, int nelems, size_t offt_elems) {
    if (nelems == 1)
        gen_.vmovss(Xmm(vmm_src.getIdx()),
                dword[reg_src + offt_elems * sizeof(float)]);
    else if (nelems == simd_w_)
        gen_.uni_vmovups(vmm_src, zword[reg_src + offt_elems * sizeof(float)]);
    else
        assert(!"unsupported nelems for load src");
}

template <>
void jit_transfer_t<f32>::store<f32>(
        Ymm &vmm_dst, Reg64 reg_dst, int nelems, size_t offt_elems) {
    if (nelems == 1)
        gen_.vmovss(dword[reg_dst + offt_elems * sizeof(float)],
                Xmm(vmm_dst.getIdx()));
    else if (nelems == simd_w_)
        gen_.uni_vmovups(zword[reg_dst + offt_elems * sizeof(float)], vmm_dst);
    else
        assert(!"unsupported nelems");
}

template <>
void jit_transfer_t<bf16>::load<f32>(
        Zmm &zmm_src, Reg64 reg_src, int nelems, size_t offt_elems) {
    jit_transfer_t<f32>::load<f32>(zmm_src, reg_src, nelems, offt_elems);
}

template <>
void jit_transfer_t<bf16>::store<f32>(
        Zmm &zmm_dst, Reg64 reg_dst, int nelems, size_t offt_elems) {
    jit_transfer_t<f32>::store<f32>(zmm_dst, reg_dst, nelems, offt_elems);
}

template <>
void jit_transfer_t<bf16>::load<bf16>(
        Zmm &zmm_src, Reg64 reg_src, int nelems, size_t offt_elems) {
    if (nelems == 1) {
        const Xmm x_reg = Xmm(zmm_src.getIdx());
        gen_.movzx(reg_tmp_, word[reg_src + offt_elems * sizeof(bfloat16_t)]);
        gen_.movq(x_reg, reg_tmp_);
        gen_.vpslld(x_reg, x_reg, 0x10);
    } else if (nelems == simd_w_) {
        gen_.vpmovzxwd(
                zmm_src, yword[reg_src + offt_elems * sizeof(bfloat16_t)]);
        gen_.vpslld(zmm_src, zmm_src, 0x10);
    } else
        assert(!"unsupported nelems for load src");
}

template <>
void jit_transfer_t<bf16>::store<bf16>(
        Zmm &zmm_dst, Reg64 reg_dst, int nelems, size_t offt_elems) {
    if (nelems == 1) {
        const Ymm ymm_dst = Ymm(zmm_dst.getIdx());
        if (emulate_bf16_)
            bf16_emu_->vcvtneps2bf16(ymm_dst, zmm_dst);
        else
            gen_.vcvtneps2bf16(ymm_dst, zmm_dst);
        const Xmm xmm_dst = Xmm(zmm_dst.getIdx());
        gen_.vpextrw(
                word[reg_dst + offt_elems * sizeof(bfloat16_t)], xmm_dst, 0);
    } else if (nelems == simd_w_) {
        const Ymm ymm_dst = Ymm(zmm_dst.getIdx());
        if (emulate_bf16_)
            bf16_emu_->vcvtneps2bf16(ymm_dst, zmm_dst);
        else
            gen_.vcvtneps2bf16(ymm_dst, zmm_dst);
        gen_.vmovdqu16(
                yword[reg_dst + offt_elems * sizeof(bfloat16_t)], ymm_dst);
    } else
        assert(!"unsupported nelems");
}

template <data_type_t data_type>
struct jit_statistics_kernel_t : statistics_kernel_t<data_type>,
                                 public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(lnorm_utils::jit_statistics_kernel_t);

    jit_statistics_kernel_t(const layer_normalization_pd_t *pd);

    using data_t = typename prec_traits<data_type>::type;
    void operator()(const data_t *src, float *mean, float *var) const override;
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    jit_transfer_t<data_type> jit_transfer_;
    static constexpr int unroll_factor_ = 8;
    static constexpr int simd_w = data_type == bf16 ? 16 : 8;
    using statistics_kernel_t<data_type>::C_;
    using Vmm = typename utils::conditional<data_type == bf16, Xbyak::Zmm,
            Xbyak::Ymm>::type;

    struct ker_args_t {
        const data_t *src;
        float *mean;
        float *var;
    };
    void generate() override;

    template <typename F>
    void compute(F op);

    void reduce();

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_mean = rbx;
    Xbyak::Reg64 reg_var = rbp;
    Xbyak::Reg64 reg_tmp = rax;

    // vector registers 0 .. unroll_factor_ are reseved for unrolling
    Vmm vmm_src = Vmm(14);
    Vmm vmm_mean = Vmm(15);
};

template <data_type_t data_type>
jit_statistics_kernel_t<data_type>::jit_statistics_kernel_t(
        const layer_normalization_pd_t *pd)
    : statistics_kernel_t<data_type>(pd), jit_transfer_ {*this} {
    assert(data_type == bf16 ? mayiuse(avx512_core) : mayiuse(avx2));
}

template <data_type_t data_type>
void jit_statistics_kernel_t<data_type>::operator()(
        const data_t *src, float *mean, float *var) const {
    ker_args_t args;
    args.src = src;
    args.mean = mean;
    args.var = var;
    jit_generator::operator()(&args);
}

template <data_type_t data_type>
void jit_statistics_kernel_t<data_type>::generate() {
    using namespace Xbyak;

    preamble();
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
    mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
    mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
#undef PARAM_OFF

    // compute mean
    compute([=](Vmm vmm_dst) { vaddps(vmm_dst, vmm_dst, vmm_src); });
    vmovss(ptr[reg_mean], Xmm(0));

    //compute var
    vbroadcastss(vmm_mean, Xmm(0));
    compute([=](Vmm vmm_dst) {
        vsubps(vmm_src, vmm_mean, vmm_src);
        vfmadd231ps(vmm_dst, vmm_src, vmm_src);
    });
    vmovss(ptr[reg_var], Xmm(0));

    postamble();
}

template <data_type_t data_type>
template <typename F>
void jit_statistics_kernel_t<data_type>::compute(F op) {
    const int C_vecs = C_ / simd_w;

    uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
    if (C_vecs > 0) {
        const int unroll = C_vecs >= unroll_factor_ ? unroll_factor_ : 1;
        assert(math::is_pow2(unroll));

        for (int i = 1; i < unroll; i++)
            uni_vpxor(Vmm(i), Vmm(i), Vmm(i));

        // unrolled loop
        for (int i = 0; i < C_vecs / unroll; i++)
            for (int j = 0; j < unroll; j++) {
                jit_transfer_.template load<data_type>(
                        vmm_src, reg_src, simd_w, (i * unroll + j) * simd_w);
                op(Vmm(j));
            }

        // unrolled loop reduction
        int n = unroll;
        while (n > 1) {
            for (int j = 0; j < n / 2; j++)
                vaddps(Vmm(j), Vmm(j), Vmm(j + n / 2));
            n = n / 2;
        }

        // unrolled loop remainder
        for (int i = utils::rnd_dn(C_vecs, unroll); i < C_vecs; i++) {
            jit_transfer_.template load<data_type>(
                    vmm_src, reg_src, simd_w, i * simd_w);
            op(Vmm(0));
        }

        // vector reduction
        reduce();
    }

    // vector remainder
    for (int i = utils::rnd_dn(C_, simd_w); i < C_; i++) {
        jit_transfer_.template load<data_type>(vmm_src, reg_src, 1, i);
        op(Vmm(0));
    }

    // scale
    Xmm xmm_tmp = Xmm(vmm_src.getIdx());
    mov(reg_tmp, float2int(C_));
    uni_vmovq(xmm_tmp, reg_tmp);
    vdivss(Xmm(0), Xmm(0), xmm_tmp);
};

template <>
void jit_statistics_kernel_t<bf16>::reduce() {
    Ymm ymm_high = Ymm(1);
    vextractf32x8(ymm_high, Zmm(0), 1);
    vaddps(Ymm(0), ymm_high, Ymm(0));
    vhaddps(Ymm(0), Ymm(0), Ymm(0));
    vhaddps(Ymm(0), Ymm(0), Ymm(0));
    Xmm xmm_high = Xmm(1);
    vextractf128(xmm_high, Ymm(0), 1);
    vaddps(Xmm(0), xmm_high, Xmm(0));
}

template <>
void jit_statistics_kernel_t<f32>::reduce() {
    Xmm xmm_high = Xmm(1);
    vextractf128(xmm_high, Ymm(0), 1);
    vaddps(Xmm(0), xmm_high, Xmm(0));
    vhaddps(Xmm(0), Xmm(0), Xmm(0));
    vhaddps(Xmm(0), Xmm(0), Xmm(0));
}

template <data_type_t data_type>
struct jit_data_kernel_t : data_kernel_t<data_type>, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(lnorm_utils::jit_data_kernel_t);

    jit_data_kernel_t(const layer_normalization_pd_t *pd);

    using data_t = typename prec_traits<data_type>::type;
    void operator()(const data_t *src, data_t *dst, const float *ss,
            const float *mean, const float *var) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    jit_transfer_t<data_type> jit_transfer_;
    static constexpr int simd_w = data_type == bf16 ? 16 : 8;
    using Vmm = typename utils::conditional<data_type == bf16, Xbyak::Zmm,
            Xbyak::Ymm>::type;
    using data_kernel_t<data_type>::C_;
    using data_kernel_t<data_type>::eps_;
    using data_kernel_t<data_type>::use_scaleshift_;

    struct ker_args_t {
        const data_t *src;
        data_t *dst;
        const float *ss;
        const float *mean;
        const float *inv_sqrtvar;
    };

    void generate() override;

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_dst = rax;
    Xbyak::Reg64 reg_ss = r9;
    Xbyak::Reg64 reg_tmp = r8;

    Vmm vmm_inv_sqrtvar = Vmm(10);
    Vmm vmm_data = Vmm(11);
    Vmm vmm_gamma = Vmm(12);
    Vmm vmm_beta = Vmm(13);
    Vmm vmm_tmp = Vmm(14);
    Vmm vmm_mean = Vmm(15);
};

template <data_type_t data_type>
jit_data_kernel_t<data_type>::jit_data_kernel_t(
        const layer_normalization_pd_t *pd)
    : data_kernel_t<data_type>(pd), jit_transfer_ {*this} {
    assert(data_type == bf16 ? mayiuse(avx512_core) : mayiuse(avx2));
}

template <data_type_t data_type>
void jit_data_kernel_t<data_type>::operator()(const data_t *src, data_t *dst,
        const float *ss, const float *mean, const float *var) const {
    ker_args_t args;
    args.src = src;
    args.dst = dst;
    args.ss = ss;
    args.mean = mean;
#ifdef __INTEL_COMPILER
    //Without volatile ICC with -O2 & -O3 optimizes out denominator from
    //inv_sqrtvar and computes 1/denom with lower precision
    const volatile float denom = sqrtf(*var + eps_);
#else
    const float denom = sqrtf(*var + eps_);
#endif
    const float inv_sqrtvar = 1.f / denom;
    args.inv_sqrtvar = &inv_sqrtvar;
    jit_generator::operator()(&args);
}

template <data_type_t data_type>
void jit_data_kernel_t<data_type>::generate() {
    preamble();
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_ss, ptr[reg_param + PARAM_OFF(ss)]);

    Xmm xmm_tmp = Xmm(vmm_tmp.getIdx());
    mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
    vmovss(xmm_tmp, dword[reg_tmp]);
    vbroadcastss(vmm_mean, xmm_tmp);

    mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
    vmovss(xmm_tmp, dword[reg_tmp]);
    vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF
    const int C_vecs = C_ / simd_w;

    auto op = [=](int nelems, size_t offt_elems) {
        if (use_scaleshift_) {
            jit_transfer_.template load<f32>(
                    vmm_gamma, reg_ss, nelems, offt_elems);
            jit_transfer_.template load<f32>(
                    vmm_beta, reg_ss, nelems, offt_elems + C_);
        }
        jit_transfer_.template load<data_type>(
                vmm_data, reg_src, nelems, offt_elems);
        vsubps(vmm_data, vmm_data, vmm_mean);
        vmulps(vmm_data, vmm_data, vmm_inv_sqrtvar);
        if (use_scaleshift_) vfmadd213ps(vmm_data, vmm_gamma, vmm_beta);
        jit_transfer_.template store<data_type>(
                vmm_data, reg_dst, nelems, offt_elems);
    };

    for (int i = 0; i < C_vecs; i++)
        op(simd_w, i * simd_w);

    for (int i = utils::rnd_dn(C_, simd_w); i < C_; i++)
        op(1, i);

    postamble();
}

template <data_type_t data_type>
struct jit_diff_ss_kernel_t : diff_ss_kernel_t<data_type>,
                              public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(lnorm_utils::jit_diff_ss_kernel_t);

    jit_diff_ss_kernel_t(const layer_normalization_pd_t *pd);

    using data_t = typename prec_traits<data_type>::type;
    void operator()(const data_t *src, const data_t *diff_dst,
            float *diff_gamma, float *diff_beta, const float *mean,
            const float *var) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    jit_transfer_t<data_type> jit_transfer_;
    static constexpr int simd_w = data_type == bf16 ? 16 : 8;
    using Vmm = typename utils::conditional<data_type == bf16, Xbyak::Zmm,
            Xbyak::Ymm>::type;
    using diff_ss_kernel_t<data_type>::C_;
    using diff_ss_kernel_t<data_type>::eps_;

    struct ker_args_t {
        const data_t *src;
        const data_t *diff_dst;
        float *diff_gamma;
        float *diff_beta;
        const float *mean;
        const float *inv_sqrtvar;
    };

    void generate() override;

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_diff_dst = rax;
    Xbyak::Reg64 reg_tmp = r10;
    Xbyak::Reg64 reg_diff_gamma = r9;
    Xbyak::Reg64 reg_diff_beta = r8;

    Xbyak::Xmm xmm_tmp = Xbyak::Xmm(9);

    Vmm vmm_inv_sqrtvar = Vmm(10);
    Vmm vmm_ddst = Vmm(11);
    Vmm vmm_dgamma = Vmm(12);
    Vmm vmm_dbeta = Vmm(13);
    Vmm vmm_src = Vmm(14);
    Vmm vmm_mean = Vmm(15);
};

template <data_type_t data_type>
jit_diff_ss_kernel_t<data_type>::jit_diff_ss_kernel_t(
        const layer_normalization_pd_t *pd)
    : diff_ss_kernel_t<data_type>(pd), jit_transfer_ {*this} {
    assert(data_type == bf16 ? mayiuse(avx512_core) : mayiuse(avx2));
}

template <data_type_t data_type>
void jit_diff_ss_kernel_t<data_type>::operator()(const data_t *src,
        const data_t *diff_dst, float *diff_gamma, float *diff_beta,
        const float *mean, const float *var) const {
    ker_args_t args;
    args.src = src;
    args.diff_dst = diff_dst;
    args.diff_gamma = diff_gamma;
    args.diff_beta = diff_beta;
    args.mean = mean;
#ifdef __INTEL_COMPILER
    //Without volatile ICC with -O2 & -O3 optimizes out denominator from
    //inv_sqrtvar and computes 1/denom with lower precision
    const volatile float denom = sqrtf(*var + eps_);
#else
    const float denom = sqrtf(*var + eps_);
#endif
    const float inv_sqrtvar = 1.f / denom;
    args.inv_sqrtvar = &inv_sqrtvar;
    jit_generator::operator()(&args);
}

template <data_type_t data_type>
void jit_diff_ss_kernel_t<data_type>::generate() {
    preamble();
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
    mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
    mov(reg_diff_gamma, ptr[reg_param + PARAM_OFF(diff_gamma)]);
    mov(reg_diff_beta, ptr[reg_param + PARAM_OFF(diff_beta)]);

    mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
    vmovss(xmm_tmp, dword[reg_tmp]);
    vbroadcastss(vmm_mean, xmm_tmp);

    mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
    vmovss(xmm_tmp, dword[reg_tmp]);
    vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF

    const int C_vecs = C_ / simd_w;
    auto op = [=](int nelems, size_t offt_elems) {
        jit_transfer_.template load<data_type>(
                vmm_ddst, reg_diff_dst, nelems, offt_elems);
        jit_transfer_.template load<f32>(
                vmm_dbeta, reg_diff_beta, nelems, offt_elems);
        jit_transfer_.template load<f32>(
                vmm_dgamma, reg_diff_gamma, nelems, offt_elems);
        jit_transfer_.template load<data_type>(
                vmm_src, reg_src, nelems, offt_elems);
        vaddps(vmm_dbeta, vmm_dbeta, vmm_ddst);
        vsubps(vmm_src, vmm_src, vmm_mean);
        vmulps(vmm_src, vmm_src, vmm_inv_sqrtvar);
        vfmadd231ps(vmm_dgamma, vmm_src, vmm_ddst);
        jit_transfer_.template store<f32>(
                vmm_dbeta, reg_diff_beta, nelems, offt_elems);
        jit_transfer_.template store<f32>(
                vmm_dgamma, reg_diff_gamma, nelems, offt_elems);
    };

    for (int i = 0; i < C_vecs; i++)
        op(simd_w, i * simd_w);

    for (int i = utils::rnd_dn(C_, simd_w); i < C_; i++)
        op(1, i);

    postamble();
}

template <data_type_t data_type>
struct jit_diff_data_kernel_t : diff_data_kernel_t<data_type>,
                                public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(lnorm_utils::jit_diff_data_kernel_t);

    jit_diff_data_kernel_t(const layer_normalization_pd_t *pd);

    using data_t = typename prec_traits<data_type>::type;
    void operator()(const data_t *src, const data_t *diff_dst, data_t *diff_src,
            const float *ss, const float *mean,
            const float *var) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    jit_transfer_t<data_type> jit_transfer_;
    static constexpr int simd_w = data_type == bf16 ? 16 : 8;
    using Vmm = typename utils::conditional<data_type == bf16, Xbyak::Zmm,
            Xbyak::Ymm>::type;
    using diff_data_kernel_t<data_type>::C_;
    using diff_data_kernel_t<data_type>::eps_;
    using diff_data_kernel_t<data_type>::calculate_diff_stats_;
    using diff_data_kernel_t<data_type>::use_scaleshift_;

    struct ker_args_t {
        const data_t *src;
        const data_t *diff_dst;
        data_t *diff_src;
        const float *ss;
        const float *mean;
        const float *inv_sqrtvar;
    };
    void generate() override;

    void reduce(Vmm vmm_vec);

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_diff_src = rax;
    Xbyak::Reg64 reg_diff_dst = rbx;
    Xbyak::Reg64 reg_gamma = r11;
    Xbyak::Reg64 reg_tmp = r10;
    Xbyak::Reg64 reg_dd_gamma = r9;
    Xbyak::Reg64 reg_dd_gamma_x = r8;

    Xbyak::Xmm xmm_tmp = Xbyak::Xmm(7);

    Vmm vmm_C = Vmm(8);
    Vmm vmm_gamma = Vmm(9);
    Vmm vmm_inv_sqrtvar = Vmm(10);
    Vmm vmm_dsrc = Vmm(11);
    Vmm vmm_dd_gamma_x = Vmm(12);
    Vmm vmm_dd_gamma = Vmm(13);
    Vmm vmm_src = Vmm(14);
    Vmm vmm_mean = Vmm(15);
};

template <data_type_t data_type>
jit_diff_data_kernel_t<data_type>::jit_diff_data_kernel_t(
        const layer_normalization_pd_t *pd)
    : diff_data_kernel_t<data_type>(pd), jit_transfer_ {*this} {
    assert(data_type == bf16 ? mayiuse(avx512_core) : mayiuse(avx2));
}

template <data_type_t data_type>
void jit_diff_data_kernel_t<data_type>::operator()(const data_t *src,
        const data_t *diff_dst, data_t *diff_src, const float *ss,
        const float *mean, const float *var) const {
    ker_args_t args;
    args.src = src;
    args.diff_dst = diff_dst;
    args.diff_src = diff_src;
    args.ss = ss;
    args.mean = mean;
#ifdef __INTEL_COMPILER
    //Without volatile ICC with -O2 & -O3 optimizes out denominator from
    //inv_sqrtvar and computes 1/denom with lower precision
    const volatile float denom = sqrtf(*var + eps_);
#else
    const float denom = sqrtf(*var + eps_);
#endif
    const float inv_sqrtvar = 1.f / denom;
    args.inv_sqrtvar = &inv_sqrtvar;
    jit_generator::operator()(&args);
}

template <data_type_t data_type>
void jit_diff_data_kernel_t<data_type>::generate() {
    preamble();
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
    mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
    mov(reg_diff_src, ptr[reg_param + PARAM_OFF(diff_src)]);
    mov(reg_gamma, ptr[reg_param + PARAM_OFF(ss)]);

    if (calculate_diff_stats_) {
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
        vmovss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(vmm_mean, xmm_tmp);
    }

    mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
    vmovss(xmm_tmp, dword[reg_tmp]);
    vbroadcastss(vmm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF

    mov(reg_tmp, float2int(C_));
    uni_vmovq(xmm_tmp, reg_tmp);
    uni_vbroadcastss(vmm_C, xmm_tmp);

    const int C_vecs = C_ / simd_w;

    auto compute_dd_gammas = [=](int nelems, size_t offt_elems) {
        Vmm vmm_ddst = vmm_dsrc;
        jit_transfer_.template load<data_type>(
                vmm_ddst, reg_diff_dst, nelems, offt_elems);
        if (use_scaleshift_) {
            jit_transfer_.template load<f32>(
                    vmm_gamma, reg_gamma, nelems, offt_elems);
            vmulps(vmm_ddst, vmm_ddst, vmm_gamma);
        }
        jit_transfer_.template load<data_type>(
                vmm_src, reg_src, nelems, offt_elems);
        vaddps(vmm_dd_gamma, vmm_dd_gamma, vmm_ddst);
        vsubps(vmm_src, vmm_src, vmm_mean);
        vfmadd231ps(vmm_dd_gamma_x, vmm_ddst, vmm_src);
    };

    auto compute_diff_src = [=](int nelems, size_t offt_elems) {
        jit_transfer_.template load<data_type>(
                vmm_dsrc, reg_diff_dst, nelems, offt_elems);
        if (use_scaleshift_) {
            jit_transfer_.template load<f32>(
                    vmm_gamma, reg_gamma, nelems, offt_elems);
            vmulps(vmm_dsrc, vmm_dsrc, vmm_gamma);
        }
        if (calculate_diff_stats_) {
            jit_transfer_.template load<data_type>(
                    vmm_src, reg_src, nelems, offt_elems);
            vsubps(vmm_src, vmm_src, vmm_mean);
            vmulps(vmm_src, vmm_src, vmm_inv_sqrtvar);
            vfmadd213ps(vmm_src, vmm_dd_gamma_x, vmm_dd_gamma);
            vdivps(vmm_src, vmm_src, vmm_C);
            vsubps(vmm_dsrc, vmm_dsrc, vmm_src);
        }
        vmulps(vmm_dsrc, vmm_dsrc, vmm_inv_sqrtvar);
        jit_transfer_.template store<data_type>(
                vmm_dsrc, reg_diff_src, nelems, offt_elems);
    };

    if (calculate_diff_stats_) {
        uni_vpxor(vmm_dd_gamma, vmm_dd_gamma, vmm_dd_gamma);
        uni_vpxor(vmm_dd_gamma_x, vmm_dd_gamma_x, vmm_dd_gamma_x);

        for (int i = 0; i < C_vecs; i++)
            compute_dd_gammas(simd_w, i * simd_w);

        reduce(vmm_dd_gamma);
        reduce(vmm_dd_gamma_x);

        for (int i = utils::rnd_dn(C_, simd_w); i < C_; i++)
            compute_dd_gammas(1, i);

        vmulps(vmm_dd_gamma_x, vmm_dd_gamma_x, vmm_inv_sqrtvar);

        Xmm xmm_dd_gamma = Xmm(vmm_dd_gamma.getIdx());
        vbroadcastss(vmm_dd_gamma, xmm_dd_gamma);
        Xmm xmm_dd_gamma_x = Xmm(vmm_dd_gamma_x.getIdx());
        vbroadcastss(vmm_dd_gamma_x, xmm_dd_gamma_x);
    }

    for (int i = 0; i < C_vecs; i++)
        compute_diff_src(simd_w, i * simd_w);

    for (int i = utils::rnd_dn(C_, simd_w); i < C_; i++)
        compute_diff_src(1, i);

    postamble();
}

template <>
void jit_diff_data_kernel_t<f32>::reduce(
        jit_diff_data_kernel_t<f32>::Vmm ymm_vec) {
    vextractf128(xmm_tmp, ymm_vec, 1);
    Xmm xmm_vec = Xmm(ymm_vec.getIdx());
    vaddps(xmm_vec, xmm_tmp, xmm_vec);
    vhaddps(xmm_vec, xmm_vec, xmm_vec);
    vhaddps(xmm_vec, xmm_vec, xmm_vec);
};

template <>
void jit_diff_data_kernel_t<bf16>::reduce(
        jit_diff_data_kernel_t<bf16>::Vmm zmm_vec) {
    Xbyak::Ymm ymm_high = Ymm(xmm_tmp.getIdx());

    vextractf32x8(ymm_high, zmm_vec, 1);
    Ymm ymm_vec = Ymm(zmm_vec.getIdx());
    vaddps(ymm_vec, ymm_high, ymm_vec);
    vhaddps(ymm_vec, ymm_vec, ymm_vec);
    vhaddps(ymm_vec, ymm_vec, ymm_vec);

    Xmm xmm_high = Xmm(ymm_high.getIdx());
    Xmm xmm_vec = Xmm(zmm_vec.getIdx());
    vextractf128(xmm_high, ymm_vec, 1);
    vaddps(xmm_vec, xmm_high, xmm_vec);
};

template <>
statistics_kernel_t<bf16> *statistics_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx512_core) ? new jit_statistics_kernel_t<bf16>(pd)
                                : nullptr;
}

template <>
statistics_kernel_t<f32> *statistics_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx2) ? new jit_statistics_kernel_t<f32>(pd) : nullptr;
}

template <>
data_kernel_t<bf16> *data_kernel_create(const layer_normalization_pd_t *pd) {
    return mayiuse(avx512_core) ? new jit_data_kernel_t<bf16>(pd) : nullptr;
}

template <>
data_kernel_t<f32> *data_kernel_create(const layer_normalization_pd_t *pd) {
    return mayiuse(avx2) ? new jit_data_kernel_t<f32>(pd) : nullptr;
}

template <>
diff_ss_kernel_t<bf16> *diff_ss_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx512_core) ? new jit_diff_ss_kernel_t<bf16>(pd) : nullptr;
}

template <>
diff_ss_kernel_t<f32> *diff_ss_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx2) ? new jit_diff_ss_kernel_t<f32>(pd) : nullptr;
}

template <>
diff_data_kernel_t<bf16> *diff_data_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx512_core) ? new jit_diff_data_kernel_t<bf16>(pd)
                                : nullptr;
}

template <>
diff_data_kernel_t<f32> *diff_data_kernel_create(
        const layer_normalization_pd_t *pd) {
    return mayiuse(avx2) ? new jit_diff_data_kernel_t<f32>(pd) : nullptr;
}

} // namespace lnorm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
