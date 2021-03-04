/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_uni_batch_normalization_s8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

using namespace Xbyak;

using data_t = int8_t;

struct call_params_t {
    // keep int sizes at 8 bytes -- jit code expects this
    size_t channel_offt_count, spat_offt_count;
    float eps;
    const float *scale_shift, *mean, *var;
    const data_t *src, *dst;
};

template <cpu_isa_t isa>
struct jit_bnorm_base_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : ((isa == avx2) ? yword : zword);
    const int vlen = cpu_isa_traits<isa>::vlen;

    const batch_normalization_pd_t *pd_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_scale_shift = rbx;
    Reg64 reg_mean = rbp;

    Reg64 reg_channel_offt_count = r8;
    Reg64 reg_spat_offt = r9;
    Reg64 reg_spat_offt_count = r10;
    Reg64 reg_tmp = r11;
    Reg64 reg_src = r12;
    Reg64 reg_dst = r13;
    Reg64 reg_var = r14;
    Reg64 reg_channel_offt_1byte = r15;
    Reg64 reg_channel_offt_4byte = rax;

    Vmm vzero = Vmm(isa == avx512_core ? 29 : 13);
    Xmm xone = Xmm(14);
    Vmm vone = Vmm(isa == avx512_core ? 30 : 14);
    Vmm veps = Vmm(isa == avx512_core ? 31 : 15);

    size_t simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    size_t c_in_xmm_ = (isa == sse41) ? 8 : 16;
    size_t chan_data_offt_;
    size_t num_c_blocks_;
    size_t c_tail_;
    bool with_relu_;

    void compute_predefined_variables() {
        chan_data_offt_ = pd_->C() * sizeof(float);
        num_c_blocks_ = pd_->C() / c_in_xmm_;
        c_tail_ = pd_->C() % c_in_xmm_;
        with_relu_ = (pd_->with_relu_post_op() || pd_->fuse_norm_relu())
                && pd_->is_fwd();
    }

    void load_common_params() {
        mov(reg_tmp, float2int(1.0f));
        uni_vmovq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        uni_vbroadcastss(veps, vmmword[reg_param + PARAM_OFF(eps)]);
        uni_vpxor(vzero, vzero, vzero);

        mov(reg_channel_offt_count,
                ptr[reg_param + PARAM_OFF(channel_offt_count)]);
        mov(reg_spat_offt_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_scale_shift, ptr[reg_param + PARAM_OFF(scale_shift)]);
        mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
#undef PARAM_OFF
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + reg_channel_offt_4byte + offt];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + reg_channel_offt_4byte + offt];
    }

    Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_channel_offt_4byte + offt
                + 0 * chan_data_offt_];
    }

    Address shift_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_channel_offt_4byte + offt
                + 1 * chan_data_offt_];
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_spat_offt + offt];
    }

    virtual void prepare_tail_mask() {}
    virtual void load_mean_and_var(const Vmm &vmean, const Vmm &vsqrtvar,
            size_t offt, bool need_tail) {}
    virtual void load_scale_and_shift(const Vmm &vscale, const Vmm &vshift,
            size_t offt, bool need_tail) {}
    virtual void compute_dst(bool need_tail) {}

    // Precomputes vscale and vshift for following
    // `vdst = vscale * vsrc + vshift`
    void compute_vscaleshift(const Vmm &vscale, const Vmm &vshift,
            const Vmm &vmean, const Vmm &vsqrtvar, size_t offt,
            bool need_tail) {
        load_mean_and_var(vmean, vsqrtvar, offt, need_tail);
        uni_vaddps(vsqrtvar, vsqrtvar, veps);
        uni_vsqrtps(vsqrtvar, vsqrtvar);

        if (pd_->use_scaleshift()) {
            load_scale_and_shift(vscale, vshift, offt, need_tail);
            uni_vdivps(vscale, vscale, vsqrtvar);
            uni_vfnmadd231ps(vshift, vmean, vscale);
        } else {
            uni_vdivps(vscale, vone, vsqrtvar, vscale);
            uni_vmulps(vmean, vmean, vscale);
            uni_vsubps(vshift, vzero, vmean, vshift);
        }
    }

    void forward() {
        xor_(reg_channel_offt_1byte, reg_channel_offt_1byte);
        xor_(reg_channel_offt_4byte, reg_channel_offt_4byte);
        mov(reg_tmp, sizeof(data_t) * c_in_xmm_);

        if (num_c_blocks_) compute_dst(false);
        if (c_tail_) compute_dst(true);
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        preamble();
        compute_predefined_variables();
        load_common_params();
        prepare_tail_mask();
        forward();
        postamble();
    }

    jit_bnorm_base_t(const batch_normalization_pd_t *pd) : pd_(pd) {}
};

template <cpu_isa_t isa>
struct jit_bnorm_t;

template <>
struct jit_bnorm_t<avx512_core> : public jit_bnorm_base_t<avx512_core> {
    Opmask tail_opmask = Opmask(1); // f32 mask for channel math

    void prepare_tail_mask() override {
        if (!c_tail_) return;

        const int mask_f32 = (1 << c_tail_) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(tail_opmask, regw_tmp);
    }

    void load_mean_and_var(const Vmm &vmean, const Vmm &vsqrtvar, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            uni_vmovups_tail(vmean, tail_opmask, mean_ptr(offt));
            uni_vmovups_tail(vsqrtvar, tail_opmask, var_ptr(offt));
        } else {
            uni_vmovups(vmean, mean_ptr(offt));
            uni_vmovups(vsqrtvar, var_ptr(offt));
        }
    }

    void load_scale_and_shift(const Vmm &vscale, const Vmm &vshift, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            uni_vmovups_tail(vscale, tail_opmask, scale_ptr(offt));
            uni_vmovups_tail(vshift, tail_opmask, shift_ptr(offt));
        } else {
            uni_vmovups(vscale, scale_ptr(offt));
            uni_vmovups(vshift, shift_ptr(offt));
        }
    }

    void compute_dst(bool need_tail) override {
        Label c_loop;
        L(c_loop);
        {
            Xmm x = Xmm(0);
            Vmm v = Vmm(0);
            Vmm vscale = Vmm(1);
            Vmm vshift = Vmm(2);
            Vmm vmean = Vmm(3);
            Vmm vsqrtvar = Vmm(4);

            // compute single vscale and vshift vectors...
            compute_vscaleshift(vscale, vshift, vmean, vsqrtvar, 0, need_tail);

            // ... then process all spatial loop with it and move to the
            // next channel chunk
            mov(reg_spat_offt, reg_channel_offt_1byte);
            Label mb_sp_loop;
            L(mb_sp_loop);
            {
                if (need_tail) {
                    for (size_t tl = 0; tl < c_tail_; tl++)
                        vpinsrb(x, x, src_ptr(tl), tl);
                    vpmovsxbd(v, x);
                } else
                    vpmovsxbd(v, src_ptr());

                vcvtdq2ps(v, v);

                uni_vfmadd213ps(v, vscale, vshift);
                if (with_relu_) uni_vmaxps(v, v, vzero);

                vcvtps2dq(v, v);
                if (need_tail) {
                    vpmovsdb(x, v);
                    for (size_t tl = 0; tl < c_tail_; tl++)
                        vpextrb(dst_ptr(tl), x, tl);
                } else
                    vpmovsdb(dst_ptr(), v);

                add(reg_spat_offt, reg_channel_offt_count);
                cmp(reg_spat_offt, reg_spat_offt_count);
                jl(mb_sp_loop);
            }

            // reg_tmp checks c_in_xmm_ channels ahead for further tail process
            add(reg_tmp, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_1byte, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_4byte, sizeof(float) * c_in_xmm_);
            cmp(reg_tmp, reg_channel_offt_count);
            jle(c_loop);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd)
        : jit_bnorm_base_t<avx512_core>(pd) {}
};

template <>
struct jit_bnorm_t<avx2> : public jit_bnorm_base_t<avx2> {
    Vmm tail_vmask = Vmm(11);
    Vmm body_vmask = Vmm(12);

    void prepare_tail_mask() override {
        // tail is always < 16, process it with two parts
        static const uint32_t mask_half_ymm[8]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(&mask_half_ymm[0]));
        vmovups(body_vmask, ptr[reg_tmp]);

        if (!c_tail_) return;

        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp,
                reinterpret_cast<size_t>(&mask_f32[7 - c_tail_ % simd_w_]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void load_mean_and_var(const Vmm &vmean, const Vmm &vsqrtvar, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            uni_vmovups_tail(vmean, tail_vmask, mean_ptr(offt));
            uni_vmovups_tail(vsqrtvar, tail_vmask, var_ptr(offt));
        } else {
            uni_vmovups(vmean, mean_ptr(offt));
            uni_vmovups(vsqrtvar, var_ptr(offt));
        }
    }

    void load_scale_and_shift(const Vmm &vscale, const Vmm &vshift, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            uni_vmovups_tail(vscale, tail_vmask, scale_ptr(offt));
            uni_vmovups_tail(vshift, tail_vmask, shift_ptr(offt));
        } else {
            uni_vmovups(vscale, scale_ptr(offt));
            uni_vmovups(vshift, shift_ptr(offt));
        }
    }

    void compute_dst(bool need_tail) override {
        Label c_loop;
        L(c_loop);
        {

            Xmm x0 = Xmm(0);
            Vmm v0 = Vmm(0);
            Xmm x1 = Xmm(1);
            Vmm v1 = Vmm(1);
            Vmm vscale0 = Vmm(2);
            Vmm vshift0 = Vmm(3);
            Vmm vmean0 = Vmm(4);
            Vmm vsqrtvar0 = Vmm(5);
            Vmm vscale1 = Vmm(6);
            Vmm vshift1 = Vmm(7);
            Vmm vmean1 = Vmm(8);
            Vmm vsqrtvar1 = Vmm(9);

            // compute couple vscale and vshift vectors each of 8 channels...
            compute_vscaleshift(vscale0, vshift0, vmean0, vsqrtvar0, 0,
                    (c_tail_ < simd_w_ && need_tail) ? true : false);
            if (!need_tail || c_tail_ > simd_w_) {
                compute_vscaleshift(vscale1, vshift1, vmean1, vsqrtvar1,
                        simd_w_ * sizeof(float), need_tail);
            }

            // ... then process all spatial loop with it and move to the
            // next channel chunk
            mov(reg_spat_offt, reg_channel_offt_1byte);
            Label mb_sp_loop;
            L(mb_sp_loop);
            {

                if (need_tail) {
                    for (size_t tl = 0; tl < c_tail_; tl++) {
                        if (tl < simd_w_) {
                            vpinsrb(x0, x0, src_ptr(tl), tl);
                        } else {
                            vpinsrb(x1, x1, src_ptr(tl), tl - simd_w_);
                        }
                    }
                    vpmovsxbd(v0, x0);
                    vpmovsxbd(v1, x1);
                } else {
                    vpmovsxbd(v0, src_ptr());
                    vpmovsxbd(v1, src_ptr(simd_w_));
                }

                vcvtdq2ps(v0, v0);
                vcvtdq2ps(v1, v1);

                uni_vfmadd213ps(v0, vscale0, vshift0);
                uni_vfmadd213ps(v1, vscale1, vshift1);
                if (with_relu_) {
                    uni_vmaxps(v0, v0, vzero);
                    uni_vmaxps(v1, v1, vzero);
                }

                vcvtps2dq(v0, v0); // BA
                vcvtps2dq(v1, v1); // DC
                vpackssdw(v0, v0, v1); // BA + DC -> DBCA
                vpermq(v0, v0, 0xD8); // DBCA -> DCBA
                vperm2i128(v1, v0, v0, 0x1); // DCBA -> BADC
                vpacksswb(v0, v0, v1); // DCBA + BADC -> badcDCBA

                if (need_tail) {
                    for (size_t tl = 0; tl < c_tail_; tl++) {
                        vpextrb(dst_ptr(tl), x0, tl);
                    }
                } else {
                    // due to vpacksswb produces 32 integers in ymm, and top
                    // half of them are garbage, do 128-b masked store
                    vmaskmovps(dst_ptr(), body_vmask, v0);
                }

                add(reg_spat_offt, reg_channel_offt_count);
                cmp(reg_spat_offt, reg_spat_offt_count);
                jl(mb_sp_loop);
            }

            // reg_tmp checks c_in_xmm_ channels ahead for further tail process
            add(reg_tmp, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_1byte, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_4byte, sizeof(float) * c_in_xmm_);
            cmp(reg_tmp, reg_channel_offt_count);
            jle(c_loop);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd)
        : jit_bnorm_base_t<avx2>(pd) {}
};

template <>
struct jit_bnorm_t<sse41> : public jit_bnorm_base_t<sse41> {
    void load_mean_and_var(const Vmm &vmean, const Vmm &vsqrtvar, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            for (size_t tl = 0; tl < c_tail_ % simd_w_; tl++) {
                pinsrd(vmean, mean_ptr(offt + tl * sizeof(float)), tl);
                pinsrd(vsqrtvar, var_ptr(offt + tl * sizeof(float)), tl);
            }
        } else {
            movups(vmean, mean_ptr(offt));
            movups(vsqrtvar, var_ptr(offt));
        }
    }

    void load_scale_and_shift(const Vmm &vscale, const Vmm &vshift, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            for (size_t tl = 0; tl < c_tail_ % simd_w_; tl++) {
                pinsrd(vscale, scale_ptr(offt + tl * sizeof(float)), tl);
                pinsrd(vshift, shift_ptr(offt + tl * sizeof(float)), tl);
            }
        } else {
            movups(vscale, scale_ptr(offt));
            movups(vshift, shift_ptr(offt));
        }
    }

    void compute_dst(bool need_tail) override {
        const size_t copy_range = need_tail ? c_tail_ : c_in_xmm_;
        Label c_loop;
        L(c_loop);
        {

            Vmm v0 = Vmm(0);
            Vmm v1 = Vmm(1);
            Vmm vscale0 = Vmm(2);
            Vmm vshift0 = Vmm(3);
            Vmm vmean0 = Vmm(4);
            Vmm vsqrtvar0 = Vmm(5);
            Vmm vscale1 = Vmm(6);
            Vmm vshift1 = Vmm(7);
            Vmm vmean1 = Vmm(8);
            Vmm vsqrtvar1 = Vmm(9);

            // compute couple vscale and vshift vectors each of 8 channels...
            compute_vscaleshift(vscale0, vshift0, vmean0, vsqrtvar0, 0,
                    (c_tail_ < simd_w_ && need_tail) ? true : false);
            if (!need_tail || c_tail_ > simd_w_) {
                compute_vscaleshift(vscale1, vshift1, vmean1, vsqrtvar1,
                        simd_w_ * sizeof(float), need_tail);
            }

            // ... then process all spatial loop with it and move to the
            // next channel chunk
            mov(reg_spat_offt, reg_channel_offt_1byte);
            Label mb_sp_loop;
            L(mb_sp_loop);
            {
                if (need_tail) {
                    for (size_t tl = 0; tl < copy_range; tl++) {
                        if (tl < simd_w_) {
                            pinsrb(v0, src_ptr(tl), tl);
                        } else {
                            pinsrb(v1, src_ptr(tl), (tl - simd_w_));
                        }
                    }
                    pmovsxbd(v0, v0);
                    pmovsxbd(v1, v1);
                } else {
                    pmovsxbd(v0, src_ptr());
                    pmovsxbd(v1, src_ptr(simd_w_));
                }

                cvtdq2ps(v0, v0);
                cvtdq2ps(v1, v1);

                uni_vfmadd213ps(v0, vscale0, vshift0);
                uni_vfmadd213ps(v1, vscale1, vshift1);
                if (with_relu_) {
                    maxps(v0, vzero);
                    maxps(v1, vzero);
                }

                cvtps2dq(v0, v0);
                cvtps2dq(v1, v1);
                packssdw(v0, v1);
                movups(v1, v0);
                packsswb(v0, v1);

                // Potential perf gain is possible if combining two halves
                // into a single vector register and use movups instead
                // of byte stores.
                for (size_t tl = 0; tl < copy_range; tl++) {
                    pextrb(dst_ptr(tl), v0, tl);
                }

                add(reg_spat_offt, reg_channel_offt_count);
                cmp(reg_spat_offt, reg_spat_offt_count);
                jl(mb_sp_loop);
            }

            // reg_tmp checks c_in_xmm_ channels ahead for further tail process
            add(reg_tmp, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_1byte, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_4byte, sizeof(float) * c_in_xmm_);
            cmp(reg_tmp, reg_channel_offt_count);
            jle(c_loop);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd)
        : jit_bnorm_base_t<sse41>(pd) {}
};

} // namespace

namespace bnorm_s8_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *pd) : pd_(pd), ker_(pd_) {}
    ~driver_t() = default;

    // TODO: for problems where thread pieces don't fit L2 cache, add spatial
    // re-balance using less pieces.
    void exec(int ithr, int nthr, const data_t *src, data_t *dst,
            const float *scale_shift, const float *mean, const float *var) {
        dim_t N = pd_->MB();
        dim_t C = pd_->C();
        dim_t D = pd_->D();
        dim_t H = pd_->H();
        dim_t W = pd_->W();
        dim_t SP = D * H * W;

        call_params_t p;

        p.eps = pd_->desc()->batch_norm_epsilon;

        p.scale_shift = scale_shift;
        p.mean = mean;
        p.var = var;

        dim_t work_amount {N * SP}, start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        p.channel_offt_count = C;
        p.spat_offt_count = (end - start) * p.channel_offt_count;
        p.src = src + start * p.channel_offt_count;
        p.dst = dst + start * p.channel_offt_count;

        if (p.spat_offt_count != 0) ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const batch_normalization_pd_t *pd_;

    jit_bnorm_t<isa> ker_;
};

} // namespace bnorm_s8_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    auto desired_fmt_tag = (ndims() == 4) ? nhwc : ndhwc;

    bool ok = true && mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
            && one_of(ndims(), 4, 5) && stats_is_src()
            && src_md()->data_type == s8 && check_scale_shift_data_type()
            && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<isa>::jit_uni_batch_normalization_s8_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_s8_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scale_shift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);
    auto mean = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN));
    auto var
            = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE));
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    // do sequential if the problem is less than one 4K memory page
    const bool force_sequential
            = pd()->MB() * pd()->C() * pd()->D() * pd()->H() * pd()->W()
            <= 4096;

    parallel(force_sequential ? 1 : 0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, dst, scale_shift, mean, var);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<
        isa>::~jit_uni_batch_normalization_s8_fwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_s8_fwd_t<avx512_core>;
template struct jit_uni_batch_normalization_s8_fwd_t<avx2>;
template struct jit_uni_batch_normalization_s8_fwd_t<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
