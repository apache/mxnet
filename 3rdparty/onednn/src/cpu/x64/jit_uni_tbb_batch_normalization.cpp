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
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_uni_tbb_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

using namespace memory_tracking::names;
using namespace Xbyak;
using acc_data_t = float;

#define PARAM_ADDR(x) (reg_param + offsetof(call_params_t, x))
template <cpu_isa_t isa>
struct jit_bnorm_process_tail_t {
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    jit_bnorm_process_tail_t(const batch_normalization_pd_t *bdesc,
            jit_generator *host, Reg64 reg_tmp, Reg64 reg_blk_has_tail,
            Reg64 reg_C, Vmm vtail_mask, Opmask ktail_mask)
        : h(host)
        , reg_tmp_(reg_tmp)
        , reg_blk_has_tail_(reg_blk_has_tail)
        , reg_C_(reg_C)
        , vtail_mask_(vtail_mask)
        , ktail_mask_(ktail_mask) {
        const memory_desc_wrapper data_d(bdesc->src_md());
        c_is_padded = bdesc->C() != data_d.padded_dims()[1];

        const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
        tail = bdesc->C() % (int)(vlen / sizeof(float));
    }

    jit_generator *const h;
    Reg64 reg_tmp_;
    Reg64 reg_blk_has_tail_;
    Reg64 reg_C_;
    Vmm vtail_mask_;
    Opmask ktail_mask_;
    bool c_is_padded;
    int tail;

    void prepare_tail_mask_avx512_common() {
        if (!c_is_padded) return;

        const int mask = (1 << tail) - 1;

        Reg32 regw_tmp = reg_tmp_.cvt32();
        h->mov(regw_tmp, mask);
        h->kmovw(ktail_mask_, regw_tmp);
    }

    void prepare_tail_mask_avx2_common() {
        if (!c_is_padded) return;

        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};

        h->mov(reg_tmp_, reinterpret_cast<size_t>(&mask[8 - tail]));
        h->vmovups(vtail_mask_, h->ptr[reg_tmp_]);
    }

    void prepare_tail() {
        if (isa == avx512_common)
            prepare_tail_mask_avx512_common();
        else if (isa == avx2)
            prepare_tail_mask_avx2_common();
    }

    void uni_vmovups_tail_avx2_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM()) {
            h->vmaskmovps(dst.getAddress(), vtail_mask_, Vmm(src.getIdx()));
        } else {
            h->vmaskmovps(Vmm(dst.getIdx()), vtail_mask_, src.getAddress());
        }
        h->jmp(l_ret);
    }

    void uni_vmovups_tail_avx512_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM())
            h->uni_vmovups(
                    dst.getAddress() | ktail_mask_ | h->T_z, Vmm(src.getIdx()));
        else
            h->uni_vmovups(
                    Vmm(dst.getIdx()) | ktail_mask_ | h->T_z, src.getAddress());

        h->jmp(l_ret);
    }

    void uni_vmovups_maybe_tail(const Operand &dst, const Operand &src) {
        Label l_no_mask, l_ret;
        if (c_is_padded) {
            h->cmp(reg_blk_has_tail_, 0);
            h->jz(l_no_mask);

            h->cmp(reg_C_, 1);
            h->jne(l_no_mask);
            assert(isa == avx512_common || isa == avx2);
            if (isa == avx512_common)
                uni_vmovups_tail_avx512_common(dst, src, l_ret);
            else if (isa == avx2)
                uni_vmovups_tail_avx2_common(dst, src, l_ret);
        }
        h->L(l_no_mask);
        if (dst.isMEM())
            h->uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
        else
            h->uni_vmovups(Vmm(dst.getIdx()), src.getAddress());

        h->L(l_ret);
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_process_relu_t {
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    jit_bnorm_process_relu_t(const batch_normalization_pd_t *bdesc,
            jit_generator *host, Reg64 reg_off_dat, Reg64 reg_tmp,
            Reg64 reg_ptr_ws, Vmm vzero, Vmm vstore_mask, Opmask kstore_mask)
        : h(host)
        , reg_off_dat_(reg_off_dat)
        , reg_tmp_(reg_tmp)
        , reg_ptr_ws_(reg_ptr_ws)
        , vzero_(vzero)
        , vstore_mask_(vstore_mask)
        , kstore_mask_(kstore_mask) {
        with_relu = bdesc->with_relu_post_op() || bdesc->fuse_norm_relu();
        with_relu_inf_only = with_relu
                && !(bdesc->fuse_norm_relu() && bdesc->is_training());
        is_bf16_ = bdesc->desc()->data_desc.data_type == data_type::bf16;
        bit_shift = 5 - is_bf16_;
    }

    jit_generator *const h;
    Reg64 reg_off_dat_;
    Reg64 reg_tmp_;
    Reg64 reg_ptr_ws_;
    Vmm vzero_, vstore_mask_;
    Opmask kstore_mask_;
    Label l_relu_mask_avx2;
    bool with_relu, with_relu_inf_only;
    int bit_shift;
    bool is_bf16_;

    void fwd_prepare_relu() {
        if (with_relu) { h->uni_vpxor(vzero_, vzero_, vzero_); }
    }

    void bwd_prepare_relu() {
        if (with_relu) {
            h->uni_vpxor(vzero_, vzero_, vzero_);
            if (isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        h->jmp(l_mask_after);
        h->align(32);
        h->L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i)
            h->dd(1 << i);
        h->L(l_mask_after);
    }

    void fwd_process_relu(Vmm v) {
        if (with_relu_inf_only) {
            h->uni_vmaxps(v, v, vzero_);
        } else if (with_relu) {
            if (isa == avx512_common)
                fwd_process_relu_avx512_common(v);
            else if (isa == avx2)
                fwd_process_relu_avx2(v);
            else
                assert(false);
        }
    }

    void bwd_process_relu(Vmm v) {
        if (with_relu) {
            if (isa == avx512_common)
                bwd_process_relu_avx512_common(v);
            else if (isa == avx2)
                bwd_process_relu_avx2(v);
            else
                assert(false);
        }
    }

    void fwd_process_relu_avx2(Vmm vdst) {
        Reg64 reg_store_mask = reg_tmp_;
        h->shr(reg_off_dat_, bit_shift);
        h->vcmpps(vstore_mask_, vzero_, vdst, jit_generator::_cmp_lt_os);
        h->vmovmskps(reg_store_mask, vstore_mask_);
        h->mov(h->ptr[reg_ptr_ws_ + reg_off_dat_], reg_store_mask.cvt8());
        h->vblendvps(vdst, vzero_, vdst, vstore_mask_);
        h->shl(reg_off_dat_, bit_shift);
    }

    void fwd_process_relu_avx512_common(Vmm vdst) {
        h->shr(reg_off_dat_, bit_shift);
        h->vcmpps(kstore_mask_, vzero_, vdst, jit_generator::_cmp_lt_os);
        h->kmovw(h->ptr[reg_ptr_ws_ + reg_off_dat_], kstore_mask_);
        h->vblendmps(vdst | kstore_mask_, vzero_, vdst);
        h->shl(reg_off_dat_, bit_shift);
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst) {
        h->shr(reg_off_dat_, bit_shift);
        h->vpbroadcastb(vstore_mask_, h->ptr[reg_ptr_ws_ + reg_off_dat_]);
        h->vpand(vstore_mask_, vstore_mask_,
                h->ptr[Xbyak::util::rip + l_relu_mask_avx2]);
        h->vpcmpeqd(vstore_mask_, vstore_mask_,
                h->ptr[Xbyak::util::rip + l_relu_mask_avx2]);
        h->vblendvps(vdiff_dst, vzero_, vdiff_dst, vstore_mask_);
        h->shl(reg_off_dat_, bit_shift);
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst) {
        h->shr(reg_off_dat_, bit_shift);
        h->kmovw(kstore_mask_, h->ptr[reg_ptr_ws_ + reg_off_dat_]);
        h->vmovups(vdiff_dst | kstore_mask_ | h->T_z, vdiff_dst);
        h->shl(reg_off_dat_, bit_shift);
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bf16_emulation_t {
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    jit_bnorm_bf16_emulation_t(const batch_normalization_pd_t *bdesc,
            jit_generator *host, Zmm zmm_reserved_1, Zmm zmm_reserved_2,
            Zmm zmm_reserved_3, Zmm zmm_reserved_4, Reg64 reg_tmp)
        : h(host) {
        is_bf16_ = bdesc->desc()->data_desc.data_type == data_type::bf16;
        if (is_bf16_ && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(h, zmm_reserved_1, zmm_reserved_2,
                    zmm_reserved_3, reg_tmp, zmm_reserved_4, zmm_reserved_4);
            bf16_emu_->init_vcvtneps2bf16();
        }
    }
    ~jit_bnorm_bf16_emulation_t() { delete bf16_emu_; }

    jit_generator *const h;
    bf16_emulation_t *bf16_emu_ = nullptr;
    bool is_bf16_;

    void uni_vmovups_data(const Operand &dst, const Operand &src) {
        if (dst.isMEM()) {
            if (is_bf16_) {
                constexpr bool isAvx2 = isa == avx2;
                const typename std::conditional<isAvx2, Xmm, Ymm>::type
                        dst_reg {src.getIdx()};
                const typename std::conditional<isAvx2, Ymm, Zmm>::type
                        src_reg {src.getIdx()};

                // convert f32 output to bf16
                if (mayiuse(avx512_core_bf16))
                    h->vcvtneps2bf16(dst_reg, src_reg);
                else
                    bf16_emu_->vcvtneps2bf16(dst_reg, src_reg);

                h->vmovdqu16(dst.getAddress(), dst_reg);
            } else {
                h->uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
            }
        } else {
            if (is_bf16_) {
                // convert bf16 input to f32
                h->vpmovzxwd(Vmm(dst.getIdx()), src.getAddress());
                h->vpslld(Vmm(dst.getIdx()), Vmm(dst.getIdx()), 0x10);
            } else {
                h->uni_vmovups(Vmm(dst.getIdx()), src.getAddress());
            }
        }
    }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_bnorm_bf16_emulation_t);
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_statistics_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_fwd_statistics_t)
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    const int simd_w = vlen / sizeof(acc_data_t);
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src;
        const acc_data_t *mean;
        const acc_data_t *var;
        size_t blk_has_tail;
        size_t do_normalise;
    };

    Reg64 reg_param = abi_param1;
    Reg64 reg_tmp = abi_not_param1;
    Reg64 reg_N = rsi;
    Reg64 reg_S = rax;
    Reg64 reg_C = rdx;
    Reg64 reg_off_c = rbx;
    Reg64 reg_blk_has_tail = rbp;

    Reg64 reg_off_dat = r8;
    Reg64 reg_off_dat_save = r9;
    Reg64 reg_ptr_mean = r10;
    Reg64 reg_ptr_var = r11;
    Reg64 reg_ptr_src = r12;
    Reg64 reg_do_normalise = r13;
    Reg64 reg_ptr_stat = r14;

    Vmm vzero = Vmm(0);
    Vmm vmean = Vmm(1);
    Vmm vstat = Vmm(2);
    Vmm v = Vmm(3);
    Vmm vtail_mask = Vmm(4);
    Vmm vNS = Vmm(5);
    Vmm vtmp = Vmm(6);
    Vmm vtmp1 = Vmm(7);

    Opmask ktail_mask = Opmask(2);

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_bf16_emulation_t<isa> jit_bf16_emu_;
    int stride_N_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_mean, PARAM_PTR(mean));
        mov(reg_ptr_var, PARAM_PTR(var));
#undef PARAM_PTR
        mov(reg_blk_has_tail, dword[PARAM_ADDR(blk_has_tail)]);
        mov(reg_do_normalise, dword[PARAM_ADDR(do_normalise)]);
    }

    void zeroise() {
        Label label_zeroise;
        xor_(reg_off_c, reg_off_c);
        uni_vpxor(vzero, vzero, vzero);
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_zeroise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat + reg_off_c], vzero);
            if (isa == sse41) {
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_stat + reg_off_c + vlen / 2], vzero);
            }
            add(reg_off_c, simd_w * acc_type_size_);
            dec(reg_C);
            jnz(label_zeroise);
        }
    }

    void compute_channels(bool compute_mean) {
        Label label_C, label_S;
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat, reg_off_dat_save);
            jit_tail_.uni_vmovups_maybe_tail(
                    vstat, vmmword[reg_ptr_stat + reg_off_c]);

            if (!compute_mean)
                jit_tail_.uni_vmovups_maybe_tail(
                        vmean, vmmword[reg_ptr_mean + reg_off_c]);

            mov(reg_S, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                jit_bf16_emu_.uni_vmovups_data(
                        v, vmmword[reg_ptr_src + reg_off_dat]);

                if (compute_mean) {
                    uni_vaddps(vstat, vstat, v);
                } else {
                    uni_vsubps(vtmp, v, vmean, vtmp1);
                    uni_vfmadd231ps(vstat, vtmp, vtmp);
                }

                add(reg_off_dat, simd_w * data_type_size_);

                dec(reg_S);
                jnz(label_S);
            }
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat + reg_off_c], vstat);

            add(reg_off_dat_save, stride_C_ * data_type_size_);
            add(reg_off_c, simd_w * acc_type_size_);

            dec(reg_C);
            jnz(label_C);
        }
    }

    void compute(bool compute_mean) {
        Label label_N;
        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            compute_channels(compute_mean);

            if (isa == sse41) {
                xor_(reg_off_dat_save, reg_off_dat_save);
                xor_(reg_off_c, reg_off_c);
                add(reg_off_dat_save, vlen / 2);
                add(reg_off_c, vlen / 2);

                compute_channels(compute_mean);
            }

            add(reg_ptr_src, stride_N_ * data_type_size_);
            dec(reg_N);
            jnz(label_N);
        }
    }

    void normalize() {
        Label label_ret, label_normalise;
        cmp(reg_do_normalise, 0);
        jz(label_ret);

        const int S = bdesc_->D() * bdesc_->H() * bdesc_->W();
        mov(reg_tmp, float2int(bdesc_->MB() * S));
        Xmm xtmp = Xmm(vtmp.getIdx());
        uni_vmovq(xtmp, reg_tmp);
        uni_vbroadcastss(vNS, xtmp);

        xor_(reg_off_c, reg_off_c);
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_normalise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    v, vmmword[reg_ptr_stat + reg_off_c]);
            uni_vdivps(v, v, vNS);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat + reg_off_c], v);

            if (isa == sse41) {
                jit_tail_.uni_vmovups_maybe_tail(
                        v, vmmword[reg_ptr_stat + reg_off_c + vlen / 2]);
                uni_vdivps(v, v, vNS);
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_stat + reg_off_c + vlen / 2], v);
            }

            add(reg_off_c, simd_w * acc_type_size_);
            dec(reg_C);
            jnz(label_normalise);
        }

        L(label_ret);
    }

    jit_bnorm_fwd_statistics_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc)
        , jit_tail_(bdesc, this, reg_tmp, reg_blk_has_tail, reg_C, vtail_mask,
                  ktail_mask)
        , jit_bf16_emu_(
                  bdesc, this, Zmm(16), Zmm(17), Zmm(18), Zmm(19), reg_tmp) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_common,
                "unsupported isa");
        stride_C_ = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int C_PADDED = bdesc_->src_md()->padded_dims[1];
        stride_N_ = (C_PADDED / simd_w) * stride_C_;
        data_type_size_
                = types::data_type_size(bdesc->desc()->data_desc.data_type);
        acc_type_size_ = sizeof(acc_data_t);
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_mean_t : jit_bnorm_fwd_statistics_t<isa> {
    using call_params_t =
            typename jit_bnorm_fwd_statistics_t<isa>::call_params_t;

    jit_bnorm_fwd_mean_t(const batch_normalization_pd_t *bdesc)
        : jit_bnorm_fwd_statistics_t<isa>(bdesc) {}

    void generate() override {
        this->preamble();
        this->load_common_params();
        this->mov(this->reg_ptr_stat, this->reg_ptr_mean);
        this->jit_tail_.prepare_tail();
        this->zeroise();
        this->compute(true);
        this->normalize();
        this->postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_var_t : jit_bnorm_fwd_statistics_t<isa> {
    using call_params_t =
            typename jit_bnorm_fwd_statistics_t<isa>::call_params_t;

    jit_bnorm_fwd_var_t(const batch_normalization_pd_t *bdesc)
        : jit_bnorm_fwd_statistics_t<isa>(bdesc) {}

    void generate() override {
        this->preamble();
        this->load_common_params();
        this->mov(this->reg_ptr_stat, this->reg_ptr_var);
        this->jit_tail_.prepare_tail();
        this->zeroise();
        this->compute(false);
        this->normalize();
        this->postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_fwd_t)
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    const int simd_w = vlen / sizeof(acc_data_t);
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src, *dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *scale_shift;
        size_t blk_has_tail;
    };

    Reg64 reg_param = abi_param1;
    Reg64 reg_tmp = abi_not_param1;
    Reg64 reg_N = rsi;
    Reg64 reg_S = rax;
    Reg64 reg_C = rdx;
    Reg64 reg_off_c = rbx;
    Reg64 reg_blk_has_tail = rbp;

    Reg64 reg_off_dat = r8;
    Reg64 reg_off_dat_save = r9;
    Reg64 reg_ptr_ws = r10;
    Reg64 reg_ptr_scale_shift = r11;
    Reg64 reg_ptr_var = r12;
    Reg64 reg_ptr_mean = r13;
    Reg64 reg_ptr_dst = r14;
    Reg64 reg_ptr_src = r15;

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);
    Vmm vmean = Vmm(2);
    Vmm vvar = Vmm(3);
    Vmm vsqrtvar = Vmm(4);
    Vmm vgamma = Vmm(5);
    Vmm vbeta = Vmm(6);
    Vmm veps = Vmm(7);
    Vmm vtmp = Vmm(8);
    Vmm v = Vmm(9);
    Vmm vtail_mask = Vmm(10);
    Vmm vstore_mask = vtmp;

    Opmask kstore_mask = Opmask(1);
    Opmask ktail_mask = Opmask(2);

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    jit_bnorm_bf16_emulation_t<isa> jit_bf16_emu_;
    int stride_N_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_dst, PARAM_PTR(dst));
        mov(reg_ptr_mean, PARAM_PTR(mean));
        mov(reg_ptr_var, PARAM_PTR(var));
        mov(reg_ptr_scale_shift, PARAM_PTR(scale_shift));
        mov(reg_ptr_ws, PARAM_PTR(ws));
#undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(vone, x);

        mov(reg_blk_has_tail, dword[PARAM_ADDR(blk_has_tail)]);
    }

    void compute_channels(bool stream_store_allowed) {
        Label label_C, label_S;
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat, reg_off_dat_save);

            jit_tail_.uni_vmovups_maybe_tail(
                    vmean, vmmword[reg_ptr_mean + reg_off_c]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vvar, vmmword[reg_ptr_var + reg_off_c]);

            uni_vmovups(vsqrtvar, vvar);
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);

            if (isa == sse41) {
                movups(vtmp, vone);
                divps(vtmp, vsqrtvar);
                movups(vsqrtvar, vtmp);
            } else
                vdivps(vsqrtvar, vone, vsqrtvar);

            if (bdesc_->use_scaleshift()) {
                jit_tail_.uni_vmovups_maybe_tail(
                        vgamma, vmmword[reg_ptr_scale_shift + reg_off_c]);
                int beta_off = bdesc_->C() * acc_type_size_;
                jit_tail_.uni_vmovups_maybe_tail(vbeta,
                        vmmword[reg_ptr_scale_shift + reg_off_c + beta_off]);
            }

            mov(reg_S, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                jit_bf16_emu_.uni_vmovups_data(
                        v, vmmword[reg_ptr_src + reg_off_dat]);
                uni_vsubps(v, v, vmean);
                uni_vmulps(v, v, vsqrtvar);

                if (bdesc_->use_scaleshift()) uni_vfmadd213ps(v, vgamma, vbeta);

                jit_relu_.fwd_process_relu(v);

                if (stream_store_allowed) {
                    uni_vmovntps(vmmword[reg_ptr_dst + reg_off_dat], v);
                } else {
                    jit_bf16_emu_.uni_vmovups_data(
                            vmmword[reg_ptr_dst + reg_off_dat], v);
                }

                add(reg_off_dat, simd_w * data_type_size_);

                dec(reg_S);
                jnz(label_S);
            }

            add(reg_off_dat_save, stride_C_ * data_type_size_);
            add(reg_off_c, simd_w * acc_type_size_);

            dec(reg_C);
            jnz(label_C);
        }
    }

    void compute(bool stream_store_allowed) {
        Label label_N;
        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            compute_channels(stream_store_allowed);

            if (isa == sse41) {
                xor_(reg_off_dat_save, reg_off_dat_save);
                xor_(reg_off_c, reg_off_c);
                add(reg_off_dat_save, vlen / 2);
                add(reg_off_c, vlen / 2);

                compute_channels(stream_store_allowed);
            }

            add(reg_ptr_src, stride_N_ * data_type_size_);
            add(reg_ptr_dst, stride_N_ * data_type_size_);
            add(reg_ptr_ws, stride_N_ / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    jit_bnorm_fwd_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc)
        , jit_tail_(bdesc, this, reg_tmp, reg_blk_has_tail, reg_C, vtail_mask,
                  ktail_mask)
        , jit_relu_(bdesc, this, reg_off_dat, reg_tmp, reg_ptr_ws, vzero,
                  vstore_mask, kstore_mask)
        , jit_bf16_emu_(
                  bdesc, this, Zmm(16), Zmm(17), Zmm(18), Zmm(19), reg_tmp) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_common,
                "unsupported isa");
        stride_C_ = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int C_PADDED = bdesc_->src_md()->padded_dims[1];
        stride_N_ = (C_PADDED / simd_w) * stride_C_;
        data_type_size_
                = types::data_type_size(bdesc->desc()->data_desc.data_type);
        acc_type_size_ = sizeof(acc_data_t);
    }

    void generate() override {
        bool is_bf16 = bdesc_->desc()->data_desc.data_type == data_type::bf16;
        preamble();
        load_common_params();
        jit_relu_.fwd_prepare_relu();
        jit_tail_.prepare_tail();

        Label normal_store, end_store;
        test(reg_ptr_dst, vlen - 1);
        jnz(normal_store, T_NEAR);
        compute(!is_bf16);
        jmp(end_store, T_NEAR);
        L(normal_store);
        { compute(false); }
        L(end_store);

        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_t)
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    const int simd_w = vlen / sizeof(acc_data_t);
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src, *diff_src, *diff_dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *scale_shift, *diff_scale_shift;
        size_t blk_has_tail;
    };

    Reg64 reg_param = abi_param1;
    Reg64 reg_tmp = abi_not_param1;
    Reg64 reg_N = rsi;
    Reg64 reg_S = rax;
    Reg64 reg_C = rdx;
    Reg64 reg_off_c = rbx;
    Reg64 reg_blk_has_tail = rbp;

    Reg64 reg_off_dat = r8;
    Reg64 reg_off_dat_save = r9;
    Reg64 reg_ptr_c = r10;
    Reg64 reg_ptr_ws = r11;
    Reg64 reg_ptr_diff_dst = r12;
    Reg64 reg_ptr_diff_src = r13;
    Reg64 reg_ptr_src = r14;

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);
    Vmm vmean = Vmm(2);
    Vmm vsqrtvar = Vmm(3);
    Vmm vgamma = Vmm(4);
    Vmm vdiff_gamma = Vmm(5);
    Vmm vdiff_beta = Vmm(6);
    Vmm veps = Vmm(7);
    Vmm vNS = Vmm(8);
    Vmm vtmp = Vmm(9);
    Vmm v = Vmm(10);
    Vmm vtail_mask = Vmm(11);
    Vmm vstore_mask = vtmp;

    Opmask kstore_mask = Opmask(1);
    Opmask ktail_mask = Opmask(2);

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    jit_bnorm_bf16_emulation_t<isa> jit_bf16_emu_;
    int stride_N_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_diff_src, PARAM_PTR(diff_src));
        mov(reg_ptr_diff_dst, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws, PARAM_PTR(ws));
#undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(vone, x);

        const int S = bdesc_->D() * bdesc_->H() * bdesc_->W();
        mov(reg_tmp, float2int(bdesc_->MB() * S));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(vNS, x);

        mov(reg_blk_has_tail, dword[PARAM_ADDR(blk_has_tail)]);
    }

    void load_c_specifics() {
        mov(reg_ptr_c, ptr[PARAM_ADDR(mean)]);
        jit_tail_.uni_vmovups_maybe_tail(vmean, vmmword[reg_ptr_c + reg_off_c]);

        mov(reg_ptr_c, ptr[PARAM_ADDR(var)]);
        jit_tail_.uni_vmovups_maybe_tail(
                vsqrtvar, vmmword[reg_ptr_c + reg_off_c]);
        uni_vaddps(vsqrtvar, vsqrtvar, veps);
        uni_vsqrtps(vsqrtvar, vsqrtvar);

        if (isa == sse41) {
            movups(vtmp, vone);
            divps(vtmp, vsqrtvar);
            movups(vsqrtvar, vtmp);
        } else
            vdivps(vsqrtvar, vone, vsqrtvar);

        if (bdesc_->use_scaleshift()) {
            mov(reg_ptr_c, ptr[PARAM_ADDR(scale_shift)]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vgamma, vmmword[reg_ptr_c + reg_off_c]);
        }

        if (calculate_diff_stats()) {
            mov(reg_ptr_c, ptr[PARAM_ADDR(diff_scale_shift)]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vdiff_gamma, vmmword[reg_ptr_c + reg_off_c]);
            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
            uni_vdivps(vdiff_gamma, vdiff_gamma, vNS);
            int off = bdesc_->C() * acc_type_size_;
            jit_tail_.uni_vmovups_maybe_tail(
                    vdiff_beta, vmmword[reg_ptr_c + reg_off_c + off]);
            uni_vdivps(vdiff_beta, vdiff_beta, vNS);
        }
    }

    void compute_channels(bool stream_store_allowed) {
        Label label_C, label_S;
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat, reg_off_dat_save);

            load_c_specifics();

            mov(reg_S, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                jit_bf16_emu_.uni_vmovups_data(
                        v, vmmword[reg_ptr_diff_dst + reg_off_dat]);
                jit_relu_.bwd_process_relu(v);

                if (calculate_diff_stats()) {
                    uni_vsubps(v, v, vdiff_beta);
                    jit_bf16_emu_.uni_vmovups_data(
                            vtmp, vmmword[reg_ptr_src + reg_off_dat]);
                    uni_vsubps(vtmp, vtmp, vmean);
                    uni_vmulps(vtmp, vtmp, vdiff_gamma);
                    uni_vsubps(v, v, vtmp);
                }

                if (bdesc_->use_scaleshift()) uni_vmulps(v, v, vgamma);
                uni_vmulps(v, v, vsqrtvar);

                if (stream_store_allowed) {
                    uni_vmovntps(vmmword[reg_ptr_diff_src + reg_off_dat], v);
                } else {
                    jit_bf16_emu_.uni_vmovups_data(
                            vmmword[reg_ptr_diff_src + reg_off_dat], v);
                }

                add(reg_off_dat, simd_w * data_type_size_);

                dec(reg_S);
                jnz(label_S);
            }

            add(reg_off_dat_save, stride_C_ * data_type_size_);
            add(reg_off_c, simd_w * acc_type_size_);

            dec(reg_C);
            jnz(label_C);
        }
    }

    void compute(bool stream_store_allowed) {
        Label label_N;
        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            compute_channels(stream_store_allowed);

            if (isa == sse41) {
                xor_(reg_off_dat_save, reg_off_dat_save);
                xor_(reg_off_c, reg_off_c);
                add(reg_off_dat_save, vlen / 2);
                add(reg_off_c, vlen / 2);

                compute_channels(stream_store_allowed);
            }

            add(reg_ptr_src, stride_N_ * data_type_size_);
            add(reg_ptr_diff_src, stride_N_ * data_type_size_);
            add(reg_ptr_diff_dst, stride_N_ * data_type_size_);
            add(reg_ptr_ws, stride_N_ / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    bool calculate_diff_stats() const { return !bdesc_->use_global_stats(); }

    jit_bnorm_bwd_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc)
        , jit_tail_(bdesc, this, reg_tmp, reg_blk_has_tail, reg_C, vtail_mask,
                  ktail_mask)
        , jit_relu_(bdesc, this, reg_off_dat, reg_tmp, reg_ptr_ws, vzero,
                  vstore_mask, kstore_mask)
        , jit_bf16_emu_(
                  bdesc, this, Zmm(16), Zmm(17), Zmm(18), Zmm(19), reg_tmp) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_common,
                "unsupported isa");
        stride_C_ = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int C_PADDED = bdesc_->src_md()->padded_dims[1];
        stride_N_ = (C_PADDED / simd_w) * stride_C_;
        data_type_size_
                = types::data_type_size(bdesc->desc()->data_desc.data_type);
        acc_type_size_ = sizeof(acc_data_t);
    }

    void generate() override {
        bool is_bf16 = bdesc_->desc()->data_desc.data_type == data_type::bf16;
        preamble();
        load_common_params();
        jit_relu_.bwd_prepare_relu();
        jit_tail_.prepare_tail();

        Label normal_store, end_store;
        test(reg_ptr_diff_src, vlen - 1);
        jnz(normal_store, T_NEAR);
        compute(!is_bf16);
        jmp(end_store, T_NEAR);
        L(normal_store);
        { compute(false); }
        L(end_store);

        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_diff_ss_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_diff_ss_t)
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    const int simd_w = vlen / sizeof(acc_data_t);
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src, *diff_dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_gamma, *diff_beta;
        size_t blk_has_tail;
    };

    Reg64 reg_param = abi_param1;
    Reg64 reg_tmp = abi_not_param1;
    Reg64 reg_N = rsi;
    Reg64 reg_S = rax;
    Reg64 reg_C = rdx;
    Reg64 reg_off_c = rbx;
    Reg64 reg_blk_has_tail = rbp;

    Reg64 reg_off_dat = r8;
    Reg64 reg_off_dat_save = r9;
    Reg64 reg_ptr_c = r10;
    Reg64 reg_ptr_diff_gamma = r11;
    Reg64 reg_ptr_diff_beta = r12;
    Reg64 reg_ptr_ws = r13;
    Reg64 reg_ptr_diff_dst = r14;
    Reg64 reg_ptr_src = r15;

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);
    Vmm vmean = Vmm(2);
    Vmm vsqrtvar = Vmm(3);
    Vmm vgamma = Vmm(4);
    Vmm vdiff_gamma = Vmm(5);
    Vmm vdiff_beta = Vmm(6);
    Vmm veps = Vmm(7);
    Vmm vtmp = Vmm(8);
    Vmm v = Vmm(9);
    Vmm vtail_mask = Vmm(10);
    Vmm vstore_mask = vtmp;

    Opmask kstore_mask = Opmask(1);
    Opmask ktail_mask = Opmask(2);

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    jit_bnorm_bf16_emulation_t<isa> jit_bf16_emu_;
    int stride_N_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_diff_dst, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws, PARAM_PTR(ws));
        mov(reg_ptr_diff_gamma, PARAM_PTR(diff_gamma));
        mov(reg_ptr_diff_beta, PARAM_PTR(diff_beta));
#undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        uni_vmovq(x, reg_tmp);
        uni_vbroadcastss(vone, x);

        mov(reg_blk_has_tail, dword[PARAM_ADDR(blk_has_tail)]);
    }

    void load_c_specifics() {
        mov(reg_ptr_c, ptr[PARAM_ADDR(mean)]);
        jit_tail_.uni_vmovups_maybe_tail(vmean, vmmword[reg_ptr_c + reg_off_c]);

        mov(reg_ptr_c, ptr[PARAM_ADDR(var)]);
        jit_tail_.uni_vmovups_maybe_tail(
                vsqrtvar, vmmword[reg_ptr_c + reg_off_c]);
        uni_vaddps(vsqrtvar, vsqrtvar, veps);
        uni_vsqrtps(vsqrtvar, vsqrtvar);

        if (isa == sse41) {
            movups(vtmp, vone);
            divps(vtmp, vsqrtvar);
            movups(vsqrtvar, vtmp);
        } else
            vdivps(vsqrtvar, vone, vsqrtvar);

        uni_vpxor(vdiff_gamma, vdiff_gamma, vdiff_gamma);
        uni_vpxor(vdiff_beta, vdiff_beta, vdiff_beta);
    }

    void zeroise() {
        Label label_zeroise;
        xor_(reg_off_c, reg_off_c);
        uni_vpxor(vzero, vzero, vzero);
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_zeroise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_gamma + reg_off_c], vzero);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_beta + reg_off_c], vzero);
            if (isa == sse41) {
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_diff_gamma + reg_off_c + vlen / 2],
                        vzero);
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_diff_beta + reg_off_c + vlen / 2],
                        vzero);
            }
            add(reg_off_c, simd_w * acc_type_size_);
            dec(reg_C);
            jnz(label_zeroise);
        }
    }

    void compute_channels() {
        Label label_C, label_S;
        mov(reg_C, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat, reg_off_dat_save);

            load_c_specifics();

            mov(reg_S, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                jit_bf16_emu_.uni_vmovups_data(
                        v, vmmword[reg_ptr_diff_dst + reg_off_dat]);

                jit_relu_.bwd_process_relu(v);

                uni_vaddps(vdiff_beta, vdiff_beta, v);

                jit_bf16_emu_.uni_vmovups_data(
                        vtmp, vmmword[reg_ptr_src + reg_off_dat]);
                uni_vsubps(vtmp, vtmp, vmean);
                uni_vfmadd231ps(vdiff_gamma, vtmp, v);

                add(reg_off_dat, simd_w * data_type_size_);

                dec(reg_S);
                jnz(label_S);
            }

            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);

            jit_tail_.uni_vmovups_maybe_tail(
                    vtmp, vmmword[reg_ptr_diff_gamma + reg_off_c]);
            uni_vaddps(vdiff_gamma, vdiff_gamma, vtmp);
            jit_tail_.uni_vmovups_maybe_tail(
                    vtmp, vmmword[reg_ptr_diff_beta + reg_off_c]);
            uni_vaddps(vdiff_beta, vdiff_beta, vtmp);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_gamma + reg_off_c], vdiff_gamma);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_beta + reg_off_c], vdiff_beta);

            add(reg_off_dat_save, stride_C_ * data_type_size_);
            add(reg_off_c, simd_w * acc_type_size_);

            dec(reg_C);
            jnz(label_C);
        }
    }

    void compute() {
        Label label_N;
        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            compute_channels();

            if (isa == sse41) {
                xor_(reg_off_dat_save, reg_off_dat_save);
                xor_(reg_off_c, reg_off_c);
                add(reg_off_dat_save, vlen / 2);
                add(reg_off_c, vlen / 2);

                compute_channels();
            }

            add(reg_ptr_src, stride_N_ * data_type_size_);
            add(reg_ptr_diff_dst, stride_N_ * data_type_size_);
            add(reg_ptr_ws, stride_N_ / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    jit_bnorm_bwd_diff_ss_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc)
        , jit_tail_(bdesc, this, reg_tmp, reg_blk_has_tail, reg_C, vtail_mask,
                  ktail_mask)
        , jit_relu_(bdesc, this, reg_off_dat, reg_tmp, reg_ptr_ws, vzero,
                  vstore_mask, kstore_mask)
        , jit_bf16_emu_(
                  bdesc, this, Zmm(16), Zmm(17), Zmm(18), Zmm(19), reg_tmp) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_common,
                "unsupported isa");
        stride_C_ = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int C_PADDED = bdesc_->src_md()->padded_dims[1];
        stride_N_ = (C_PADDED / simd_w) * stride_C_;
        data_type_size_
                = types::data_type_size(bdesc->desc()->data_desc.data_type);
        acc_type_size_ = sizeof(acc_data_t);
    }

    void generate() override {
        preamble();
        load_common_params();
        jit_relu_.bwd_prepare_relu();
        jit_tail_.prepare_tail();
        zeroise();
        compute();
        postamble();
    }
};
} // namespace
namespace bnorm_tbb_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
private:
    struct bnorm_dims_t {
        dim_t N, C, S;
        dim_t glob;
    };

    DNNL_DISALLOW_COPY_AND_ASSIGN(driver_t);

public:
    driver_t(const batch_normalization_pd_t *bdesc) : bdesc_(bdesc) {
        nthr_ = dnnl_get_max_threads();
        N_ = bdesc_->MB();
        S_ = bdesc_->D() * bdesc_->H() * bdesc_->W();
        C_ = bdesc_->C();
        C_blks_ = get_c_padded(bdesc_) / simd_w;

        const size_t l3_size = platform::get_per_core_cache_size(3) * nthr_ / 2;
        int num_tensors = bdesc_->is_fwd() ? 1 : 2;
        dt_size_ = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        const size_t working_set_size
                = dt_size_ * N_ * S_ * simd_w * num_tensors;

        do_blocking_ = working_set_size * C_blks_ >= l3_size / 2 && l3_size > 0;

        C_blk_step_ = l3_size / working_set_size;
        C_blk_step_ = nstl::max<dim_t>(C_blk_step_, 1);
        C_blk_step_ = nstl::min<dim_t>(C_blk_step_, C_blks_);
    }

    status_t create_kernel() {
        if (bdesc_->is_fwd()) {
            CHECK(safe_ptr_assign(ker_fwd_, new jit_bnorm_fwd_t<isa>(bdesc_)));
            CHECK(ker_fwd_->create_kernel());
            if (!bdesc_->stats_is_src()) {
                CHECK(safe_ptr_assign(
                        ker_fwd_mean_, new jit_bnorm_fwd_mean_t<isa>(bdesc_)));
                CHECK(safe_ptr_assign(
                        ker_fwd_var_, new jit_bnorm_fwd_var_t<isa>(bdesc_)));
                CHECK(ker_fwd_mean_->create_kernel());
                CHECK(ker_fwd_var_->create_kernel());
            }
        } else {
            CHECK(safe_ptr_assign(ker_bwd_, new jit_bnorm_bwd_t<isa>(bdesc_)));
            CHECK(safe_ptr_assign(ker_bwd_diff_ss_,
                    new jit_bnorm_bwd_diff_ss_t<isa>(bdesc_)));
            CHECK(ker_bwd_->create_kernel());
            CHECK(ker_bwd_diff_ss_->create_kernel());
        }
        return status::success;
    }

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *bdesc) {

        int nthrs = dnnl_get_max_threads();
        int C_PADDED = get_c_padded(bdesc);

        int sbuf_sz = use_tmp_stats(bdesc) * 2 * C_PADDED;
        int pbuf_sz = use_tmp_diff_scale_shift(bdesc) * 2 * C_PADDED;
        int rbuf_sz = (bdesc->is_fwd() ? 1 : 2) * C_PADDED * nthrs;

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);
    }

    void exec_fwd_step_stats(const dim_t C_blks, const bnorm_dims_t &nthr,
            const void *src, acc_data_t *mean, acc_data_t *var,
            acc_data_t *rbuf, bool blk_has_tail) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;
        const dim_t tail_size = blk_has_tail ? C_ % simd_w : simd_w;

        const dim_t size_C_stat = (C_blks - 1) * simd_w + tail_size;

        auto reduce = [&](acc_data_t *stat, acc_data_t *r_stat) {
            if (!need_reduction) return;
            acc_data_t *loc_stat = r_stat;

            for (dim_t c = 0; c < size_C_stat; ++c)
                stat[c] = loc_stat[c];

            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_stat += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    stat[c] += loc_stat[c];
            }

            for (dim_t c = 0; c < size_C_stat; ++c)
                stat[c] /= N_ * S_;
        };

        // find local mean
        acc_data_t *r_mean = need_reduction ? rbuf : mean;
        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_mean_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * simd_w;
            c.src = (void *)((char *)src + d_off * dt_size_);
            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            c.mean = &r_mean[ithr_NS * size_C_stat + start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            c.do_normalise = !need_reduction;
            (*ker_fwd_mean_)(&c);
        });

        // mean reduction
        reduce(mean, r_mean);

        // find local var
        acc_data_t *r_var = need_reduction ? rbuf : var;
        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_var_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * simd_w;
            c.src = (void *)((char *)src + d_off * dt_size_);
            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            c.mean = &mean[start.C * simd_w];
            c.var = &r_var[ithr_NS * size_C_stat + start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            c.do_normalise = !need_reduction;
            (*ker_fwd_var_)(&c);
        });

        // var reduction
        reduce(var, r_var);
    }

    void exec_fwd_step_normalization(const dim_t C_blks,
            const bnorm_dims_t &nthr, const void *src, void *dst,
            const acc_data_t *scale_shift, const acc_data_t *mean,
            const acc_data_t *var, uint8_t *ws, bool blk_has_tail) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;
        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * simd_w;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.dst = (void *)((char *)dst + d_off * dt_size_);
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale_shift = &scale_shift[start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            (*ker_fwd_)(&c);
        });
    }

    void exec_fwd(const void *src, void *dst, const acc_data_t *scale_shift,
            acc_data_t *mean, acc_data_t *var, uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        if (use_tmp_stats(bdesc_)) {
            auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
            mean = sbuf;
            var = sbuf + C_blks_ * simd_w;
        }

        const size_t stride_C = (size_t)S_ * simd_w;

        dim_t C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (dim_t C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            if (!bdesc_->stats_is_src()) {
                exec_fwd_step_stats(C_blk_step, nthr,
                        (void *)((char *)src
                                + (C_blk_st * stride_C) * dt_size_),
                        mean + C_blk_st * simd_w, var + C_blk_st * simd_w, rbuf,
                        (C_blk_st + C_blk_step) * simd_w > C_);
            }

            exec_fwd_step_normalization(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)dst + (C_blk_st * stride_C) * dt_size_),
                    scale_shift + C_blk_st * simd_w, mean + C_blk_st * simd_w,
                    var + C_blk_st * simd_w, ws + C_blk_st * stride_C / 8,
                    (C_blk_st + C_blk_step) * simd_w > C_);
        }
    }

    void exec_bwd_step_diff_ss(const dim_t C_blks, const bnorm_dims_t &nthr,
            const void *src, const void *diff_dst, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws, acc_data_t *diff_ss,
            acc_data_t *rbuf, bool blk_has_tail) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;
        const dim_t tail_size = blk_has_tail ? C_ % simd_w : simd_w;
        const dim_t size_C_stat = (C_blks - 1) * simd_w + tail_size;

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;

        acc_data_t *diff_gamma = diff_ss;
        acc_data_t *diff_beta = diff_ss + C_;

        acc_data_t *const r_diff_gamma = need_reduction ? rbuf : diff_gamma;
        acc_data_t *const r_diff_beta
                = need_reduction ? rbuf + nthr_NS * size_C_stat : diff_beta;

        auto reduce = [&]() {
            if (!need_reduction) return;

            // diff_gamma
            const acc_data_t *loc_diff_gamma = r_diff_gamma;
            for (dim_t c = 0; c < size_C_stat; ++c)
                diff_gamma[c] = loc_diff_gamma[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_gamma += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    diff_gamma[c] += loc_diff_gamma[c];
            }

            // diff_beta
            const acc_data_t *loc_diff_beta = r_diff_beta;
            for (dim_t c = 0; c < size_C_stat; ++c)
                diff_beta[c] = loc_diff_beta[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_beta += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    diff_beta[c] += loc_diff_beta[c];
            }
        };

        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            acc_data_t *loc_diff_gamma = &r_diff_gamma[ithr_NS * size_C_stat];
            acc_data_t *loc_diff_beta = &r_diff_beta[ithr_NS * size_C_stat];

            auto c = typename jit_bnorm_bwd_diff_ss_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * simd_w;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.diff_dst = (void *)((char *)diff_dst + d_off * dt_size_);
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.diff_gamma = &loc_diff_gamma[start.C * simd_w];
            c.diff_beta = &loc_diff_beta[start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;

            (*ker_bwd_diff_ss_)(&c);
        });

        reduce();
    }

    void exec_bwd_step_normalization(const dim_t C_blks,
            const bnorm_dims_t &nthr, const void *src, void *diff_src,
            const void *diff_dst, const acc_data_t *mean, const acc_data_t *var,
            const uint8_t *ws, const acc_data_t *scale_shift,
            const acc_data_t *diff_ss, bool blk_has_tail) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_bwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * simd_w;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.diff_src = (void *)((char *)diff_src + d_off * dt_size_);
            c.diff_dst = (void *)((char *)diff_dst + d_off * dt_size_);
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale_shift = &scale_shift[start.C * simd_w];
            c.diff_scale_shift = &diff_ss[start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;

            (*ker_bwd_)(&c);
        });
    }

    void exec_bwd(const void *src, void *diff_src, const void *diff_dst,
            const acc_data_t *scale_shift, acc_data_t *diff_scale_shift,
            const acc_data_t *mean, const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        if (use_tmp_diff_scale_shift(bdesc_)) {
            auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
            diff_scale_shift = pbuf;
        }

        const size_t stride_C = (size_t)S_ * simd_w;

        dim_t C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (dim_t C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            exec_bwd_step_diff_ss(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_dst
                            + (C_blk_st * stride_C) * dt_size_),
                    mean + C_blk_st * simd_w, var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / 8,
                    diff_scale_shift + C_blk_st * simd_w, rbuf,
                    (C_blk_st + C_blk_step) * simd_w > C_);

            exec_bwd_step_normalization(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_src
                            + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_dst
                            + (C_blk_st * stride_C) * dt_size_),
                    mean + C_blk_st * simd_w, var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / 8,
                    scale_shift + C_blk_st * simd_w,
                    diff_scale_shift + C_blk_st * simd_w,
                    (C_blk_st + C_blk_step) * simd_w > C_);
        }
    }

private:
    enum {
        simd_w = isa == sse41 ? 8
                              : cpu_isa_traits<isa>::vlen
                        / sizeof(acc_data_t) // BF16 will expand to FP32
    };

    static bool use_tmp_stats(const batch_normalization_pd_t *bdesc) {
        return true && !bdesc->stats_is_src()
                && bdesc->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale_shift(
            const batch_normalization_pd_t *bdesc) {
        return false || (bdesc->is_bwd() && !bdesc->use_scaleshift())
                || bdesc->desc()->prop_kind == prop_kind::backward_data;
    }

    static dim_t get_c_padded(const batch_normalization_pd_t *bdesc) {
        return bdesc->src_md()->padded_dims[1];
    }

    void thread_distribution(dim_t C_blks, bnorm_dims_t &nthr) {
        if (do_blocking_) {
            nthr.N = nstl::min<dim_t>(N_, nthr_);
            nthr.C = nstl::min<dim_t>(C_blks, nthr_ / nthr.N);
        } else {
            nthr.C = math::gcd((dim_t)nthr_, C_blks);
            nthr.N = nstl::max<dim_t>(1, nstl::min(N_, nthr_ / nthr.C));
        }
        nthr.S = nstl::max<dim_t>(
                1, nstl::min<dim_t>(S_, nthr_ / nthr.C / nthr.N));
        nthr.glob = nthr.N * nthr.C * nthr.S;
    }

    int map_thread_c(int ithr_glob, const bnorm_dims_t &nthr) {
        return ithr_glob / nthr.N / nthr.S;
    }

    bnorm_dims_t map_thread(int ithr_glob, const bnorm_dims_t &nthr) {
        auto ithr = bnorm_dims_t();
        ithr.glob = ithr_glob;
        ithr.C = map_thread_c(ithr.glob, nthr);
        ithr.N = ithr.glob / nthr.S % nthr.N;
        ithr.S = ithr.glob % nthr.S;
        return ithr;
    }

    void work_distribution_c(dim_t C_blks, int ithr_c, int nthr_c,
            dim_t &start_c, dim_t &stop_c) {
        balance211(C_blks, nthr_c, ithr_c, start_c, stop_c);
    }

    void work_distribution(dim_t C_blks, const bnorm_dims_t &ithr,
            const bnorm_dims_t &nthr, bnorm_dims_t &start, bnorm_dims_t &stop) {
        work_distribution_c(C_blks, ithr.C, nthr.C, start.C, stop.C);
        balance211(N_, nthr.N, ithr.N, start.N, stop.N);
        balance211(S_, nthr.S, ithr.S, start.S, stop.S);
    }

    const batch_normalization_pd_t *bdesc_;

    bool do_blocking_;

    int nthr_;

    dim_t N_, S_; // MB, D * H *W
    dim_t C_, C_blks_; // C / simd_w
    dim_t C_blk_step_; // for C_blks = 0 .. C_blks_, += C_blk_step_

    jit_bnorm_fwd_t<isa> *ker_fwd_;
    jit_bnorm_fwd_mean_t<isa> *ker_fwd_mean_;
    jit_bnorm_fwd_var_t<isa> *ker_fwd_var_;
    jit_bnorm_bwd_t<isa> *ker_bwd_;
    jit_bnorm_bwd_diff_ss_t<isa> *ker_bwd_diff_ss_;

    size_t dt_size_;
};
} // namespace bnorm_tbb_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */
template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    auto desired_fmt_tag = (ndims() == 4)
            ? isa == avx512_common ? nChw16c : nChw8c
            : isa == avx512_common ? nCdhw16c : nCdhw8c;

    bool ok = true && mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
            && one_of(ndims(), 4, 5) && one_of(src_md()->data_type, f32, bf16)
            && IMPLICATION(src_md()->data_type == bf16, mayiuse(avx512_core))
            && check_scale_shift_data_type()
            && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    if (is_training() && fuse_norm_relu()) {
        if (isa < avx2) return status::unimplemented;
        init_default_ws(1);
    }

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < avx2)
        return status::unimplemented;

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_tbb_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<
        isa>::jit_uni_tbb_batch_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_tbb_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale_shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);

    auto mean = pd()->stats_is_src() ? const_cast<acc_data_t *>(
                        CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN))
                                     : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
    auto var = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                    CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->exec_fwd(src, dst, scale_shift, mean, var, ws, scratchpad);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<
        isa>::~jit_uni_tbb_batch_normalization_fwd_t()
        = default;

template struct jit_uni_tbb_batch_normalization_fwd_t<sse41>;
template struct jit_uni_tbb_batch_normalization_fwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_fwd_t<avx512_common>;

/* bwd */
template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::pd_t::init(
        engine_t *engine) {
    auto desired_fmt_tag = (ndims() == 4)
            ? one_of(isa, sse41, avx2) ? nChw8c : nChw16c
            : one_of(isa, sse41, avx2) ? nCdhw8c : nCdhw16c;

    bool ok = true && mayiuse(isa) && is_bwd() && !has_zero_dim_memory()
            && one_of(ndims(), 4, 5) && set_default_formats_common()
            && one_of(true,
                    everyone_is(
                            f32, src_md()->data_type, diff_src_md()->data_type),
                    everyone_is(bf16, src_md()->data_type,
                            diff_src_md()->data_type))
            && IMPLICATION(src_md()->data_type == bf16, mayiuse(avx512_core))
            && check_scale_shift_data_type()
            && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
            && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag)
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < avx2)
        return status::unimplemented;

    if (fuse_norm_relu()) {
        if (isa < avx2) return status::unimplemented;
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_tbb_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<
        isa>::jit_uni_tbb_batch_normalization_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_tbb_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto var = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale_shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    auto diff_scale_shift
            = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->exec_bwd(src, diff_src, diff_dst, scale_shift,
            diff_scale_shift, mean, var, ws, scratchpad);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<
        isa>::~jit_uni_tbb_batch_normalization_bwd_t()
        = default;

template struct jit_uni_tbb_batch_normalization_bwd_t<sse41>;
template struct jit_uni_tbb_batch_normalization_bwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_bwd_t<avx512_common>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
