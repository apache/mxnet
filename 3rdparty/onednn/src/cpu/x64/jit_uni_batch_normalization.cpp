/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_uni_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

using namespace memory_tracking::names;

using namespace Xbyak;
namespace barrier = simple_barrier;

using acc_data_t = float;

template <cpu_isa_t isa>
struct jit_bnorm_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size, spat_size_loc;
        size_t S_s, S_tail;
        size_t is_cblk_tail;
        acc_data_t chan_size, eps, one;
        const acc_data_t *scale_shift;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_scale_shift;
        const void *src, *dst;
        const void *diff_src, *diff_dst;
        const acc_data_t *rbuf1, *rbuf2;
        const uint8_t *ws;
        barrier::ctx_64_t *barrier;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    int vlen_spat_data_; // set by ctor depending on data type (BF16 or FP32);

    const batch_normalization_pd_t *bdesc_;
    bool is_spatial_thr_;
    bool is_nspc_;
    bool is_bf16_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_scale_shift = rbx;
    Reg64 reg_rbuf1 = abi_not_param1;
    Reg64 reg_rbuf2 = rdx;
    Reg64 reg_coff_max_fwd_copy = reg_rbuf2;

    Reg64 reg_mean = rbp;
    Reg64 reg_var = reg_param;
    Reg64 reg_diff_scale_shift = rax;
    Reg64 reg_coff_max_bwd_copy = reg_diff_scale_shift;

    Reg64 reg_coff = r8;
    Reg64 reg_coff_max = r9;
    Reg64 reg_soff = r10;
    Reg64 reg_soff_max = r11;
    Reg64 reg_ctr = r12;
    Reg64 reg_roff = r13;

    Reg64 reg_mb_stride_Bc = r14;
    Reg64 reg_soff_nspc = reg_mb_stride_Bc;

    Reg64 reg_src = r15;
    Reg64 reg_diff_src = reg_rbuf1;
    Reg64 reg_dst = rsi;
    Reg64 reg_diff_dst = reg_dst;

    Reg64 reg_tmp_off = reg_roff;

    // Reuse loop counters
    Reg64 reg_bar = reg_coff;
    Reg64 reg_nnthr = reg_soff; // must be usable w/ loops over coff
    Reg64 reg_tmp = reg_ctr;

    // Relu section
    bool with_relu, with_relu_inf_only;
    Vmm vzero; // is_fwd() ? vdiff_beta : vbeta
    Reg64 reg_ws = reg_roff;
    Label l_relu_mask_avx2;
    Opmask kstore_mask = Opmask(1);

    // channel tail processing
    Opmask ktail_mask = Opmask(2);

    // FP32->BF16 emulation
    bf16_emulation_t *bf16_emu_ {nullptr};
    Reg64 reg_bf16_tmp = reg_tmp;
    Zmm bf16_emu_reserved_1 = Zmm(16);
    Zmm bf16_emu_reserved_2 = Zmm(17);
    Zmm bf16_emu_reserved_3 = Zmm(18);
    Zmm bf16_emu_reserved_4 = Zmm(19);

    size_t unroll_blocks;
    size_t unroll_regs;
    Vmm vbuf = Vmm(isa == avx512_common ? 20 : 5);
    Vmm vdiff_beta = Vmm(isa == avx512_common ? 21 : 6);
    Vmm vdiff_gamma = Vmm(isa == avx512_common ? 22 : 7);
    Vmm vsqrtvar = Vmm(isa == avx512_common ? 23 : 8);
    Vmm vone = Vmm(isa == avx512_common ? 24 : 9);
    Vmm vmean = Vmm(isa == avx512_common ? 25 : 10);
    Vmm vgamma = Vmm(isa == avx512_common ? 26 : 11);
    Vmm vbeta = Vmm(isa == avx512_common ? 27 : 12);
    Vmm veps = Vmm(isa == avx512_common ? 28 : 13);
    Vmm vchan_size = Vmm(isa == avx512_common ? 29 : 14);
    Vmm vtail_mask = Vmm(isa == avx512_common ? 30 : 15);

    size_t t0_pf_offt;
    size_t t1_pf_offt;
    size_t spat_size;
    size_t chan_data_offt;
    size_t spat_step;
    size_t mb_offt;
    size_t ws_mb_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_diff_scale_shift = 48,
        stack_off_ws = 56,
        stack_off_barrier = 64,
        stack_off_spat_size_loc = 72,
        stack_off_s_s = 80,
        stack_off_s_tail = 88,
        stack_off_is_cblk_tail = 96,
        stack_off_ws_off_copy = 104,
        stack_size_required = 112,
    };

    int bit_shift() { return 5 - is_bf16_; }

    bool stream_store_supported() { return !is_bf16_; }

    bool is_c_padded() const {
        const memory_desc_wrapper data_d(bdesc_->src_md());
        return bdesc_->C() != data_d.padded_dims()[1];
    }

    void compute_static_strides() {
        spat_size = bdesc_->D() * bdesc_->W() * bdesc_->H();
        chan_data_offt = bdesc_->C() * sizeof(acc_data_t);
        spat_step
                = is_nspc_ ? chan_data_offt / (1 + is_bf16_) : vlen_spat_data_;
        mb_offt = spat_step * spat_size;
        ws_mb_offt = (spat_step / (is_bf16_ ? 16 : 32)) * spat_size;

        if (isa == avx512_mic) {
            t0_pf_offt = 4096;
            t1_pf_offt = 0;
        } else {
            t0_pf_offt = 0;
            t1_pf_offt = 0;
        }
    }

    void load_common_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_rbuf1, ptr[reg_param + PARAM_OFF(rbuf1)]);
        if (bdesc_->is_bwd()) mov(reg_rbuf2, ptr[reg_param + PARAM_OFF(rbuf2)]);
        mov(reg_coff_max, ptr[reg_param + PARAM_OFF(coff_max)]);
        mov(reg_soff_max, ptr[reg_param + PARAM_OFF(soff_max)]);
        mov(reg_mb_stride_Bc, ptr[reg_param + PARAM_OFF(mb_stride_Bc)]);
        shl(reg_coff_max, 2);

        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_scale_shift, ptr[reg_param + PARAM_OFF(scale_shift)]);

        uni_vbroadcastss(vchan_size, vmmword[reg_param + PARAM_OFF(chan_size)]);
        uni_vbroadcastss(vone, vmmword[reg_param + PARAM_OFF(one)]);
        uni_vbroadcastss(veps, vmmword[reg_param + PARAM_OFF(eps)]);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_nthr)]);
        mov(ptr[rsp + stack_off_N_nthr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_ithr)]);
        mov(ptr[rsp + stack_off_N_ithr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(src)]);
        mov(ptr[rsp + stack_off_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(dst)]);
        mov(ptr[rsp + stack_off_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_src)]);
        mov(ptr[rsp + stack_off_diff_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(ptr[rsp + stack_off_diff_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(ws)]);
        mov(ptr[rsp + stack_off_ws], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(barrier)]);
        mov(ptr[rsp + stack_off_barrier], reg_tmp);
        if (is_spatial_thr_) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(spat_size_loc)]);
            mov(ptr[rsp + stack_off_spat_size_loc], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(S_s)]);
            mov(ptr[rsp + stack_off_s_s], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(S_tail)]);
            mov(ptr[rsp + stack_off_s_tail], reg_tmp);
        }
        if (is_c_padded()) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(is_cblk_tail)]);
            mov(ptr[rsp + stack_off_is_cblk_tail], reg_tmp);
        }

        if (bdesc_->is_fwd()) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        } else {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_scale_shift)]);
            mov(ptr[rsp + stack_off_diff_scale_shift], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        }
#undef PARAM_OFF
    }

    void prepare_tail_mask_avx512_common() {
        if (!is_c_padded()) return;

        const int tail = bdesc_->C() % (int)(vlen / sizeof(float));
        const int mask = (1 << tail) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    void prepare_tail_mask_avx2_common() {
        if (!is_c_padded()) return;

        const int tail = bdesc_->C() % (int)(vlen / sizeof(float));
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask[8 - tail]));
        vmovups(vtail_mask, ptr[reg_tmp]);
    }

    void prepare_relu() {
        with_relu = bdesc_->is_fwd()
                ? bdesc_->with_relu_post_op() || bdesc_->fuse_norm_relu()
                : bdesc_->fuse_norm_relu();
        with_relu_inf_only = with_relu && bdesc_->is_fwd()
                && !(bdesc_->fuse_norm_relu() && bdesc_->is_training());

        vzero = bdesc_->is_fwd() ? vdiff_beta : vbeta;
        if (with_relu) {
            uni_vpxor(vzero, vzero, vzero);
            if (!bdesc_->is_fwd() && isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        jmp(l_mask_after);
        align(32);
        L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i)
            dd(1 << i);
        L(l_mask_after);
    }

    void fwd_process_relu_avx2(Vmm vdst, int offt, Vmm vstore_mask) {
        Reg64 reg_store_mask = reg_diff_scale_shift;
        shr(reg_soff, bit_shift());
        vcmpps(vstore_mask, vzero, vdst, _cmp_lt_os);
        vmovmskps(reg_store_mask, vstore_mask);
        mov(ptr[reg_ws + reg_soff + offt / (1 << bit_shift())],
                reg_store_mask.cvt8());
        vblendvps(vdst, vzero, vdst, vstore_mask);
        shl(reg_soff, bit_shift());
    }

    void fwd_process_relu_avx512_common(Vmm vdst, int offt = 0) {
        shr(is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
        vcmpps(kstore_mask, vzero, vdst, _cmp_lt_os);
        kmovw(ptr[reg_ws + (is_nspc_ ? reg_soff_nspc : reg_soff)
                      + offt / (1 << bit_shift())],
                kstore_mask);
        vblendmps(vdst | kstore_mask, vzero, vdst);
        shl(is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst, int offt, Vmm vstore_mask) {
        shr(reg_soff, bit_shift());
        vpbroadcastb(vstore_mask,
                ptr[reg_ws + reg_soff + offt / (1 << bit_shift())]);
        vpand(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vpcmpeqd(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vblendvps(vdiff_dst, vzero, vdiff_dst, vstore_mask);
        shl(reg_soff, bit_shift());
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst, int offt = 0) {
        shr(is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
        kmovw(kstore_mask,
                ptr[reg_ws + (is_nspc_ ? reg_soff_nspc : reg_soff)
                        + offt / (1 << bit_shift())]);
        vmovups(vdiff_dst | kstore_mask | T_z, vdiff_dst);
        shl(is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
    }

    void uni_vmovups_spat_data(const Operand &dst, const Operand &src) {
        if (dst.isMEM()) {
            if (is_bf16_) {
                constexpr bool isAvx2 = isa == avx2;
                const typename std::conditional<isAvx2, Xmm, Ymm>::type
                        dst_reg {src.getIdx()};
                const typename std::conditional<isAvx2, Ymm, Zmm>::type
                        src_reg {src.getIdx()};

                // convert f32 output to bf16
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(dst_reg, src_reg);
                else
                    bf16_emu_->vcvtneps2bf16(dst_reg, src_reg);

                vmovdqu16(dst.getAddress(), dst_reg);
            } else {
                uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
            }
        } else {
            if (is_bf16_) {
                // convert bf16 input to f32
                vpmovzxwd(Vmm(dst.getIdx()), src.getAddress());
                vpslld(Vmm(dst.getIdx()), Vmm(dst.getIdx()), 0x10);
            } else {
                uni_vmovups(Vmm(dst.getIdx()), src.getAddress());
            }
        }
    }

    void uni_vmovups_tail_avx2_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM()) {
            vmaskmovps(dst.getAddress(), vtail_mask, Vmm(src.getIdx()));
        } else {
            vmaskmovps(Vmm(dst.getIdx()), vtail_mask, src.getAddress());
        }
        jmp(l_ret);
    }

    void uni_vmovups_tail_avx512_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM())
            uni_vmovups(dst.getAddress() | ktail_mask | T_z, Vmm(src.getIdx()));
        else
            uni_vmovups(Vmm(dst.getIdx()) | ktail_mask | T_z, src.getAddress());

        jmp(l_ret);
    }

    void uni_vmovups_maybe_tail(const Operand &dst, const Operand &src) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            mov(reg_tmp, ptr[rsp + stack_off_is_cblk_tail]);
            cmp(reg_tmp, 0);
            jz(l_no_mask);

            lea(reg_tmp, ptr[reg_coff + vlen]);
            cmp(reg_tmp, reg_coff_max);
            jl(l_no_mask);
            assert(isa == avx512_common || isa == avx2);
            if (isa == avx512_common)
                uni_vmovups_tail_avx512_common(dst, src, l_ret);
            else if (isa == avx2)
                uni_vmovups_tail_avx2_common(dst, src, l_ret);
        }
        L(l_no_mask);
        if (dst.isMEM())
            uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
        else
            uni_vmovups(Vmm(dst.getIdx()), src.getAddress());

        L(l_ret);
    }

    void barrier() {
        mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
        mov(reg_bar, ptr[rsp + stack_off_barrier]);
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + reg_coff + offt + 0 * chan_data_offt];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + reg_coff + offt + 0 * chan_data_offt];
    }

    Address diff_gamma_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale_shift + reg_coff + offt
                + 0 * chan_data_offt];
    }

    Address diff_beta_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale_shift + reg_coff + offt
                + 1 * chan_data_offt];
    }

    Address gamma_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_coff + offt + 0 * chan_data_offt];
    }

    Address beta_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + reg_coff + offt + 1 * chan_data_offt];
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs, init_t init,
            body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;
        for (size_t i = 0; i < num_active_regs; i++)
            init(i);
        if (loop_unroll) {
            if (is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, loop_unroll);
            }
            Label label;
            L(label);
            {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }
                add(reg_soff, factor * spat_step);
                sub(reg_ctr, factor);
                jnz(label);
            }
            if (is_spatial_thr_) { add(reg_soff, ptr[rsp + stack_off_s_tail]); }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail) add(reg_soff, loop_tail * spat_step);

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) uni_vpxor(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v0 = Vmm(base_reg * 2 + 0);
                        Vmm v1 = Vmm(base_reg * 2 + 1);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                v1, vmmword[reg_src + reg_soff + offt]);
                        uni_vaddps(v0, v0, v1);
                        mic_prefetcht0(
                                ptr[reg_src + reg_soff + offt + t0_pf_offt]);
                        mic_prefetcht1(
                                ptr[reg_src + reg_soff + offt + t1_pf_offt]);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void mean_variance_nspc(
            const int num_ch_blks, int num_spat_pts, bool compute_mean) {

        auto mean_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    uni_vmovups_spat_data(Vmm(sp_idx),
                            vmmword[reg_src + reg_soff_nspc + offt]);

                    uni_vaddps(Vmm(ch_idx), Vmm(ch_idx), Vmm(sp_idx++));

                    offt += vlen_spat_data_;
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        auto variance_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int coff = 0, offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    uni_vmovups_maybe_tail(vmean, mean_ptr(coff));

                    uni_vmovups_spat_data(Vmm(sp_idx),
                            vmmword[reg_src + reg_soff_nspc + offt]);

                    vsubps(Vmm(30), vmean, Vmm(sp_idx++));
                    uni_vfmadd231ps(Vmm(ch_idx), Vmm(30), Vmm(30));

                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen)
            uni_vmovups(Vmm(idx), vmmword[reg_rbuf1 + reg_coff + offt]);

        xor_(reg_soff_nspc, reg_soff_nspc);

        if (is_spatial_thr_) {
            mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
            add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            // TODO: need a better heuristic for num_spat_pts
            num_spat_pts = 1;
        } else {
            mov(reg_ctr, spat_size);
            num_spat_pts = nstl::min((size_t)num_spat_pts, spat_size);
            // TODO: unroll by spatial
            if (spat_size % num_spat_pts != 0) num_spat_pts = 1;
        }

        Label spatial;
        L(spatial);
        {
            compute_mean ? mean_compute(num_ch_blks, num_spat_pts)
                         : variance_compute(num_ch_blks, num_spat_pts);
            sub(reg_ctr, num_spat_pts);
            jnz(spatial, T_NEAR);
        }

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen)
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff + offt], Vmm(idx));
    }

    void forward_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            // Overwritten during mean and variance computation
            uni_vpxor(vzero, vzero, vzero);

            xor_(reg_soff_nspc, reg_soff_nspc);

            if (is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    uni_vmovups_maybe_tail(vmean, mean_ptr(coff));
                    uni_vmovups_maybe_tail(vsqrtvar, var_ptr(coff));
                    uni_vaddps(vsqrtvar, vsqrtvar, veps);
                    uni_vsqrtps(vsqrtvar, vsqrtvar);

                    if (bdesc_->use_scaleshift()) {
                        uni_vmovups_maybe_tail(vgamma, gamma_ptr(coff));
                        uni_vmovups_maybe_tail(vbeta, beta_ptr(coff));
                    }

                    Vmm vscale = bdesc_->use_scaleshift() ? vgamma : vone;
                    Vmm vdiv = bdesc_->use_scaleshift() ? vgamma : vsqrtvar;

                    vdivps(vdiv, vscale, vsqrtvar);

                    uni_vmovups_spat_data(
                            Vmm(idx), vmmword[reg_src + reg_soff_nspc + offt]);

                    uni_vsubps(Vmm(idx), Vmm(idx), vmean);

                    if (bdesc_->use_scaleshift()) { // --flags=S
                        uni_vfmadd213ps(Vmm(idx), vgamma, vbeta);
                    } else {
                        uni_vmulps(Vmm(idx), Vmm(idx), vsqrtvar);
                    }

                    if (with_relu_inf_only) { // --attr=post_ops='relu'
                        uni_vmaxps(Vmm(idx), Vmm(idx), vzero);
                    } else if (with_relu) { // --flags=R
                        fwd_process_relu_avx512_common(Vmm(idx));
                    }

                    if (stream_store_allowed) {
                        uni_vmovntps(vmmword[reg_dst + reg_soff_nspc + offt],
                                Vmm(idx));
                    } else {
                        uni_vmovups_spat_data(
                                vmmword[reg_dst + reg_soff_nspc + offt],
                                Vmm(idx));
                    }

                    add(reg_ws, 2);
                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add(reg_soff_nspc, spat_step);
                sub(reg_ws, 2 * num_ch_blks);
                sub(reg_ctr, num_spat_pts);
                jnz(spatial, T_NEAR);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            test(reg_dst, vlen - 1);
            jnz(normal_store, T_NEAR);
            compute(true);
            jmp(end_store, T_NEAR);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void compute_mean_variance_nspc(bool compute_mean = true) {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll, sp_idx = 1; ch_idx > 0;
                --ch_idx, ++sp_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                const int spat_blk_size = (1 << sp_idx);
                mean_variance_nspc(ch_blk_size, spat_blk_size, compute_mean);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_coff, vlen * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_bf16_) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        if (is_bf16_) shl(reg_coff_max, 1);
    }

    void var_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg > 0) uni_vpxor(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v = Vmm(3 * base_reg);
                        Vmm vtmp0 = Vmm(3 * base_reg + 1);
                        Vmm vtmp1 = Vmm(3 * base_reg + 2);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                vtmp0, vmmword[reg_src + reg_soff + offt]);
                        if (isa == sse41) {
                            movups(vtmp1, vmean);
                            subps(vtmp1, vtmp0);
                        } else {
                            vsubps(vtmp1, vmean, vtmp0);
                        }
                        uni_vfmadd231ps(v, vtmp1, vtmp1);

                        mic_prefetcht0(
                                ptr[reg_src + reg_soff + offt + t0_pf_offt]);
                        mic_prefetcht1(
                                ptr[reg_src + reg_soff + offt + t1_pf_offt]);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg) uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void compute_mean_variance() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf;
        L(zero_rbuf);
        {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);

        xor_(reg_soff, reg_soff);
        Label mean_spatial;
        L(mean_spatial);
        {
            xor_(reg_coff, reg_coff);

            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            is_nspc_ ? compute_mean_variance_nspc() : mean_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                mean_channels();

                sub(reg_src, vlen / 2);
            }

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_soff, mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(mean_spatial);
        }

        if (is_nspc_) mov(reg_src, ptr[rsp + stack_off_src]); // comeback

        Label no_mean_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_mean_reduction);
            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs);
                {
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    uni_vmovups(vmmword[reg_rbuf1 + reg_roff], Vmm(0));
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(mean_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups_maybe_tail(mean_ptr(), Vmm(1));

                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jl(mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        xor_(reg_soff, reg_soff);
        Label var_spatial;
        L(var_spatial);
        {
            xor_(reg_coff, reg_coff);

            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            is_nspc_ ? compute_mean_variance_nspc(false) : var_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                var_channels();

                sub(reg_src, vlen / 2);
            }

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_soff, mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(var_spatial);
        }

        if (is_nspc_) mov(reg_src, ptr[rsp + stack_off_src]); // comeback

        Label no_var_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_var_reduction);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label var_reduction_channels;
            L(var_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label var_reduction_thrs;
                L(var_reduction_thrs);
                { // TODO: unroll (?)
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(var_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups_maybe_tail(var_ptr(), Vmm(1));
                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jne(var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);

            if (bdesc_->use_scaleshift()) {
                uni_vmovups_maybe_tail(vgamma, gamma_ptr());
                uni_vmovups_maybe_tail(vbeta, beta_ptr());
            }

            Vmm vscale = bdesc_->use_scaleshift() ? vgamma : vone;
            Vmm vdiv = bdesc_->use_scaleshift() ? vgamma : vsqrtvar;

            if (isa == sse41) {
                movups(vbuf, vscale);
                divps(vbuf, vsqrtvar);
                movups(vdiv, vbuf);
            } else {
                vdivps(vdiv, vscale, vsqrtvar);
            }

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            Vmm v = Vmm(base_reg);
                            size_t offt = i * vlen_spat_data_;
                            uni_vmovups_spat_data(
                                    v, vmmword[reg_src + reg_soff + offt]);
                            mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                    + t1_pf_offt]);
                            uni_vsubps(v, v, vmean);
                            if (bdesc_->use_scaleshift()) {
                                uni_vfmadd213ps(v, vgamma, vbeta);
                            } else {
                                uni_vmulps(v, v, vsqrtvar);
                            }
                            if (with_relu_inf_only) {
                                uni_vmaxps(v, v, vzero);
                            } else if (with_relu) {
                                if (isa == avx512_common)
                                    fwd_process_relu_avx512_common(v, offt);
                                else
                                    fwd_process_relu_avx2(v, offt, Vmm(3));
                            }
                            if (stream_store_allowed) {
                                uni_vmovntps(
                                        vmmword[reg_dst + reg_soff + offt], v);
                            } else {
                                uni_vmovups_spat_data(
                                        vmmword[reg_dst + reg_soff + offt], v);
                            }
                        },
                        [](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                test(reg_dst, vlen - 1);
                jnz(normal_store, T_NEAR);
                compute(true);
                jmp(end_store, T_NEAR);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void forward_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                forward_channels_nspc_compute(ch_blk_size);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_dst, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, 2 * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_bf16_) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        sub(reg_dst, reg_coff_max);
        if (is_bf16_) shl(reg_coff_max, 1);

        shr(reg_coff_max, 5);
        sub(reg_ws, reg_coff_max);
        shl(reg_coff_max, 5);
    }

    void forward() {
        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_dst, ptr[rsp + stack_off_dst]);
        mov(reg_ws, ptr[rsp + stack_off_ws]);

        xor_(reg_soff, reg_soff);
        Label dst_spatial;
        L(dst_spatial);
        {
            xor_(reg_coff, reg_coff);
            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            is_nspc_ ? forward_channels_nspc() : forward_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                add(reg_dst, vlen / 2);
                mov(reg_coff, vlen / 2);

                forward_channels();

                sub(reg_src, vlen / 2);
                sub(reg_dst, vlen / 2);
            }

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_dst, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(dst_spatial);
        }

        if (is_nspc_) {
            // comeback
            mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_dst, ptr[rsp + stack_off_dst]);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }
    }

    void backward_sh_channels() {
        Label sh_channels;
        L(sh_channels);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            uni_vmovups(Vmm(1), vmmword[reg_rbuf2 + reg_coff]);
            spat_loop(
                    spat_size, 1, 1,
                    [=](size_t base_reg) {
                        if (base_reg > 0) {
                            for (int i = 0; i < 2; i++) {
                                Vmm v(base_reg * 5 + i);
                                uni_vpxor(v, v, v);
                            }
                        }
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm o0 = Vmm(base_reg * 5 + 0);
                        Vmm o1 = Vmm(base_reg * 5 + 1);
                        Vmm t1 = Vmm(base_reg * 5 + 2);
                        Vmm t2 = Vmm(base_reg * 5 + 3);
                        Vmm t3 = Vmm(base_reg * 5 + 4);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                t1, vmmword[reg_src + reg_soff + offt]);
                        uni_vmovups_spat_data(
                                t2, vmmword[reg_diff_dst + reg_soff + offt]);
                        if (with_relu) {
                            if (isa == avx512_common)
                                bwd_process_relu_avx512_common(t2, offt);
                            else if (isa == avx2)
                                bwd_process_relu_avx2(t2, offt, t3);
                            else
                                assert(false);
                        }
                        uni_vsubps(t3, vmean, t1, t3);
                        if (isa == sse41) {
                            mulps(t3, t2);
                            subps(o0, t3);
                        } else {
                            vfnmadd231ps(o0, t3, t2);
                        }
                        uni_vaddps(o1, o1, t2);
                        mic_prefetcht0(ptr[reg_diff_dst + reg_soff + offt
                                + t0_pf_offt]);
                        mic_prefetcht0(
                                ptr[reg_src + reg_soff + offt + t0_pf_offt]);
                        mic_prefetcht1(ptr[reg_diff_dst + reg_soff + offt
                                + t1_pf_offt]);
                        mic_prefetcht1(
                                ptr[reg_src + reg_soff + offt + t1_pf_offt]);
                    },
                    [=](size_t base_reg) {
                        Vmm b0 = Vmm(0);
                        Vmm b1 = Vmm(1);
                        if (base_reg) {
                            uni_vaddps(b0, b0, Vmm(base_reg * 5 + 0));
                            uni_vaddps(b1, b1, Vmm(base_reg * 5 + 1));
                        }
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(1));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(sh_channels);
        }
    }

    void backward_sh_channels_nspc_compute(const int num_ch_blks) {
        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            uni_vmovups(Vmm(idx++), vmmword[reg_rbuf1 + reg_coff + offt]);
            uni_vmovups(Vmm(idx++), vmmword[reg_rbuf2 + reg_coff + offt]);
        }

        xor_(reg_soff_nspc, reg_soff_nspc);

        if (is_spatial_thr_) {
            mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
            add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
        } else {
            mov(reg_ctr, spat_size);
        }

        // TODO: spatial blocking
        const int num_spat_pts = 1;

        Label spatial;
        L(spatial);
        {
            int coff = 0, offt = 0, sp_idx = 2 * num_ch_blks;
            for (int ch_idx = 0; ch_idx < 2 * num_ch_blks; ch_idx += 2) {
                uni_vmovups_maybe_tail(vmean, mean_ptr(coff));

                uni_vmovups_spat_data(
                        Vmm(sp_idx), vmmword[reg_src + reg_soff_nspc + offt]);
                uni_vmovups_spat_data(Vmm(sp_idx + 1),
                        vmmword[reg_diff_dst + reg_soff_nspc + offt]);

                if (with_relu) {
                    if (isa == avx512_common)
                        bwd_process_relu_avx512_common(Vmm(sp_idx + 1), offt);
                    else
                        assert(false);
                }

                uni_vsubps(
                        Vmm(sp_idx + 2), vmean, Vmm(sp_idx), Vmm(sp_idx + 2));
                vfnmadd231ps(Vmm(ch_idx), Vmm(sp_idx + 2), Vmm(sp_idx + 1));
                uni_vaddps(Vmm(ch_idx + 1), Vmm(ch_idx + 1), Vmm(sp_idx + 1));

                coff += vlen;
                offt += vlen_spat_data_;
                sp_idx += 3;
            }
            add(reg_soff_nspc, spat_step);
            sub(reg_ctr, num_spat_pts);
            jnz(spatial, T_NEAR);
        }

        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff + offt], Vmm(idx++));
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff + offt], Vmm(idx++));
        }
    }

    void backward_sh_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                backward_sh_channels_nspc_compute(ch_blk_size);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_diff_dst, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, 2 * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        mov(reg_diff_scale_shift, ptr[rsp + stack_off_diff_scale_shift]);

        if (is_bf16_) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        sub(reg_diff_dst, reg_coff_max);
        if (is_bf16_) shl(reg_coff_max, 1);

        if (with_relu) {
            shr(reg_coff_max, 5);
            sub(reg_ws, reg_coff_max);
            shl(reg_coff_max, 5);
        }
    }

    void backward_diff_channels() {
        Label diff_channels;
        L(diff_channels);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);
            uni_vdivps(vsqrtvar, vone, vsqrtvar, vbuf);
            if (bdesc_->use_scaleshift())
                uni_vmovups_maybe_tail(vgamma, gamma_ptr());
            uni_vmovups_maybe_tail(vdiff_gamma, diff_gamma_ptr());
            uni_vmovups_maybe_tail(vdiff_beta, diff_beta_ptr());
            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
            uni_vdivps(vdiff_beta, vdiff_beta, vchan_size);
            uni_vdivps(vdiff_gamma, vdiff_gamma, vchan_size);

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [=](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            Vmm v(base_reg * 2 + 0);
                            Vmm t(base_reg * 2 + 1);
                            Vmm t1(base_reg * 2 + 2);
                            size_t offt = i * vlen_spat_data_;
                            uni_vmovups_spat_data(
                                    v, vmmword[reg_diff_dst + reg_soff + offt]);
                            if (with_relu) {
                                if (isa == avx512_common)
                                    bwd_process_relu_avx512_common(v, offt);
                                else if (isa == avx2)
                                    bwd_process_relu_avx2(v, offt, t);
                                else
                                    assert(false);
                            }
                            if (!bdesc_->use_global_stats()) {
                                uni_vsubps(v, v, vdiff_beta);
                                uni_vmovups_spat_data(
                                        t, vmmword[reg_src + reg_soff + offt]);
                                uni_vsubps(t, vmean, t, t1);
                                uni_vmulps(t, t, vdiff_gamma);
                                uni_vaddps(v, v, t);
                            }
                            uni_vmulps(v, v, vsqrtvar);
                            if (bdesc_->use_scaleshift()) {
                                uni_vmulps(v, v, vgamma);
                            }
                            if (stream_store_allowed) {
                                uni_vmovntps(
                                        vmmword[reg_diff_src + reg_soff + offt],
                                        v);
                            } else {
                                uni_vmovups_spat_data(
                                        vmmword[reg_diff_src + reg_soff + offt],
                                        v);
                            }
                            mic_prefetcht0(ptr[reg_diff_dst + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht0(ptr[reg_src + reg_soff + offt
                                    + t0_pf_offt]);
                            mic_prefetcht1(ptr[reg_diff_dst + reg_soff + offt
                                    + t1_pf_offt]);
                            mic_prefetcht1(ptr[reg_src + reg_soff + offt
                                    + t1_pf_offt]);
                        },
                        [=](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                test(reg_diff_src, vlen - 1);
                jnz(normal_store, T_NEAR);
                compute(true);
                jmp(end_store, T_NEAR);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(diff_channels);
        }
    }

    void backward_diff_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            xor_(reg_soff_nspc, reg_soff_nspc);
            if (is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < 3 * num_ch_blks; idx += 3) {
                    uni_vmovups_maybe_tail(vmean, mean_ptr(coff));
                    uni_vmovups_maybe_tail(vsqrtvar, var_ptr(coff));

                    uni_vaddps(vsqrtvar, vsqrtvar, veps);
                    uni_vsqrtps(vsqrtvar, vsqrtvar);
                    uni_vdivps(vsqrtvar, vone, vsqrtvar, vbuf);

                    if (bdesc_->use_scaleshift())
                        uni_vmovups_maybe_tail(vgamma, gamma_ptr(coff));

                    mov(ptr[rsp + stack_off_ws_off_copy], reg_ws);
                    mov(reg_ws, ptr[rsp + stack_off_diff_scale_shift]);
                    uni_vmovups_maybe_tail(
                            vdiff_gamma, vmmword[reg_ws + reg_coff + coff]);
                    uni_vmovups_maybe_tail(vdiff_beta,
                            vmmword[reg_ws + reg_coff + coff
                                    + 1 * chan_data_offt]);
                    mov(reg_ws, ptr[rsp + stack_off_ws_off_copy]);

                    uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
                    uni_vdivps(vdiff_beta, vdiff_beta, vchan_size);
                    uni_vdivps(vdiff_gamma, vdiff_gamma, vchan_size);

                    uni_vmovups_spat_data(Vmm(idx),
                            vmmword[reg_diff_dst + reg_soff_nspc + offt]);

                    if (with_relu) {
                        if (isa == avx512_common)
                            bwd_process_relu_avx512_common(Vmm(idx), offt);
                        else
                            assert(false);
                    }

                    if (!bdesc_->use_global_stats()) {
                        uni_vsubps(Vmm(idx), Vmm(idx), vdiff_beta);
                        uni_vmovups_spat_data(Vmm(idx + 1),
                                vmmword[reg_src + reg_soff_nspc + offt]);
                        uni_vsubps(Vmm(idx + 1), vmean, Vmm(idx + 1),
                                Vmm(idx + 2));
                        uni_vmulps(Vmm(idx + 1), Vmm(idx + 1), vdiff_gamma);
                        uni_vaddps(Vmm(idx), Vmm(idx), Vmm(idx + 1));
                    }

                    uni_vmulps(Vmm(idx), Vmm(idx), vsqrtvar);

                    if (bdesc_->use_scaleshift()) {
                        uni_vmulps(Vmm(idx), Vmm(idx), vgamma);
                    }

                    if (stream_store_allowed) {
                        uni_vmovntps(
                                vmmword[reg_diff_src + reg_soff_nspc + offt],
                                Vmm(idx));
                    } else {
                        uni_vmovups_spat_data(
                                vmmword[reg_diff_src + reg_soff_nspc + offt],
                                Vmm(idx));
                    }

                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add(reg_soff_nspc, spat_step);
                sub(reg_ctr, num_spat_pts);
                jnz(spatial, T_NEAR);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            test(reg_diff_src, vlen - 1);
            jnz(normal_store, T_NEAR);
            compute(true);
            jmp(end_store, T_NEAR);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void backward_diff_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                backward_diff_channels_nspc_compute(ch_blk_size);

                add(reg_diff_dst, vlen_spat_data_ * ch_blk_size);
                if (!bdesc_->use_global_stats())
                    add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_diff_src, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, 2 * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        mov(reg_diff_scale_shift, ptr[rsp + stack_off_diff_scale_shift]);

        if (is_bf16_) shr(reg_coff_max, 1);
        sub(reg_diff_dst, reg_coff_max);
        if (!bdesc_->use_global_stats()) sub(reg_src, reg_coff_max);
        sub(reg_diff_src, reg_coff_max);
        if (is_bf16_) shl(reg_coff_max, 1);

        shr(reg_coff_max, 5);
        sub(reg_ws, reg_coff_max);
        shl(reg_coff_max, 5);
    }

    void backward() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf, sh_spatial;

        L(zero_rbuf);
        {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(0));
            add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
        if (with_relu) {
            assert(isa == avx2 || isa == avx512_common);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }

        xor_(reg_soff, reg_soff);
        L(sh_spatial);
        {
            xor_(reg_coff, reg_coff);
            if (isa == sse41) { mov(reg_tmp_off, reg_soff); }
            is_nspc_ ? backward_sh_channels_nspc() : backward_sh_channels();
            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_diff_dst, vlen / 2);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);
                backward_sh_channels();
                sub(reg_diff_dst, vlen / 2);
                sub(reg_src, vlen / 2);
            }
            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_diff_dst, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }
            cmp(reg_soff, reg_soff_max);
            jl(sh_spatial);
        }

        if (is_nspc_) {
            // comeback
            mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
        }

        mov(reg_diff_scale_shift, ptr[rsp + stack_off_diff_scale_shift]);

        Label no_sh_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            Label sh_reduction_channels;
            jne(no_sh_reduction, T_NEAR);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            L(sh_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
                uni_vaddps(vsqrtvar, vsqrtvar, veps);
                uni_vsqrtps(vsqrtvar, vsqrtvar);
                uni_vdivps(vsqrtvar, vone, vsqrtvar, vbuf);
                mov(reg_ctr, reg_nnthr);
                Label sh_reduction_thrs;
                L(sh_reduction_thrs);
                { // TODO: unroll (?)
                    uni_vaddps(Vmm(0), Vmm(0), vmmword[reg_rbuf1 + reg_roff]);
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf2 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(sh_reduction_thrs);
                }
                uni_vmulps(Vmm(0), Vmm(0), vsqrtvar);
                uni_vmovups_maybe_tail(diff_gamma_ptr(), Vmm(0));
                uni_vmovups_maybe_tail(diff_beta_ptr(), Vmm(1));
                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
                cmp(reg_coff, reg_coff_max);
                jne(sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        mov(reg_diff_src, ptr[rsp + stack_off_diff_src]);
        if (with_relu) {
            assert(isa == avx2 || isa == avx512_common);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }

        xor_(reg_soff, reg_soff);
        Label diff_spatial;
        L(diff_spatial);
        {
            xor_(reg_coff, reg_coff);
            if (isa == sse41) { mov(reg_tmp_off, reg_soff); }
            is_nspc_ ? backward_diff_channels_nspc() : backward_diff_channels();
            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_diff_dst, vlen / 2);
                add(reg_diff_src, vlen / 2);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);
                backward_diff_channels();
                sub(reg_diff_dst, vlen / 2);
                sub(reg_diff_src, vlen / 2);
                sub(reg_src, vlen / 2);
            }
            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (!bdesc_->use_global_stats()) add(reg_src, mb_offt);
                add(reg_diff_dst, mb_offt);
                add(reg_diff_src, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }
            cmp(reg_soff, reg_soff_max);
            jl(diff_spatial);
        }
        if (is_nspc_) {
            // comeback
            if (!bdesc_->use_global_stats())
                mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
            mov(reg_diff_src, ptr[rsp + stack_off_diff_src]);
            if (with_relu) mov(reg_ws, ptr[rsp + stack_off_ws]);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *bdesc) : bdesc_(bdesc) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_common
                        || isa == avx512_mic,
                "unsupported isa");

        const int simd_w = isa == sse41
                ? 8
                : cpu_isa_traits<isa>::vlen / sizeof(acc_data_t);
        is_bf16_ = bdesc_->desc()->data_desc.data_type == data_type::bf16;
        size_t dt_size
                = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);
        is_spatial_thr_ = bnorm_utils::is_spatial_thr(
                bdesc_, is_nspc_, simd_w, dt_size);
        vlen_spat_data_ = vlen / (1 + is_bf16_); // 32B of BF16 -> 64B of FP32

        unroll_blocks = isa == avx512_common && !is_spatial_thr_ ? 4 : 1;
        unroll_regs = isa == avx512_common && !is_spatial_thr_ ? 4 : 1;
    }

    void generate() override {
        preamble();

        if (is_bf16_) {
            // init emulation of bfloat16 operations
            if (!mayiuse(avx512_core_bf16)) {
                bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserved_1,
                        bf16_emu_reserved_2, bf16_emu_reserved_3, reg_bf16_tmp,
                        bf16_emu_reserved_4, bf16_emu_reserved_4);
                bf16_emu_->init_vcvtneps2bf16();
            }
        }

        if (isa == avx512_common)
            prepare_tail_mask_avx512_common();
        else if (isa == avx2)
            prepare_tail_mask_avx2_common();

        compute_static_strides();
        sub(rsp, stack_size_required);
        load_common_params();
        prepare_relu();

        if (bdesc_->is_fwd()) {
            if (!bdesc_->stats_is_src()) { compute_mean_variance(); }
            forward();
        } else {
            backward();
        }
        add(rsp, stack_size_required);
        postamble();
    }

    void operator()(const call_params_t *p) { jit_generator::operator()(p); }

    ~jit_bnorm_t() override { delete bf16_emu_; }
};
} // namespace

namespace bnorm_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc), ker_(bdesc_) {
        const dim_t C_PADDED = get_c_padded(bdesc_);

        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);

        dt_size_ = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        size_t data_size = dt_size_ * bdesc_->MB() * C_PADDED * bdesc_->D()
                * bdesc_->H() * bdesc_->W();
        l3_size_ = platform::get_per_core_cache_size(3) * dnnl_get_max_threads()
                / 2; // XXX
        // TODO: cache balancing for nspc
        do_blocking_ = is_nspc_ ? false
                                : (data_size >= l3_size_ / 2 && l3_size_ > 0);
    }

    ~driver_t() = default;

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *bdesc) {
        dim_t C_PADDED = get_c_padded(bdesc);

        int sbuf_sz = use_tmp_stats(bdesc) * 2 * C_PADDED;
        int pbuf_sz = use_tmp_diff_scale_shift(bdesc) * 2 * C_PADDED;
        int rbuf_sz
                = (bdesc->is_fwd() ? 1 : 2) * C_PADDED * dnnl_get_max_threads();

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);

        if (dnnl_thr_syncable()) {
            int n_barriers = C_PADDED / simd_w;
            scratchpad.book<barrier::ctx_64_t>(key_barrier, n_barriers);
        }
    }

    void exec(int ithr, int nthr, const void *src, void *diff_src, void *dst,
            const void *diff_dst, const acc_data_t *scale_shift,
            acc_data_t *diff_scale_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
        auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);

        dim_t N = bdesc_->MB();
        dim_t C = bdesc_->C();
        dim_t C_PADDED = get_c_padded(bdesc_);
        dim_t D = bdesc_->D();
        dim_t H = bdesc_->H();
        dim_t W = bdesc_->W();
        dim_t SP = D * H * W;
        dim_t img_size = C_PADDED * D * H * W;
        const int vlen_spat_data = ker_.spat_step;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.0f;
        p.spat_size = D * H * W;
        p.chan_size = 1.0f * N * p.spat_size;

        dim_t C_blks = C_PADDED / simd_w;

        int C_ithr {0}, C_nthr {0}, N_ithr {0}, N_nthr {0}, S_ithr {0},
                S_nthr {0};
        dim_t C_blk_s {0}, C_blk_e {0}, N_s {0}, N_e {0}, S_s {0}, S_e {0};

        dim_t C_blks_per_iter {1};
        int64_t iters {1};
        if (do_blocking_) {
            int num_tensors = bdesc_->is_fwd() ? 1 : 2;
            size_t working_set_size
                    = dt_size_ * (N * D * H * W * simd_w) * num_tensors;
            bnorm_utils::cache_balance(
                    working_set_size, C_blks, N, nthr, C_blks_per_iter, iters);
        }

        bool spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                true /* spatial_thr_allowed */, is_nspc_, ithr, nthr, N,
                do_blocking_ ? C_blks_per_iter : C_blks, SP,
                /* outputs */ C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr,
                N_s, N_e, S_ithr, S_nthr, S_s, S_e);

        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;
        assert(IMPLICATION(!dnnl_thr_syncable(), SP_N_nthr == 1));

        p.N_ithr = SP_N_ithr;
        p.N_nthr = SP_N_nthr;

        int last_iter_blks = C_blks - (iters - 1) * C_blks_per_iter;
        int global_C_blk_s;
        int global_barriers_per_iter = C_nthr;

        for (int64_t it = 0; it < iters; it++) {
            if (it == iters - 1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                        spatial_thr_allowed, is_nspc_, ithr, nthr, N,
                        last_iter_blks, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);

                // Update call parameters for JIT, last iteration
                p.N_ithr = N_ithr * S_nthr + S_ithr;
                p.N_nthr = N_nthr * S_nthr;
            }

            global_C_blk_s = do_blocking_
                    ? (C_blk_s == -1) ? -1 : it * C_blks_per_iter + C_blk_s
                    : C_blk_s;

            int C_blks_thr = C_blk_e - C_blk_s;
            int N_thr = N_e - N_s;

            size_t coff_base = global_C_blk_s * simd_w;
            size_t soff_base = is_nspc_
                    ? coff_base + N_s * img_size
                    : global_C_blk_s * p.spat_size * simd_w + N_s * img_size;

            p.spat_size_loc = S_e - S_s;
            p.S_s = S_s * vlen_spat_data;
            p.S_tail = (p.spat_size - S_e) * vlen_spat_data;
            p.coff_max = C_blks_thr * simd_w;
            p.mean = (use_tmp_stats(bdesc_) ? sbuf : mean) + coff_base;
            p.var = (use_tmp_stats(bdesc_) ? sbuf + C_PADDED : var) + coff_base;
            p.scale_shift = scale_shift + coff_base;
            p.diff_scale_shift
                    = (use_tmp_diff_scale_shift(bdesc_) ? pbuf
                                                        : diff_scale_shift)
                    + coff_base;

            p.soff_max = dt_size_ * N_thr * img_size;
            p.src = (void *)((char *)src + soff_base * dt_size_);
            p.dst = (void *)((char *)dst + soff_base * dt_size_);
            p.diff_src = (void *)((char *)diff_src + soff_base * dt_size_);
            p.diff_dst = (void *)((char *)diff_dst + soff_base * dt_size_);
            p.ws = ws + soff_base / 8;

            p.mb_stride_Bc = dt_size_ * (img_size - p.coff_max * p.spat_size);

            // use SP_N_nthr which is the same as p.N_nthr except maybe for
            // the last iteration.
            p.rbuf1 = rbuf
                    + ((it * C_blks_per_iter) * SP_N_nthr + C_blk_s * p.N_nthr
                              + p.N_ithr * C_blks_thr)
                            * simd_w;
            // rbuf1 and rbuf2 have to be disjoint
            p.rbuf2 = p.rbuf1 + C_PADDED * nthr;
            p.is_cblk_tail = (it * C_blks_per_iter + C_blk_e) * simd_w > C;

            size_t iter_bariers
                    = do_blocking_ ? it * global_barriers_per_iter : 0;
            p.barrier = barriers + C_ithr + iter_bariers;
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    void init_barriers(const memory_tracking::grantor_t &scratchpad) {
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);
        if (barriers) {
            const int n_barriers = get_c_padded(bdesc_) / simd_w;
            for (int i = 0; i < n_barriers; ++i)
                barrier::ctx_init(&barriers[i]);
        }
    }

    status_t create_kernel() { return ker_.create_kernel(); }

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

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_t<isa> ker_;
    bool do_blocking_;
    bool is_nspc_;
    size_t l3_size_;
    size_t dt_size_;
};
} // namespace bnorm_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_fwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && one_of(src_md()->data_type, f32, bf16)
            && IMPLICATION(src_md()->data_type == bf16, mayiuse(avx512_core))
            && check_scale_shift_data_type()
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    if (isa == avx512_common) {
        if (!src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc))
            return status::unimplemented;
    } else {
        if (!src_d.matches_one_of_tag(nChw8c, nCdhw8c))
            return status::unimplemented;
    }

    if (is_training() && fuse_norm_relu()) {
        if (isa < avx2) return status::unimplemented;
        init_default_ws(1);
    }

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < avx2)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute(
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

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, nullptr, dst, nullptr, scale_shift,
                nullptr, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_bwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && set_default_formats_common()
            && one_of(true,
                    everyone_is(
                            f32, src_md()->data_type, diff_src_md()->data_type),
                    everyone_is(bf16, src_md()->data_type,
                            diff_src_md()->data_type))
            && IMPLICATION(src_md()->data_type == bf16, mayiuse(avx512_core))
            && check_scale_shift_data_type() && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());

    format_tag_t src_tag, diff_src_tag;
    if (isa == avx512_common) {
        src_tag = src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc);
        diff_src_tag
                = diff_src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc);
    } else {
        src_tag = src_d.matches_one_of_tag(nChw8c, nCdhw8c);
        diff_src_tag = diff_src_d.matches_one_of_tag(nChw8c, nCdhw8c);
    }
    ok = (src_tag != format_tag::undef && diff_src_tag != format_tag::undef
            && src_tag == diff_src_tag);
    if (!ok) return status::unimplemented;

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < avx2)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    if (fuse_norm_relu()) {
        if (isa < avx2) return status::unimplemented;
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    /* TODO: extra checks required */

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute(
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

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, diff_src, nullptr, diff_dst,
                scale_shift, diff_scale_shift, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<sse41>;
template struct jit_uni_batch_normalization_bwd_t<sse41>;
template struct jit_uni_batch_normalization_fwd_t<avx2>;
template struct jit_uni_batch_normalization_bwd_t<avx2>;
template struct jit_uni_batch_normalization_fwd_t<avx512_common>;
template struct jit_uni_batch_normalization_bwd_t<avx512_common>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
