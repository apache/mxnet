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

#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/bf16/jit_avx512_core_gemm_bf16bf16f32_kern.hpp"

#ifdef _WIN32
static const bool is_windows = true;
#else
static const bool is_windows = false;
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

// Convert between vector register lengths.
static inline Xmm make_xmm(const Xmm &v) {
    return Xmm(v.getIdx());
}
static inline Ymm make_ymm(const Xmm &v) {
    return Ymm(v.getIdx());
}
static inline Zmm make_zmm(const Xmm &v) {
    return Zmm(v.getIdx());
}

// Load from or store to C.
void jit_avx512_core_gemm_bf16bf16f32_kern::c_load(
        const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems) {
    switch (nelems) {
        case 1: vmovss(make_xmm(dst), src); break;
        case 2: vmovsd(make_xmm(dst), src); break;
        case 4: vmovups(make_xmm(dst), src); break;
        case 8: vmovups(make_ymm(dst), src); break;
        default:
            assert(nelems >= 16);
            vmovups(dst, src);
            break;
    }
}

void jit_avx512_core_gemm_bf16bf16f32_kern::c_store(
        const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems) {
    switch (nelems) {
        case 1: vmovss(dst, make_xmm(src)); break;
        case 2: vmovlps(dst, make_xmm(src)); break;
        case 4: vmovups(dst, make_xmm(src)); break;
        case 8: vmovups(dst, make_ymm(src)); break;
        default:
            assert(nelems >= 16);
            vmovups(dst, src);
            break;
    }
}

// Perform length-2 dot product accumulations of bfloat16 in parallel.
// Use vdpbf16ps if available, otherwise emulate.
void jit_avx512_core_gemm_bf16bf16f32_kern::dot_product(
        const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    if (bfloat16_)
        vdpbf16ps(dst, src1, src2);
    else
        bf16_emu_->vdpbf16ps(make_zmm(dst), make_zmm(src1), make_zmm(src2));
}

// Inner kernel.
void jit_avx512_core_gemm_bf16bf16f32_kern::kernel_loop(
        int unroll_m, int unroll_n, bool cfetch) {
    int um_vecs = utils::div_up(unroll_m, c_nelems_);
    Label label_kernel_loop;

    L_aligned(label_kernel_loop);
    {
        for (int h = 0; h < 4; h++) {
            for (int j = 0; j < unroll_n; j++) {
                const auto b = b_regs_[j & 1];

                vbroadcastss(b,
                        ptr[BO_
                                + isize_
                                        * (2 * j + 2 * h * unroll_n
                                                - offset_b_)]);
                dot_product(c_regs_[0][j], b, a_regs_[0]);

                if (j == 1 && !(h & 1))
                    prefetch_b(ptr[BO_
                            + isize_
                                    * (prefetch_size_b_ + 2 * h * unroll_n
                                            - offset_b_)]);
                else if (j % 3 == 0)
                    prefetch_a(ptr[AO_
                            + isize_
                                    * (prefetch_size_a_ + a_nelems_ * (j / 3)
                                            + 2 * h * unroll_m - offset_a_)]);

                for (int i = 1; i < um_vecs; i++)
                    dot_product(c_regs_[i][j], b, a_regs_[i]);

                if (cfetch && (j == std::min(1, unroll_n - 1))) {
                    if (h == 3)
                        lea(CO2_, ptr[CO2_ + LDC_]);
                    else if (h < um_vecs)
                        prefetch_c(ptr[CO2_ + (16 * h * size_)]);
                }

                if (h == 3 && j == std::min(3, unroll_n - 1))
                    lea(AA_, ptr[AA_ + (32 * isize_)]);
            }

            for (int i = 0; i < um_vecs; i++)
                vmovups(a_regs_[i],
                        ptr[AO_
                                + isize_
                                        * (a_nelems_ * i
                                                + 2 * (h + 1) * unroll_m
                                                - offset_a_)]);

            if (h == 2) prefetch_x(ptr[AA_ - (offset_a_ * isize_)]);
        }

        add(AO_, 8 * isize_ * unroll_m);
        add(BO_, 8 * isize_ * unroll_n);
        sub(LoopCount_, 1);
        jg(label_kernel_loop, T_NEAR);
    }
}

// k remainder loop for kernel.
void jit_avx512_core_gemm_bf16bf16f32_kern::remainder_kernel(
        int unroll_m, int unroll_n, int unroll_k, int bwidth) {
    int um_vecs = utils::div_up(unroll_m, c_nelems_);

    for (int h = 0; h < unroll_k; h++) {
        for (int j = 0; j < unroll_n; j++) {
            auto b = b_regs_[j & 1];
            auto b_src = ptr[BO_
                    + (-isize_ * offset_b_ + bwidth * (j + h * unroll_n))];

            switch (bwidth) {
                case 4: vbroadcastss(b, b_src); break;
                case 2: vpbroadcastw(b, b_src); break;
            }
            for (int i = 0; i < um_vecs; i++)
                dot_product(c_regs_[i][j], b, a_regs_[i]);
        }

        if (unroll_k > 1) {
            for (int i = 0; i < um_vecs; i++)
                vmovups(a_regs_[i],
                        ptr[AO_
                                + isize_
                                        * (a_nelems_ * i
                                                + (h + 1) * 2 * unroll_m
                                                - offset_a_)]);
        }
    }

    add(AO_, unroll_k * unroll_m * bwidth);
    add(BO_, unroll_k * unroll_n * bwidth);
}

// Inner loop.
void jit_avx512_core_gemm_bf16bf16f32_kern::innerloop(
        int unroll_m, int unroll_n) {
    int um_vecs = utils::div_up(unroll_m, c_nelems_);
    int stage1 = unroll_n, stage2 = unroll_n;

    Label label_kernel_loop_1, label_k_main_loop_2, label_kernel_loop_2;
    Label label_k_main_loop_3, label_kernel_loop_3;
    Label label_k_remainder_loop_begin, label_k_rem_2;
    Label label_k_rem_1, label_update_begin;

    mov(AO_, A_);
    for (int i = 0; i < um_vecs; i++)
        vmovups(a_regs_[i], ptr[AO_ + isize_ * (a_nelems_ * i - offset_a_)]);

    mov(LoopCount_, K_);
    sar(LoopCount_, 3);
    jle(label_k_remainder_loop_begin, T_NEAR);

    // Main k loops, broken into three parts to time C prefetching.
    sub(LoopCount_, stage1 + stage2);
    jle(label_k_main_loop_2, T_NEAR);

    kernel_loop(unroll_m, unroll_n, false);

    L_aligned(label_k_main_loop_2);
    lea(CO2_, ptr[CO1_ + size_ * (std::min(unroll_m, c_nelems_) - 1)]);
    add(LoopCount_, stage1);
    jle(label_k_main_loop_3, T_NEAR);

    kernel_loop(unroll_m, unroll_n, true);

    L_aligned(label_k_main_loop_3);
    lea(CO2_, ptr[CO1_ + size_ * (std::min(unroll_m, c_nelems_) - 1)]);
    add(LoopCount_, stage2);
    jle(label_k_remainder_loop_begin, T_NEAR);

    kernel_loop(unroll_m, unroll_n, true);

    // k remainder handling
    L_aligned(label_k_remainder_loop_begin);
    mov(LoopCount_, K_);
    test(LoopCount_, 4);
    je(label_k_rem_2, T_NEAR);

    remainder_kernel(unroll_m, unroll_n, 2, 4);

    L_aligned(label_k_rem_2);
    mov(LoopCount_, K_);
    test(LoopCount_, 2);
    je(label_k_rem_1, T_NEAR);

    remainder_kernel(unroll_m, unroll_n, 1, 4);

    L_aligned(label_k_rem_1);
    mov(LoopCount_, K_);
    test(LoopCount_, 1);
    je(label_update_begin, T_NEAR);

    auto zero = Xmm(kind_, 6);

    vpxorq(zero, zero, zero);
    for (int i = 0; i < um_vecs; i++) {

        if (kind_ == Operand::ZMM) {
            Zmm a = make_zmm(a_regs_[i]);
            Zmm tmp = zmm5;

            vbroadcasti64x4(
                    a, ptr[AO_ + isize_ * (a_nelems_ / 2 * i - offset_a_)]);
            vpunpcklwd(tmp, a, zero);
            vpunpckhwd(a, a, zero);
            vshufi32x4(a, tmp, a, 0x44);
            vshufi32x4(a, a, a, 0xD8);
        } else { // kind_ == Operand::YMM
            Ymm a = make_ymm(a_regs_[i]);
            Ymm tmp = ymm5;

            vbroadcasti64x2(
                    a, ptr[AO_ + isize_ * (a_nelems_ / 2 * i - offset_a_)]);
            vpunpcklwd(tmp, a, zero);
            vpunpckhwd(a, a, zero);
            vshufi32x4(a, tmp, a, 0x44);
        }
    }

    remainder_kernel(unroll_m, unroll_n, 1, 2);

    // Update C.
    L_aligned(label_update_begin);

    Reg64 LDC3 = rax;
    lea(LDC3, ptr[LDC_ + LDC_ * 2]);

    // C updates.
    int c_off_j = 0;
    for (int j = 0; j < unroll_n; j++) {
        if (j > 0 && (j & 3) == 0) {
            lea(CO1_, ptr[CO1_ + LDC_ * 4]);
            c_off_j += 4;
        }

        int jj = j - c_off_j;

        for (int i = 0; i < um_vecs; i++) {
            auto c = c_regs_[i][j];
            auto c_old = Xmm(kind_, 0);
            decltype(LDC_ * jj) ldc_mult = (jj == 3) ? LDC3 : LDC_ * jj;

            auto c_mem = ptr[CO1_ + ldc_mult + size_ * c_nelems_ * i];

            if (beta_zero_) {
                if (!alpha_one_) vmulps(c, c, alpha_);
                c_store(c_mem, c, unroll_m);
            } else {
                c_load(c_old, c_mem, unroll_m);
                if (alpha_one_)
                    vaddps(c_old, c, c_old);
                else
                    vfmadd231ps(c_old, c, alpha_);
                c_store(c_mem, c_old, unroll_m);
            }

            vpxorq(c, c, c);
        }
    }

    lea(CO1_, ptr[CO1_ + LDC_ * (unroll_n - c_off_j)]);
}

// Outer loop.
void jit_avx512_core_gemm_bf16bf16f32_kern::outerloop(
        int unroll_x, int unroll_y, Label *&cur_outerloop_label) {
    Label label_m_loop, label_n_loop, label_n_remainder_loops[6];

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_x >= UNROLL_M_) {
        mov(J_, M_);
        cmp(J_, unroll_x);
        jl(*cur_outerloop_label, T_NEAR); // Jump to next outerloop label.
    } else {
        test(J_, unroll_x);
        jle(*cur_outerloop_label, T_NEAR);
    }

    L_aligned(label_m_loop);
    {
        mov(CO1_, C_);
        add(C_, unroll_x * size_);

        mov(BO_, B_);

        mov(AA_, K_);
        imul(AA_, AA_, unroll_x * isize_);
        lea(AA_, ptr[A_ + AA_ + isize_ * prefetch_size_a_]);

        mov(I_, N_);
        cmp(I_, unroll_y);
        jl(label_n_remainder_loops[0], T_NEAR);

        L_aligned(label_n_loop);
        {
            innerloop(unroll_x, unroll_y);
            sub(I_, unroll_y);
            cmp(I_, unroll_y);
            jge(label_n_loop, T_NEAR);
        }

        align(16);

        int label_idx = 0;
        for (int uy = 16; uy > 0; uy >>= 1) {
            L(label_n_remainder_loops[label_idx++]);
            if (unroll_y > uy) {
                test(I_, uy);
                jle(label_n_remainder_loops[label_idx], T_NEAR);

                innerloop(unroll_x, uy);
                align(16);
            }
        }
        L(label_n_remainder_loops[label_idx]);

        mov(A_, AO_);
        if (unroll_x >= UNROLL_M_) {
            sub(J_, unroll_x);
            cmp(J_, unroll_x);
            jge(label_m_loop);
        }
    }

    align(16);
}

void jit_avx512_core_gemm_bf16bf16f32_kern::generate() {
    // Prologue
    preamble();
    sub(rsp, stack_alloc_size_);

    if (!alpha_one_) vbroadcastss(alpha_, qword[ALPHA_]);

    if (is_windows) {
        mov(A_, arg_a_);
        mov(B_, arg_b_);
    }

    mov(C_, arg_c_);
    mov(LDC_, arg_ldc_);

    sub(A_, -offset_a_ * isize_);
    sub(B_, -offset_b_ * isize_);

    mov(M_, qword[M_]);
    mov(N_, qword[N_]);
    mov(K_, qword[K_]);

    lea(LDC_, ptr[LDC_ * size_]);

    for (int i = 0; i < (max_unroll_m_ >> 4); i++) {
        for (int j = 0; j < max_unroll_n_; j++) {
            auto &c = c_regs_[i][j];
            vpxorq(c, c, c);
        }
    }

    Label outerloop_labels[8];
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main m loop.
    outerloop(UNROLL_M_, UNROLL_N_, cur_outerloop_label);

    // m remainder loops.
    for (int um = 32; um > 0; um >>= 1)
        if (UNROLL_M_ > um) outerloop(um, UNROLL_N_, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    add(rsp, stack_alloc_size_);
    postamble();
}

jit_avx512_core_gemm_bf16bf16f32_kern::jit_avx512_core_gemm_bf16bf16f32_kern(
        bool beta_zero, bool alpha_one, bool use_zmm)
    : jit_generator(nullptr, 170000)
    , arg_a_(0)
    , arg_b_(0)
    , arg_c_(0)
    , arg_ldc_(0)
    , arg_coffset_c_(0)
    , arg_coffset_r_(0) {

    beta_zero_ = beta_zero;
    alpha_one_ = alpha_one;
    bfloat16_ = mayiuse(avx512_core_bf16);
    assert(mayiuse(avx512_core));
    assert(IMPLICATION(!use_zmm, bfloat16_));

    // Assign integer registers
    M_ = is_windows ? rcx : rdi;
    N_ = is_windows ? rdx : rsi;
    K_ = is_windows ? r8 : rdx;
    ALPHA_ = is_windows ? r9 : rcx;
    A_ = is_windows ? rsi : r8;
    B_ = r9;
    C_ = r10;
    LDC_ = r11;
    I_ = r12;
    J_ = r13;
    LoopCount_ = rax;
    AO_ = r14;
    BO_ = r15;
    CO1_ = rbx;
    CO2_ = rbp;
    AA_ = is_windows ? rdi : rcx;

    UNROLL_M_ = use_zmm ? 48 : 24;
    kind_ = use_zmm ? Operand::ZMM : Operand::YMM;

    // Assign vector registers.
    alpha_ = Xmm(kind_, 7);
    for (int i = 0; i < (max_unroll_m_ >> 4); i++)
        a_regs_[i] = Xmm(kind_, i);
    b_regs_[0] = Xmm(kind_, 4);
    b_regs_[1] = Xmm(kind_, 5);

    int rn = 0;
    for (int i = 0; i < (max_unroll_m_ >> 4); i++)
        for (int j = 0; j < max_unroll_n_; j++)
            c_regs_[i][j] = Xmm(kind_, 8 + rn++);

    // Set number elements in a vector register.
    a_nelems_ = a_regs_[0].getBit() / (isize_ * 8);
    b_nelems_ = b_regs_[0].getBit() / (isize_ * 8);
    c_nelems_ = c_regs_[0][0].getBit() / (size_ * 8);

    // Assign stack variables.
    stack_alloc_size_ = 32;
    auto args_offset = stack_alloc_size_ + get_size_of_abi_save_regs() + 8
            + (is_windows ? 48 : 0);

    arg_a_ = ptr[rsp + (args_offset - 16)];
    arg_b_ = ptr[rsp + (args_offset - 8)];
    arg_c_ = ptr[rsp + (args_offset + 0)];

    arg_ldc_ = ptr[rsp + (args_offset + 8)];

    arg_coffset_c_ = ptr[rsp + (args_offset + 16)];
    arg_coffset_r_ = ptr[rsp + (args_offset + 24)];

    bf16_emu_ = nullptr;
    if (!bfloat16_ && use_zmm) {

        // Those register will not be used by bf16 emulation since we only use
        // r_vdpbf16ps.
        scratch_ = rax;
        one_ = zmm6;
        even_ = zmm6;
        selector_ = zmm6;

        zmm_tmp0_ = zmm6;
        zmm_tmp1_ = zmm3;
        bf16_emu_ = new bf16_emulation_t(
                this, one_, even_, selector_, scratch_, zmm_tmp0_, zmm_tmp1_);
    }
}

jit_avx512_core_gemm_bf16bf16f32_kern::
        ~jit_avx512_core_gemm_bf16bf16f32_kern() {
    delete bf16_emu_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
