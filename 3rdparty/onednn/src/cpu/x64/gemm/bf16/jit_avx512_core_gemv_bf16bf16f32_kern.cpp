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

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/bf16/jit_avx512_core_gemv_bf16bf16f32_kern.hpp"

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

// Perform length-2 dot product accumulations of bfloat16 in parallel.
// Use vdpbf16ps if available, otherwise emulate.
void jit_avx512_core_gemv_bf16bf16f32_kern::dot_product(
        const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    if (bfloat16_)
        vdpbf16ps(dst, src1, src2);
    else
        bf16_emu_->vdpbf16ps(make_zmm(dst), make_zmm(src1), make_zmm(src2));
}

// Vector load for 16-bit values.
void jit_avx512_core_gemv_bf16bf16f32_kern::v_load(
        const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems) {
    if (nelems >= 32)
        vmovdqu16(dst, src);
    else if (nelems > 16)
        vmovdqu16(dst | k1 | T_z, src);
    else if (nelems == 16)
        vmovdqu16(make_ymm(dst), src);
    else if (nelems > 8)
        vmovdqu16(make_ymm(dst) | k1 | T_z, src);
    else if (nelems == 8)
        vmovdqu16(make_xmm(dst), src);
    else if (nelems > 4)
        vmovdqu16(make_xmm(dst) | k1 | T_z, src);
    else if (nelems == 4)
        vmovsd(make_xmm(dst), src);
    else if (nelems > 2)
        vmovdqu16(make_xmm(dst) | k1 | T_z, src);
    else if (nelems == 2)
        vmovss(make_xmm(dst), src);
    else
        vmovdqu16(make_xmm(dst) | k1 | T_z, src);
}

void jit_avx512_core_gemv_bf16bf16f32_kern::y_load(
        const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems) {
    if (nelems >= 16)
        vmovups(dst, src);
    else if (nelems > 8)
        vmovups(dst | k1 | T_z, src);
    else if (nelems == 8)
        vmovups(make_ymm(dst), src);
    else if (nelems > 4)
        vmovups(make_ymm(dst) | k1 | T_z, src);
    else if (nelems == 4)
        vmovups(make_xmm(dst), src);
    else if (nelems > 2)
        vmovups(make_xmm(dst) | k1 | T_z, src);
    else if (nelems == 2)
        vmovsd(make_xmm(dst), src);
    else
        vmovss(make_xmm(dst), src);
}

void jit_avx512_core_gemv_bf16bf16f32_kern::y_store(
        const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems) {
    if (nelems >= 16)
        vmovups(dst, src);
    else if (nelems > 8)
        vmovups(dst, src | k1);
    else if (nelems == 8)
        vmovups(dst, make_ymm(src));
    else if (nelems > 4)
        vmovups(dst, make_ymm(src) | k1);
    else if (nelems == 4)
        vmovups(dst, make_xmm(src));
    else if (nelems > 2)
        vmovups(dst, make_xmm(src) | k1);
    else if (nelems == 2)
        vmovlps(dst, make_xmm(src));
    else
        vmovss(dst, make_xmm(src));
}

void jit_avx512_core_gemv_bf16bf16f32_kern::kernel_loop_n(
        int unroll_m, int unroll_n, bool fetch, bool last) {
    int zmm_vecs = utils::div_up(unroll_m, 32);

    for (int un = 0, i = 0; un < unroll_n; un += 2, i++) {
        int mult = un % 4;
        auto A = un < 4 ? A1_ : A2_;
        for (int j = 0; j < zmm_vecs; j++) {
            // Load A.
            if (fetch)
                prefetch_a(ptr[A + LDA_ * mult
                        + size_a_ * (prefetch_size_a_ + j * 32 - offset_a_)]);
            v_load(a_[0], ptr[A + LDA_ * mult + size_a_ * (j * 32 - offset_a_)],
                    unroll_m);

            if (un + 1 == unroll_n) {
                if (unroll_m <= 16)
                    vpxorq(make_ymm(a_[1]), make_ymm(a_[1]), make_ymm(a_[1]));
                else
                    vpxorq(a_[1], a_[1], a_[1]);
            } else {
                decltype(LDA_ * (mult + 1)) lda_mult
                        = (mult + 1) == 3 ? LDA3_ : LDA_ * (mult + 1);
                if (fetch)
                    prefetch_a(ptr[A + lda_mult
                            + size_a_
                                    * (prefetch_size_a_ + j * 32 - offset_a_)]);
                v_load(a_[1],
                        ptr[A + lda_mult + size_a_ * (j * 32 - offset_a_)],
                        unroll_m);
            }

            vpunpcklwd(a_pack_[0], a_[0], a_[1]);
            vpunpckhwd(a_pack_[1], a_[0], a_[1]);

            dot_product(acc_[2 * j + 0], a_pack_[0], x_pack_[i]);
            dot_product(acc_[2 * j + 1], a_pack_[1], x_pack_[i]);
        }
    }

    int ymm_vecs = utils::div_up(unroll_m, 16);
    uint8 imm[] = {0x44, 0xee};
    for (int j = 0, i = 0; j < ymm_vecs; i += j % 2, j++) {
        vshuff32x4(scratch_[j], acc_[2 * i], acc_[2 * i + 1], imm[j % 2]);
    }

    for (int j = 0; j < ymm_vecs; j++) {
        vshuff32x4(acc_[j], scratch_[j], scratch_[j], 0xd8);
    }

    // Update y.
    for (int j = 0; j < ymm_vecs; j++) {
        if (fetch)
            prefetch_y(ptr[Y1_
                    + size_y_ * (prefetch_size_y_ + j * 16 - offset_y_)]);
        y_load(y_[j], ptr[Y1_ + size_y_ * (j * 16 - offset_y_)], unroll_m);
    }

    for (int j = 0; j < ymm_vecs; j++) {
        vfmadd231ps(y_[j], acc_[j], alpha_);
        if (unroll_m >= 32) vpxorq(acc_[j], acc_[j], acc_[j]);
    }

    if (unroll_m == 16) {
        vpxorq(acc_[0], acc_[0], acc_[0]);
        vpxorq(acc_[1], acc_[1], acc_[1]);
    }

    // Store y.
    for (int j = 0; j < ymm_vecs; j++) {
        y_store(ptr[Y1_ + size_y_ * (j * 16 - offset_y_)], y_[j], unroll_m);
    }

    if (!last) {
        add(A1_, unroll_m * size_a_);
        if (unroll_n > 4) add(A2_, unroll_m * size_a_);
        add(Y1_, unroll_m * size_y_);
    }
}

// Inner loop for A non-transposed.
void jit_avx512_core_gemv_bf16bf16f32_kern::innerloop_n(int unroll_n) {
    mov(A1_, A_);
    if (unroll_n > 4) {
        lea(A2_, ptr[A1_ + LDA_ * 4]);
        lea(A_, ptr[A_ + LDA_ * 8]);
    }
    mov(Y1_, Y_);

    // Load x.
    prefetch_x(ptr[X_ + size_x_ * (prefetch_size_x_ - offset_x_)]);
    for (int un = 1, i = 0; un <= unroll_n; un += 2, i++) {
        auto x0 = make_xmm(x_[0]);
        auto x1 = make_xmm(x_[1]);
        vpbroadcastw(x0, ptr[X_ + size_x_ * (0 - offset_x_)]);
        add(X_, INCX_);

        if (un == unroll_n) {
            vpxorq(x1, x1, x1);
        } else {
            vpbroadcastw(x1, ptr[X_ + size_x_ * (0 - offset_x_)]);
            add(X_, INCX_);
        }

        vpunpcklwd(x0, x0, x1);
        vpbroadcastd(x_pack_[i], x0);
    }

    for (int i = 0; i < (UNROLL_N_ >> 1); i++)
        vpxorq(acc_[i], acc_[i], acc_[i]);

    Label label_m_tail_begin;
    mov(I_, M_);
    sar(I_, 6);
    jle(label_m_tail_begin, T_NEAR);

    Label label_m_loop;
    L_aligned(label_m_loop);
    {
        kernel_loop_n(64, unroll_n, true, false);

        dec(I_);
        jg(label_m_loop, T_NEAR);
    }

    Label label_m_tail_2;
    L_aligned(label_m_tail_begin);
    mov(I_, M_);
    test(I_, 32);
    jle(label_m_tail_2, T_NEAR);

    kernel_loop_n(32, unroll_n, false, false);

    Label label_m_tail_last;
    L_aligned(label_m_tail_2);
    mov(I_, M_);
    test(I_, 16);
    jle(label_m_tail_last, T_NEAR);

    kernel_loop_n(16, unroll_n, false, false);

    Label label_m_tail_end;
    L_aligned(label_m_tail_last);
    mov(I_, M_);
    and_(I_, 15);
    jle(label_m_tail_end, T_NEAR);

    // Prepare mask.
    mov(rbx, rcx);
    mov(rcx, rax);
    mov(rax, -1);
    shl(rax, cl);
    kmovq(k1, rax);
    knotq(k1, k1);
    mov(rcx, rbx);

    kernel_loop_n(15, unroll_n, false, true);

    L_aligned(label_m_tail_end);
}

void jit_avx512_core_gemv_bf16bf16f32_kern::kernel_loop_t(
        int unroll_m, int unroll_n, bool fetch, bool last) {

    // Load x.
    if (fetch) prefetch_x(ptr[X1_ + size_x_ * (prefetch_size_x_ - offset_x_)]);
    v_load(x_[0], ptr[X1_ + size_x_ * (0 - offset_x_)], unroll_m);

    for (int j = 0; j < unroll_n; j++) {
        int mult = j % 4;
        auto A = j < 4 ? A1_ : A2_;
        decltype(LDA_ * mult) lda_mult = mult == 3 ? LDA3_ : LDA_ * mult;

        // Load A.
        if (fetch)
            prefetch_a(ptr[A + lda_mult
                    + size_a_ * (prefetch_size_a_ - offset_a_)]);
        v_load(a_[j], ptr[A + lda_mult + size_a_ * (0 - offset_a_)], unroll_m);

        dot_product(acc_[j], x_[0], a_[j]);
    }

    if (!last) {
        add(A1_, unroll_m * size_a_);
        if (unroll_n > 4) add(A2_, unroll_m * size_a_);
        add(X1_, unroll_m * size_x_);
    }
}

// Inner loop for A transposed.
void jit_avx512_core_gemv_bf16bf16f32_kern::innerloop_t(int unroll_n) {
    mov(A1_, A_);
    if (unroll_n > 4) {
        lea(A2_, ptr[A1_ + LDA_ * 4]);
        lea(A_, ptr[A_ + LDA_ * 8]);
    }
    mov(X1_, X_);
    prefetch_y(ptr[Y_ + size_y_ * (prefetch_size_y_ - offset_y_)]);

    for (int j = 0; j < UNROLL_N_; j++)
        vpxorq(acc_[j], acc_[j], acc_[j]);

    Label label_m_tail;
    mov(I_, M_);
    sar(I_, 5);
    jle(label_m_tail, T_NEAR);

    Label label_m_loop;
    L_aligned(label_m_loop);
    {
        kernel_loop_t(32, unroll_n, true, false);

        dec(I_);
        jg(label_m_loop, T_NEAR);
    }

    Label label_m_tail_end;
    L_aligned(label_m_tail);
    mov(I_, M_);
    and_(I_, 31);
    je(label_m_tail_end, T_NEAR);

    // Prepare mask.
    mov(rbx, rcx);
    mov(rcx, rax);
    mov(rax, -1);
    shl(rax, cl);
    kmovq(k1, rax);
    knotq(k1, k1);
    mov(rcx, rbx);

    kernel_loop_t(31, unroll_n, false, true);

    L_aligned(label_m_tail_end);

    // Reduction step.
    for (int j = 0; j < utils::rnd_up(unroll_n, 4); j += 2) {
        auto s0 = make_ymm(scratch_[(j + 0) % 4]);
        auto s1 = make_ymm(scratch_[(j + 1) % 4]);
        vextractf64x4(s0, acc_[j + 0], 0x01);
        vextractf64x4(s1, acc_[j + 1], 0x01);
        vaddps(make_ymm(acc_[j + 0]), s0, make_ymm(acc_[j + 0]));
        vaddps(make_ymm(acc_[j + 1]), s1, make_ymm(acc_[j + 1]));
    }

    int nloops = unroll_n > 2 ? 4 : 2;
    for (int j = 0; j < nloops; j++) {
        auto s0 = make_ymm(scratch_[(2 * j + 0) % 4]);
        auto s1 = make_ymm(scratch_[(2 * j + 1) % 4]);
        auto acc0 = make_ymm(acc_[j + 0]);
        auto acc4 = make_ymm(acc_[j + 4]);
        vperm2f128(s0, acc0, acc4, 0x20);
        vperm2f128(s1, acc0, acc4, 0x31);
        vhaddps(acc0, s0, s1);
    }

    vhaddps(make_ymm(acc_[0]), make_ymm(acc_[0]), make_ymm(acc_[1]));
    vhaddps(make_ymm(acc_[2]), make_ymm(acc_[2]), make_ymm(acc_[3]));
    vhaddps(make_ymm(acc_[0]), make_ymm(acc_[0]), make_ymm(acc_[2]));

    Label label_incy_isnt_1;
    cmp(INCY_, size_y_);
    jne(label_incy_isnt_1, T_NEAR);

    // Update y for incy == 1.
    auto is_pow_2 = [](unsigned int n) { return n != 0 && !(n & (n - 1)); };

    if (!is_pow_2(unroll_n)) {
        mov(rax, (1 << unroll_n) - 1);
        kmovq(k1, rax);
    }

    auto y_mem = ptr[Y_ + size_y_ * (0 - offset_y_)];
    y_load(y_[0], y_mem, unroll_n);
    vfmadd231ps(make_ymm(y_[0]), make_ymm(acc_[0]), make_ymm(alpha_));
    y_store(y_mem, y_[0], unroll_n);

    Label label_innerloop_end;
    add(Y_, unroll_n * size_y_);
    jmp(label_innerloop_end, T_NEAR);

    // Update y for incy != 1.
    L_aligned(label_incy_isnt_1);

    // Shuffle ps to ss in 4 registers.
    auto shufps2ss = [this](Xbyak::Xmm *v) {
        uint8 imm[] = {0x00, 0x55, 0xaa, 0xff};
        auto v0 = make_ymm(v[0]);
        for (int j = 1; j <= 3; j++) {
            auto vj = make_ymm(v[j]);
            vshufps(vj, v0, v0, imm[j]);
        }
    };

    vmulps(make_ymm(acc_[0]), make_ymm(acc_[0]), make_ymm(alpha_));
    shufps2ss(&acc_[0]);
    vperm2f128(make_ymm(acc_[4]), make_ymm(acc_[0]), make_ymm(acc_[0]), 0x11);
    shufps2ss(&acc_[4]);

    for (int j = 0; j < unroll_n; j++) {
        auto y = make_xmm(y_[0]);
        vmovss(y, y_mem);
        vaddss(y, make_xmm(acc_[j]), y);
        vmovss(y_mem, y);
        add(Y_, INCY_);
    }

    L_aligned(label_innerloop_end);
}

// Outer loop.
void jit_avx512_core_gemv_bf16bf16f32_kern::outerloop(int unroll_y,
        Label *&cur_outerloop_label, Label *&outerloop_end_label) {

    bool is_tail = unroll_y < UNROLL_N_;

    if (is_tail) {
        L_aligned(*cur_outerloop_label);
        cur_outerloop_label++;
    }
    cmp(N_, unroll_y);
    jl(*cur_outerloop_label, T_NEAR); // Jump to next outerloop label.

    if (!is_tail) {
        Label label_n_loop;
        L_aligned(label_n_loop);
        {
            if (trans_)
                innerloop_t(unroll_y);
            else
                innerloop_n(unroll_y);
            sub(N_, unroll_y);
            cmp(N_, unroll_y);
            jge(label_n_loop, T_NEAR);
        }
    } else {
        if (trans_)
            innerloop_t(unroll_y);
        else
            innerloop_n(unroll_y);
        if (unroll_y > 1) jmp(*outerloop_end_label, T_NEAR);
    }
}

void jit_avx512_core_gemv_bf16bf16f32_kern::generate() {
    // Prologue
    preamble();

    if (is_windows) {
        mov(LDA_, arg_lda_);
        mov(X_, arg_x_);
    }
    if (!trans_) mov(INCX_, arg_incx_); // incy is assumed 1.
    mov(Y_, arg_y_);
    if (trans_) mov(INCY_, arg_incy_); // incx is assumed 1.

    vbroadcastss(alpha_, qword[ALPHA_]);

    mov(M_, qword[M_]);
    mov(N_, qword[N_]);
    mov(LDA_, qword[LDA_]);
    if (!trans_)
        mov(INCX_, qword[INCX_]);
    else
        mov(INCY_, qword[INCY_]);

    sub(A_, -offset_a_ * size_a_);
    sub(X_, -offset_x_ * size_x_);
    sub(Y_, -offset_y_ * size_y_);

    lea(LDA_, ptr[LDA_ * size_a_]);
    if (!trans_)
        lea(INCX_, ptr[INCX_ * size_x_]);
    else
        lea(INCY_, ptr[INCY_ * size_y_]);

    lea(LDA3_, ptr[LDA_ + LDA_ * 2]);

    Label outerloop_labels[UNROLL_N_];
    Label *cur_outerloop_label = &outerloop_labels[0];
    Label *outerloop_end_label = &outerloop_labels[UNROLL_N_ - 1];

    // Main n loop.
    for (int un = UNROLL_N_; un > 0; un--)
        outerloop(un, cur_outerloop_label, outerloop_end_label);

    L_aligned(*outerloop_end_label);

    // Epilogue.
    postamble();
}

// Function signature: gemv(*m, *n, *alpha, *a, *lda, *x, *incx, *y, *incy)
jit_avx512_core_gemv_bf16bf16f32_kern::jit_avx512_core_gemv_bf16bf16f32_kern(
        bool trans)
    : jit_generator(nullptr, 20000)
    , arg_lda_(0)
    , arg_x_(0)
    , arg_incx_(0)
    , arg_y_(0)
    , arg_incy_(0) {

    trans_ = trans;
    bfloat16_ = mayiuse(avx512_core_bf16);
    assert(mayiuse(avx512_core));

    // Assign integer registers
    M_ = is_windows ? rcx : rdi;
    N_ = is_windows ? rdx : rsi;
    ALPHA_ = is_windows ? r8 : rdx;
    A_ = is_windows ? r9 : rcx;
    LDA_ = is_windows ? r10 : r8;
    X_ = is_windows ? r11 : r9;
    INCX_ = is_windows ? r12 : r10;
    Y_ = is_windows ? rdi : r11;
    INCY_ = is_windows ? rsi : r12;

    I_ = rax;
    A1_ = r13;
    A2_ = is_windows ? r8 : rdx;

    if (!trans_)
        Y1_ = r14;
    else
        X1_ = r14;

    LDA3_ = r15;

    // Assign vector registers
    alpha_ = zmm15;

    if (!trans_) {
        a_[0] = zmm4;
        a_[1] = zmm5;

        a_pack_[0] = zmm6;
        a_pack_[1] = zmm7;

        x_[0] = zmm8;
        x_[1] = zmm9;

        for (int i = 0; i < (UNROLL_M_ >> 4); i++)
            y_[i] = Zmm(8 + i);

        for (int i = 0; i < (UNROLL_N_ >> 1); i++) {
            x_pack_[i] = Zmm(i);
            acc_[i] = Zmm(16 + i);
        }
    } else {
        x_[0] = zmm8;
        y_[0] = zmm8;

        for (int i = 0; i < UNROLL_N_; i++) {
            acc_[i] = Zmm(i);
            a_[i] = Zmm(16 + i);
        }
    }

    for (int i = 0; i < 4; i++)
        scratch_[i] = Zmm(8 + i);

    // Assign stack variables.
    auto args_offset = get_size_of_abi_save_regs() + 8 + (is_windows ? 48 : 0);

    arg_lda_ = ptr[rsp + (args_offset - 16)];
    arg_x_ = ptr[rsp + (args_offset - 8)];
    arg_incx_ = ptr[rsp + (args_offset + 0)]; // Assumed 1 for A transpose.
    arg_y_ = ptr[rsp + (args_offset + 8)];
    arg_incy_ = ptr[rsp + (args_offset + 16)]; // Assumed 1 for A non-transpose.

    bf16_emu_ = nullptr;
    if (!bfloat16_) {
        // Those register are only used if we use bf16 convert instruction
        // emulation.
        gpr_ = rbp;
        one_ = zmm24;
        even_ = zmm25;
        selector_ = zmm26;

        zmm_tmp0_ = zmm12;
        zmm_tmp1_ = zmm13;
        bf16_emu_ = new bf16_emulation_t(
                this, one_, even_, selector_, gpr_, zmm_tmp0_, zmm_tmp1_);
    }
}

jit_avx512_core_gemv_bf16bf16f32_kern::
        ~jit_avx512_core_gemv_bf16bf16f32_kern() {
    delete bf16_emu_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
