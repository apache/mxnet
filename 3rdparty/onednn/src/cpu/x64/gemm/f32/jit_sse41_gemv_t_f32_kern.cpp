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

#include "cpu/x64/gemm/f32/jit_sse41_gemv_t_f32_kern.hpp"

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

// Load vector register data for x, y or A.
void jit_sse41_gemv_t_f32_kern::v_load(
        const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems) {
    switch (nelems) {
        case 1: movss(dst, src); break;
        case 2: movsd(dst, src); break;
        default:
            assert(nelems >= 4);
            movups(dst, src);
            break;
    }
}

// Store vector register data for x, y or A.
void jit_sse41_gemv_t_f32_kern::v_store(
        const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems) {
    switch (nelems) {
        case 1: movss(dst, src); break;
        case 2: movsd(dst, src); break;
        default:
            assert(nelems >= 4);
            movups(dst, src);
            break;
    }
}

// Perform Hadamard product of 2 vectors and accumulate.
void jit_sse41_gemv_t_f32_kern::dot_product(
        const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    mulps(src2, src1);
    addps(dst, src2);
}

// Inner loop.
void jit_sse41_gemv_t_f32_kern::innerloop(int unroll_m, int unroll_n) {
    if ((unroll_m > M_UNROLL_) || (unroll_n > N_UNROLL_) || (unroll_m < 0)
            || (unroll_n < 0))
        return;

    int um_vecs = (unroll_m + 3) >> 2;

    // Load x.
    for (int i = 0; i < um_vecs; i++) {
        auto x_mem = ptr[XO_ + size_ * (4 * i - offset_x_)];
        v_load(x_regs_[i], x_mem, unroll_m);
    }
    add(XO_, size_ * unroll_m);

    Reg64 LDA3 = rax;
    lea(LDA3, ptr[LDA_ + LDA_ * 2]);

    // Load A
    for (int j = 0; j < unroll_n; j++) {
        for (int i = 0; i < um_vecs; i++) {
            Xmm a = a_regs_[i][j];

            decltype(LDA_ * j) lda_mult = (j == 3) ? LDA3 : LDA_ * j;

            auto a_mem = ptr[AO_ + lda_mult + size_ * (4 * i - offset_a_)];
            v_load(a, a_mem, unroll_m);
        }
    }

    lea(AO_, ptr[AO_ + size_ * unroll_m]);

    for (int j = 0; j < unroll_n; j++) {
        Xmm acc = acc_[j];

        for (int i = 0; i < um_vecs; i++) {
            dot_product(acc, x_regs_[i], a_regs_[i][j]);
        }
    }
}

// Outer loop.
void jit_sse41_gemv_t_f32_kern::outerloop(
        int unroll_x, int unroll_y, Label *&cur_outerloop_label) {
    if ((unroll_x > M_UNROLL_) || (unroll_y > N_UNROLL_) || (unroll_y < 0)
            || (unroll_y < 0))
        return;

    Label label_m_loop, label_n_loop, label_m_remainder_loops[4];

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_y >= N_UNROLL_) {
        mov(I_, N_);
        cmp(I_, unroll_y);
        jl(*cur_outerloop_label, T_NEAR); // Jump to next outerloop label.
    } else {
        test(I_, unroll_y);
        jle(*cur_outerloop_label, T_NEAR);
    }

    L_aligned(label_n_loop);
    {

        mov(YO_, Y_);
        lea(Y_, ptr[YO_ + INCY_ * unroll_y]);

        mov(AO_, A_);
        lea(A_, ptr[AO_ + LDA_ * unroll_y]);

        mov(XO_, X_);

        for (int i = 0; i < unroll_y; i++) {
            auto acc = acc_[i];
            xorps(acc, acc);
        }

        mov(J_, M_);
        cmp(J_, unroll_x);
        jl(label_m_remainder_loops[0], T_NEAR);

        L_aligned(label_m_loop);
        {
            innerloop(unroll_x, unroll_y);
            sub(J_, unroll_x);
            cmp(J_, unroll_x);
            jge(label_m_loop, T_NEAR);
        }

        align(16);

        // Update y.
        for (int j = 0; j < unroll_y; j++) {
            Xmm acc = acc_[j];

            haddps(acc, acc);
            haddps(acc, acc);
        }
        for (int j = 0; j < unroll_y; j++) {
            // TODO Handle negative increments
            Xmm y = y_regs_[j];
            Xmm acc = acc_[j];

            imul(YO2_, INCY_, j);
            lea(YO2_, ptr[YO_ + YO2_]);
            auto y_mem = ptr[YO2_];

            v_load(y, y_mem, 1);

            mulps(acc, alpha_);
            addps(y, acc);

            v_store(y_mem, y, 1);
        }

        int label_idx = 0;
        for (int ux = 4; ux > 0; ux >>= 1) {
            L(label_m_remainder_loops[label_idx++]);
            if (unroll_x > ux) {
                test(J_, ux);
                jle(label_m_remainder_loops[label_idx], T_NEAR);

                for (int i = 0; i < unroll_y; i++) {
                    auto acc = acc_[i];
                    xorps(acc, acc);
                }

                innerloop(ux, unroll_y);

                align(16);

                // Update y.
                for (int j = 0; j < unroll_y; j++) {
                    Xmm acc = acc_[j];

                    haddps(acc, acc);
                    haddps(acc, acc);
                }
                for (int j = 0; j < unroll_y; j++) {
                    // TODO Handle negative increments
                    Xmm y = y_regs_[j];
                    Xmm acc = acc_[j];

                    imul(YO2_, INCY_, j);
                    lea(YO2_, ptr[YO_ + YO2_]);
                    auto y_mem = ptr[YO2_];

                    v_load(y, y_mem, 1);

                    mulps(acc, alpha_);
                    addps(y, acc);

                    v_store(y_mem, y, 1);
                }
            }
        }
        L(label_m_remainder_loops[label_idx]);

        if (unroll_y >= N_UNROLL_) {
            sub(I_, unroll_y);
            cmp(I_, unroll_y);
            jge(label_n_loop);
        }
    }

    align(16);
}

void jit_sse41_gemv_t_f32_kern::generate() {
    // Prologue
    preamble();

    movss(alpha_, qword[ALPHA_]);

    if (is_windows) {
        mov(LDA_, arg_lda_);
        mov(X_, arg_x_);
    }

    mov(Y_, arg_y_);
    mov(INCY_, arg_incy_);

    sub(A_, -offset_a_ * size_);
    sub(X_, -offset_x_ * size_);

    mov(M_, qword[M_]);
    mov(N_, qword[N_]);
    mov(LDA_, qword[LDA_]);
    mov(INCY_, qword[INCY_]);

    lea(LDA_, ptr[LDA_ * size_]);
    lea(INCY_, ptr[INCY_ * size_]);

    Label outerloop_labels[4];
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main n loop.
    outerloop(M_UNROLL_, N_UNROLL_, cur_outerloop_label);

    // n remainder loops.
    for (int un = 2; un > 0; un >>= 1)
        if (N_UNROLL_ > un) outerloop(M_UNROLL_, un, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    postamble();
}

// Function signature: gemv(*m, *n, *alpha, *a, *lda, *x, *incx, *y, *incy)
jit_sse41_gemv_t_f32_kern::jit_sse41_gemv_t_f32_kern()
    : jit_generator(nullptr, 100000)
    , arg_lda_(0)
    , arg_x_(0)
    , arg_incx_(0)
    , arg_y_(0)
    , arg_incy_(0) {

    // Assign integer registers
    M_ = abi_param1;
    N_ = abi_param2;
    ALPHA_ = abi_param3;
    A_ = abi_param4;
    LDA_ = is_windows ? rdi : r8;
    X_ = is_windows ? rsi : r9;
    INCY_ = r10;
    Y_ = r11;

    J_ = r12;
    I_ = r13;

    AO_ = r14;
    XO_ = r15;

    YO_ = rbx;
    YO2_ = rbp;

    // Assign vector registers
    for (int i = 0; i < (N_UNROLL_); i++)
        y_regs_[i] = Xmm(i);

    int rn = 0;
    for (int i = 0; i < (M_UNROLL_ >> 2); i++)
        for (int j = 0; j < N_UNROLL_; j++)
            a_regs_[i][j] = Xmm(rn++);

    x_regs_[0] = xmm8;
    x_regs_[1] = xmm9;

    alpha_ = xmm10;
    scratch_ = xmm11;

    for (int i = 0; i < (N_UNROLL_); i++)
        acc_[i] = Xmm(12 + i);

    // Assign stack variables.
    auto args_offset = get_size_of_abi_save_regs() + 8 + (is_windows ? 48 : 0);

    arg_lda_ = ptr[rsp + (args_offset - 16)];
    arg_x_ = ptr[rsp + (args_offset - 8)];
    arg_incx_ = ptr[rsp + (args_offset + 0)]; // Assumed 1 for A transpose.
    arg_y_ = ptr[rsp + (args_offset + 8)];
    arg_incy_ = ptr[rsp + (args_offset + 16)]; // Assumed 1 for A non-transpose.
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
