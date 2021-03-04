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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/bf16/common_s16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_avx512_core_s16_24x8_copy_bt_kern::jit_avx512_core_s16_24x8_copy_bt_kern()
    : jit_generator(nullptr, S16_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_s16_24x8_copy_bt_kern::generate() {

#ifndef _WIN32
#define M rdi
#define N rsi
#define A rdx
#define LDA rcx
#define ALPHA r8
#define B r9

#define I rax
#define A1 r10
#define A2 r8
#define LDA3 r11

#else

#define M rcx
#define N rdx
#define A r8
#define LDA r9
#define ALPHA rax
#define B rdi

#define I rax
#define A1 rsi
#define A2 r10
#define LDA3 r11

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp

#endif

    inLocalLabel();
    {

        Xbyak::Label l13c;
        Xbyak::Label l170;
        Xbyak::Label l18c;
        Xbyak::Label l19c;
        Xbyak::Label l1a8;
        Xbyak::Label l1b8;
        Xbyak::Label l234;
        Xbyak::Label l24;
        Xbyak::Label l27c;
        Xbyak::Label l2a8;
        Xbyak::Label l2c4;
        Xbyak::Label l2d4;
        Xbyak::Label l2e0;
        Xbyak::Label l2f0;
        Xbyak::Label l368;
        Xbyak::Label l38;
        Xbyak::Label l3ac;
        Xbyak::Label l3d8;
        Xbyak::Label l3f4;
        Xbyak::Label l402;
        Xbyak::Label l40c;
        Xbyak::Label l41c;
        Xbyak::Label l494;
        Xbyak::Label l4dc;
        Xbyak::Label l50c;
        Xbyak::Label l524;
        Xbyak::Label l534;
        Xbyak::Label le0;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        shl(LDA, 1);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x8);
        jl(l19c, T_NEAR);
        align(4);

        L(l24);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(le0, T_NEAR);
        align(4);

        L(l38);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm4, xmm0, xmm1);
        vpunpckhwd(xmm5, xmm0, xmm1);
        vperm2f128(ymm0, ymm4, ymm5, 0x20);
        vpunpcklwd(xmm4, xmm2, xmm3);
        vpunpckhwd(xmm5, xmm2, xmm3);
        vperm2f128(ymm2, ymm4, ymm5, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm4, xmm0, xmm1);
        vpunpckhwd(xmm5, xmm0, xmm1);
        vperm2f128(ymm0, ymm4, ymm5, 0x20);
        vpunpcklwd(xmm4, xmm2, xmm3);
        vpunpckhwd(xmm5, xmm2, xmm3);
        vperm2f128(ymm2, ymm4, ymm5, 0x20);
        vmovdqu(yword[B - 0x40], ymm0);
        vmovdqu(yword[B - 0x20], ymm2);
        sub(B, -128);
        dec(I);
        jg(l38, T_NEAR);
        align(4);

        L(le0);
        test(M, 0x4);
        jle(l13c, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm4, xmm0, xmm1);
        vpunpckhwd(xmm5, xmm0, xmm1);
        vperm2f128(ymm0, ymm4, ymm5, 0x20);
        vpunpcklwd(xmm4, xmm2, xmm3);
        vpunpckhwd(xmm5, xmm2, xmm3);
        vperm2f128(ymm2, ymm4, ymm5, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm2);
        sub(B, -64);
        align(4);

        L(l13c);
        test(M, 0x2);
        jle(l170, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        vmovdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm0, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(l170);
        test(M, 0x1);
        jle(l18c, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l18c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l24, T_NEAR);
        align(4);

        L(l19c);
        cmp(N, 0x4);
        jl(l2d4, T_NEAR);
        align(4);

        L(l1a8);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l234, T_NEAR);
        align(4);

        L(l1b8);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vmovdqu(yword[B - 0x60], ymm0);
        sub(B, -64);
        dec(I);
        jg(l1b8, T_NEAR);
        align(4);

        L(l234);
        test(M, 0x4);
        jle(l27c, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vmovdqu(xword[B - 0x80], xmm0);
        vmovdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        align(4);

        L(l27c);
        test(M, 0x2);
        jle(l2a8, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l2a8);
        test(M, 0x1);
        jle(l2c4, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l2c4);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l1a8, T_NEAR);
        align(4);

        L(l2d4);
        cmp(N, 0x2);
        jl(l402, T_NEAR);
        align(4);

        L(l2e0);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l368, T_NEAR);
        align(4);

        L(l2f0);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovdqu(xword[B - 0x80], xmm0);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(l2f0, T_NEAR);
        align(4);

        L(l368);
        test(M, 0x4);
        jle(l3ac, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vpunpcklwd(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l3ac);
        test(M, 0x2);
        jle(l3d8, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l3d8);
        test(M, 0x1);
        jle(l3f4, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l3f4);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l2e0, T_NEAR);
        align(4);

        L(l402);
        cmp(N, 0x1);
        jl(l534, T_NEAR);
        align(4);

        L(l40c);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l494, T_NEAR);
        align(4);

        L(l41c);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(l41c, T_NEAR);
        align(4);

        L(l494);
        test(M, 0x4);
        jle(l4dc, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l4dc);
        test(M, 0x2);
        jle(l50c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l50c);
        test(M, 0x1);
        jle(l524, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l524);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l40c, T_NEAR);
        align(4);

        L(l534);
        vzeroupper();
        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
#ifdef _WIN32
#undef ARG_ALPHA
#undef ARG_B
#endif
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
