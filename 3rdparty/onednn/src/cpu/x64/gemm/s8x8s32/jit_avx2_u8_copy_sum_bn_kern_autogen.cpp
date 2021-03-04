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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_avx2_u8_copy_sum_bn_kern::jit_avx2_u8_copy_sum_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_u8_copy_sum_bn_kern::generate() {

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

#define ARG_BIAS (24 + stacksize + rsp)

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
#define ARG_BIAS 72 + stacksize + rsp

#endif

    inLocalLabel();
    {

        Xbyak::Label l138;
        Xbyak::Label l1cc;
        Xbyak::Label l20;
        Xbyak::Label l230;
        Xbyak::Label l288;
        Xbyak::Label l2d0;
        Xbyak::Label l2f0;
        Xbyak::Label l2fc;
        Xbyak::Label l31c;
        Xbyak::Label l3ac;
        Xbyak::Label l40;
        Xbyak::Label l404;
        Xbyak::Label l448;
        Xbyak::Label l48c;
        Xbyak::Label l4c0;
        Xbyak::Label l4de;
        Xbyak::Label l4e8;
        Xbyak::Label l4fc;
        Xbyak::Label l540;
        Xbyak::Label l57c;
        Xbyak::Label l5b4;
        Xbyak::Label l5e8;
        Xbyak::Label l60c;
        Xbyak::Label l62c;

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
        sub(B, -128);
        lea(LDA3, ptr[LDA + LDA * 2]);
        cmp(N, 0x4);
        jl(l2f0, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(l138, T_NEAR);
        align(4);

        L(l40);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm2, xword[A2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovzxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm1);
        pmovzxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x60], xmm4);
        pmovzxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x50], xmm3);
        sub(B, -64);
        dec(I);
        jg(l40, T_NEAR);
        align(4);

        L(l138);
        test(M, 0x8);
        jle(l1cc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        sub(A1, -8);
        movq(xmm2, qword[A2 - 0x80]);
        movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovzxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l1cc);
        test(M, 0x4);
        jle(l230, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        movd(xmm2, dword[A2 - 0x80]);
        movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l230);
        test(M, 0x2);
        jle(l288, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x3);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l288);
        test(M, 0x1);
        jle(l2d0, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l2d0);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm7);
        add(qword[ARG_BIAS], 0x10);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l20, T_NEAR);
        align(4);

        L(l2f0);
        cmp(N, 0x2);
        jl(l4de, T_NEAR);
        align(4);

        L(l2fc);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(l3ac, T_NEAR);
        align(4);

        L(l31c);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        movdqa(xmm2, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm2, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovzxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pshufd(xmm6, xmm2, 0xd8);
        pmovzxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        dec(I);
        jg(l31c, T_NEAR);
        align(4);

        L(l3ac);
        test(M, 0x8);
        jle(l404, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovzxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l404);
        test(M, 0x4);
        jle(l448, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l448);
        test(M, 0x2);
        jle(l48c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l48c);
        test(M, 0x1);
        jle(l4c0, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        align(4);

        L(l4c0);
        mov(A1, qword[ARG_BIAS]);
        movq(qword[A1], xmm7);
        add(qword[ARG_BIAS], 0x8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l2fc, T_NEAR);
        align(4);

        L(l4de);
        cmp(N, 0x1);
        jl(l62c, T_NEAR);
        align(4);

        L(l4e8);
        mov(A1, A);
        add(A, LDA);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(l540, T_NEAR);
        align(4);

        L(l4fc);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(l4fc, T_NEAR);
        align(4);

        L(l540);
        test(M, 0x8);
        jle(l57c, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l57c);
        test(M, 0x4);
        jle(l5b4, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l5b4);
        test(M, 0x2);
        jle(l5e8, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(l5e8);
        test(M, 0x1);
        jle(l60c, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(l60c);
        mov(A1, qword[ARG_BIAS]);
        movd(dword[A1], xmm7);
        add(qword[ARG_BIAS], 0x4);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l4e8, T_NEAR);
        align(4);

        L(l62c);
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
#undef ARG_BIAS
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
