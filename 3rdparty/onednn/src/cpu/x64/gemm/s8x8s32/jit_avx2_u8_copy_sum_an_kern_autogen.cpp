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

jit_avx2_u8_copy_sum_an_kern::jit_avx2_u8_copy_sum_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_u8_copy_sum_an_kern::generate() {

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

        Xbyak::Label l148;
        Xbyak::Label l1dc;
        Xbyak::Label l20;
        Xbyak::Label l230;
        Xbyak::Label l260;
        Xbyak::Label l26c;
        Xbyak::Label l28c;
        Xbyak::Label l3a8;
        Xbyak::Label l440;
        Xbyak::Label l48;
        Xbyak::Label l494;
        Xbyak::Label l4cc;
        Xbyak::Label l4f0;
        Xbyak::Label l4fc;
        Xbyak::Label l514;
        Xbyak::Label l5d0;
        Xbyak::Label l634;
        Xbyak::Label l670;
        Xbyak::Label l694;
        Xbyak::Label l6b4;
        Xbyak::Label l6c0;
        Xbyak::Label l6d8;
        Xbyak::Label l79c;
        Xbyak::Label l808;
        Xbyak::Label l84c;
        Xbyak::Label l874;
        Xbyak::Label l892;
        Xbyak::Label l89c;
        Xbyak::Label l8b4;
        Xbyak::Label l94c;
        Xbyak::Label l9ac;
        Xbyak::Label l9f0;
        Xbyak::Label la14;
        Xbyak::Label la34;

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x10);
        jl(l260, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        add(A, 0x10);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        pxor(xmm10, xmm10);
        pxor(xmm11, xmm11);
        mov(I, M);
        sar(I, 0x2);
        jle(l148, T_NEAR);
        align(4);

        L(l48);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm4, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm4, xmm1);
        movdqa(xmm1, xmm2);
        punpcklbw(xmm2, xmm3);
        punpckhbw(xmm1, xmm3);
        movdqa(xmm3, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm3, xmm2);
        movdqa(xmm2, xmm4);
        punpcklwd(xmm4, xmm1);
        punpckhwd(xmm2, xmm1);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm3);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pmovsxbw(xmm5, xmm2);
        movhlps(xmm6, xmm2);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm2);
        sub(B, -64);
        dec(I);
        jg(l48, T_NEAR);
        align(4);

        L(l148);
        test(M, 0x2);
        jle(l1dc, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm2, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm2, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pmovsxbw(xmm5, xmm2);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movhlps(xmm6, xmm2);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        align(4);

        L(l1dc);
        test(M, 0x1);
        jle(l230, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm8, xmm5);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pshufd(xmm5, xmm0, 0xaa);
        pmovsxbd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pshufd(xmm6, xmm0, 0xff);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l230);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        movdqu(xword[A1 + 0x20], xmm10);
        movdqu(xword[A1 + 0x30], xmm11);
        add(qword[ARG_BIAS], 0x40);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l20, T_NEAR);
        align(4);

        L(l260);
        cmp(N, 0x8);
        jl(l4f0, T_NEAR);
        align(4);

        L(l26c);
        mov(A1, A);
        add(A, 0x8);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        mov(I, M);
        sar(I, 0x3);
        jle(l3a8, T_NEAR);
        align(4);

        L(l28c);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        dec(I);
        jg(l28c, T_NEAR);
        align(4);

        L(l3a8);
        test(M, 0x4);
        jle(l440, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l440);
        test(M, 0x2);
        jle(l494, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l494);
        test(M, 0x1);
        jle(l4cc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        pmovsxbd(xmm5, xmm0);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm8, xmm5);
        paddd(xmm9, xmm6);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l4cc);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        add(qword[ARG_BIAS], 0x20);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l26c, T_NEAR);
        align(4);

        L(l4f0);
        cmp(N, 0x4);
        jl(l6b4, T_NEAR);
        align(4);

        L(l4fc);
        mov(A1, A);
        add(A, 0x4);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x3);
        jle(l5d0, T_NEAR);
        align(4);

        L(l514);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(l514, T_NEAR);
        align(4);

        L(l5d0);
        test(M, 0x4);
        jle(l634, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l634);
        test(M, 0x2);
        jle(l670, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l670);
        test(M, 0x1);
        jle(l694, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l694);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm7);
        add(qword[ARG_BIAS], 0x10);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l4fc, T_NEAR);
        align(4);

        L(l6b4);
        cmp(N, 0x2);
        jl(l892, T_NEAR);
        align(4);

        L(l6c0);
        mov(A1, A);
        add(A, 0x2);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l79c, T_NEAR);
        align(4);

        L(l6d8);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm4, eax, 0x0);
        punpcklbw(xmm1, xmm2);
        punpcklbw(xmm3, xmm4);
        punpcklwd(xmm1, xmm3);
        punpcklqdq(xmm0, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovsxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(l6d8, T_NEAR);
        align(4);

        L(l79c);
        test(M, 0x4);
        jle(l808, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l808);
        test(M, 0x2);
        jle(l84c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l84c);
        test(M, 0x1);
        jle(l874, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l874);
        mov(A1, qword[ARG_BIAS]);
        movq(qword[A1], xmm7);
        add(qword[ARG_BIAS], 0x8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l6c0, T_NEAR);
        align(4);

        L(l892);
        cmp(N, 0x1);
        jl(la34, T_NEAR);
        align(4);

        L(l89c);
        mov(A1, A);
        add(A, 0x1);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l94c, T_NEAR);
        align(4);

        L(l8b4);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x7);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        dec(LDA3);
        jg(l8b4, T_NEAR);
        align(4);

        L(l94c);
        test(M, 0x4);
        jle(l9ac, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l9ac);
        test(M, 0x2);
        jle(l9f0, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(l9f0);
        test(M, 0x1);
        jle(la14, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(la14);
        mov(A1, qword[ARG_BIAS]);
        movd(dword[A1], xmm7);
        add(qword[ARG_BIAS], 0x4);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l89c, T_NEAR);
        align(4);

        L(la34);

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
