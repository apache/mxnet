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

jit_avx2_u8_copy_at_kern::jit_avx2_u8_copy_at_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_u8_copy_at_kern::generate() {

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

        Xbyak::Label l1ec;
        Xbyak::Label l20;
        Xbyak::Label l2fc;
        Xbyak::Label l3c;
        Xbyak::Label l3d8;
        Xbyak::Label l4b4;
        Xbyak::Label l584;
        Xbyak::Label l594;
        Xbyak::Label l5a0;
        Xbyak::Label l5bc;
        Xbyak::Label l698;
        Xbyak::Label l728;
        Xbyak::Label l798;
        Xbyak::Label l80c;
        Xbyak::Label l878;
        Xbyak::Label l888;
        Xbyak::Label l894;
        Xbyak::Label l8ac;
        Xbyak::Label l920;
        Xbyak::Label l970;
        Xbyak::Label l9b4;
        Xbyak::Label l9fc;
        Xbyak::Label la3c;
        Xbyak::Label la4c;
        Xbyak::Label la58;
        Xbyak::Label la70;
        Xbyak::Label laa8;
        Xbyak::Label lad4;
        Xbyak::Label lb00;
        Xbyak::Label lb30;
        Xbyak::Label lb50;
        Xbyak::Label lb5e;
        Xbyak::Label lb68;
        Xbyak::Label lb78;
        Xbyak::Label lb94;
        Xbyak::Label lbb4;
        Xbyak::Label lbd4;
        Xbyak::Label lbf0;
        Xbyak::Label lc08;
        Xbyak::Label lc18;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
        sub(B, -128);
        lea(LDA3, ptr[LDA + LDA * 2]);
        cmp(N, 0x10);
        jl(l594, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        mov(I, LDA);
        shl(I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l1ec, T_NEAR);
        align(4);

        L(l3c);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x40], xmm1);
        movdqu(xword[B], xmm4);
        movdqu(xword[B + 0x40], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
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
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x30], xmm1);
        movdqu(xword[B + 0x10], xmm4);
        movdqu(xword[B + 0x50], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
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
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movdqu(xword[B + 0x20], xmm4);
        movdqu(xword[B + 0x60], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
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
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        movdqu(xword[B + 0x30], xmm4);
        movdqu(xword[B + 0x70], xmm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(l3c, T_NEAR);
        align(4);

        L(l1ec);
        test(M, 0x8);
        jle(l2fc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x40], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x30], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(l2fc);
        test(M, 0x4);
        jle(l3d8, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(l3d8);
        test(M, 0x2);
        jle(l4b4, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x70], xmm0);
        sub(A1, -2);
        sub(B, -32);
        align(4);

        L(l4b4);
        test(M, 0x1);
        jle(l584, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0x7);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x8);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x9);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xa);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0xb);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0xc);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xd);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xe);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xf);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l584);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l20, T_NEAR);
        align(4);

        L(l594);
        cmp(N, 0x8);
        jl(l888, T_NEAR);
        align(4);

        L(l5a0);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        lea(I, ptr[A1 + LDA * 8]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l698, T_NEAR);
        align(4);

        L(l5bc);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -16);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x60], xmm1);
        movdqu(xword[B - 0x40], xmm4);
        movdqu(xword[B - 0x20], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
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
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        movdqu(xword[B - 0x30], xmm4);
        movdqu(xword[B - 0x10], xmm3);
        sub(B, -128);
        dec(I);
        jg(l5bc, T_NEAR);
        align(4);

        L(l698);
        test(M, 0x8);
        jle(l728, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x60], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        align(4);

        L(l728);
        test(M, 0x4);
        jle(l798, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(l798);
        test(M, 0x2);
        jle(l80c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l80c);
        test(M, 0x1);
        jle(l878, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x7);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l878);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l5a0, T_NEAR);
        align(4);

        L(l888);
        cmp(N, 0x4);
        jl(la4c, T_NEAR);
        align(4);

        L(l894);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l920, T_NEAR);
        align(4);

        L(l8ac);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm3);
        sub(B, -64);
        dec(I);
        jg(l8ac, T_NEAR);
        align(4);

        L(l920);
        test(M, 0x8);
        jle(l970, T_NEAR);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l970);
        test(M, 0x4);
        jle(l9b4, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        movd(xmm2, dword[A2 - 0x80]);
        movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l9b4);
        test(M, 0x2);
        jle(l9fc, T_NEAR);
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
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l9fc);
        test(M, 0x1);
        jle(la3c, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(la3c);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l894, T_NEAR);
        align(4);

        L(la4c);
        cmp(N, 0x2);
        jl(lb5e, T_NEAR);
        align(4);

        L(la58);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(laa8, T_NEAR);
        align(4);

        L(la70);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        movdqa(xmm2, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        dec(I);
        jg(la70, T_NEAR);
        align(4);

        L(laa8);
        test(M, 0x8);
        jle(lad4, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(lad4);
        test(M, 0x4);
        jle(lb00, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lb00);
        test(M, 0x2);
        jle(lb30, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lb30);
        test(M, 0x1);
        jle(lb50, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(lb50);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(la58, T_NEAR);
        align(4);

        L(lb5e);
        cmp(N, 0x1);
        jl(lc18, T_NEAR);
        align(4);

        L(lb68);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(lb94, T_NEAR);
        align(4);

        L(lb78);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(lb78, T_NEAR);
        align(4);

        L(lb94);
        test(M, 0x8);
        jle(lbb4, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lbb4);
        test(M, 0x4);
        jle(lbd4, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lbd4);
        test(M, 0x2);
        jle(lbf0, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(lbf0);
        test(M, 0x1);
        jle(lc08, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(lc08);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(lb68, T_NEAR);
        align(4);

        L(lc18);

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
