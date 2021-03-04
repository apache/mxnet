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

#include "cpu/x64/gemm/f32/common_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_sse41_f32_copy_at_kern::jit_sse41_f32_copy_at_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_at_kern::generate() {

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
#define ALPHA rsi
#define B rdi
#define I rax
#define A1 r10
#define A2 rsi
#define LDA3 r11

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp

#endif

    inLocalLabel();
    {

        Xbyak::Label l130;
        Xbyak::Label l1b4;
        Xbyak::Label l21c;
        Xbyak::Label l22c;
        Xbyak::Label l24c;
        Xbyak::Label l2b0;
        Xbyak::Label l2fc;
        Xbyak::Label l33c;
        Xbyak::Label l340;
        Xbyak::Label l360;
        Xbyak::Label l39c;
        Xbyak::Label l3cc;
        Xbyak::Label l3f8;
        Xbyak::Label l3fc;
        Xbyak::Label l41c;
        Xbyak::Label l458;
        Xbyak::Label l488;
        Xbyak::Label l4ac;
        Xbyak::Label l4b0;
        Xbyak::Label l4b8;
        Xbyak::Label l4d4;
        Xbyak::Label l4f0;
        Xbyak::Label l54;
        Xbyak::Label l5c8;
        Xbyak::Label l65c;
        Xbyak::Label l6cc;
        Xbyak::Label l6dc;
        Xbyak::Label l6fc;
        Xbyak::Label l70;
        Xbyak::Label l76c;
        Xbyak::Label l7c0;
        Xbyak::Label l804;
        Xbyak::Label l808;
        Xbyak::Label l828;
        Xbyak::Label l868;
        Xbyak::Label l89c;
        Xbyak::Label l8cc;
        Xbyak::Label l8d0;
        Xbyak::Label l8f0;
        Xbyak::Label l930;
        Xbyak::Label l964;
        Xbyak::Label l98c;
        Xbyak::Label l990;
        Xbyak::Label l998;
        Xbyak::Label l9a4;
        Xbyak::Label l9c0;
        Xbyak::Label la98;
        Xbyak::Label lb2c;
        Xbyak::Label lb9c;
        Xbyak::Label lbac;
        Xbyak::Label lbcc;
        Xbyak::Label lc3c;
        Xbyak::Label lc90;
        Xbyak::Label lcd4;
        Xbyak::Label lcd8;
        Xbyak::Label lcf8;
        Xbyak::Label ld38;
        Xbyak::Label ld6c;
        Xbyak::Label ld9c;
        Xbyak::Label lda0;
        Xbyak::Label ldc0;
        Xbyak::Label le00;
        Xbyak::Label le34;
        Xbyak::Label le5c;
        Xbyak::Label le60;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        sub(A, 0x0);
        sub(B, -128);
        shl(LDA, 0x2);
        lea(LDA3, ptr[LDA + LDA * 2]);
        movss(xmm6, dword[ALPHA]);
        pshufd(xmm6, xmm6, 0x0);
        pcmpeqb(xmm3, xmm3);
        psrld(xmm3, 0x17);
        pslld(xmm3, 0x19);
        psrld(xmm3, 0x2);
        pcmpeqb(xmm4, xmm4);
        pslld(xmm4, 0x1f);
        ucomiss(xmm6, xmm3);
        jne(l4b8, T_NEAR);
        cmp(N, 0x8);
        jl(l22c, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l130, T_NEAR);
        align(4);

        L(l70);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(l70, T_NEAR);
        align(4);

        L(l130);
        test(M, 0x2);
        jle(l1b4, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(l1b4);
        test(M, 0x1);
        jle(l21c, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(l21c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l54, T_NEAR);
        align(4);

        L(l22c);
        cmp(N, 0x4);
        jl(l340, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l2b0, T_NEAR);
        align(4);

        L(l24c);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(l24c, T_NEAR);
        align(4);

        L(l2b0);
        test(M, 0x2);
        jle(l2fc, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(l2fc);
        test(M, 0x1);
        jle(l33c, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(l33c);
        sub(N, 0x4);
        align(4);

        L(l340);
        cmp(N, 0x2);
        jl(l3fc, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l39c, T_NEAR);
        align(4);

        L(l360);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(l360, T_NEAR);
        align(4);

        L(l39c);
        test(M, 0x2);
        jle(l3cc, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l3cc);
        test(M, 0x1);
        jle(l3f8, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l3f8);
        sub(N, 0x2);
        align(4);

        L(l3fc);
        cmp(N, 0x1);
        jl(l4b0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l458, T_NEAR);
        align(4);

        L(l41c);
        movups(xmm0, xword[A1]);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(l41c, T_NEAR);
        align(4);

        L(l458);
        test(M, 0x2);
        jle(l488, T_NEAR);
        movsd(xmm0, qword[A1]);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l488);
        test(M, 0x1);
        jle(l4ac, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l4ac);
        sub(N, 0x1);
        align(4);

        L(l4b0);
        jmp(le60, T_NEAR);
        align(4);

        L(l4b8);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(l998, T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x8);
        jl(l6dc, T_NEAR);
        align(4);

        L(l4d4);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l5c8, T_NEAR);
        align(4);

        L(l4f0);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(l4f0, T_NEAR);
        align(4);

        L(l5c8);
        test(M, 0x2);
        jle(l65c, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(l65c);
        test(M, 0x1);
        jle(l6cc, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(l6cc);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l4d4, T_NEAR);
        align(4);

        L(l6dc);
        cmp(N, 0x4);
        jl(l808, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l76c, T_NEAR);
        align(4);

        L(l6fc);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(l6fc, T_NEAR);
        align(4);

        L(l76c);
        test(M, 0x2);
        jle(l7c0, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(l7c0);
        test(M, 0x1);
        jle(l804, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(l804);
        sub(N, 0x4);
        align(4);

        L(l808);
        cmp(N, 0x2);
        jl(l8d0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l868, T_NEAR);
        align(4);

        L(l828);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(l828, T_NEAR);
        align(4);

        L(l868);
        test(M, 0x2);
        jle(l89c, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l89c);
        test(M, 0x1);
        jle(l8cc, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l8cc);
        sub(N, 0x2);
        align(4);

        L(l8d0);
        cmp(N, 0x1);
        jl(l990, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l930, T_NEAR);
        align(4);

        L(l8f0);
        movups(xmm0, xword[A1]);
        xorps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(l8f0, T_NEAR);
        align(4);

        L(l930);
        test(M, 0x2);
        jle(l964, T_NEAR);
        movsd(xmm0, qword[A1]);
        xorps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l964);
        test(M, 0x1);
        jle(l98c, T_NEAR);
        movss(xmm0, dword[A1]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l98c);
        sub(N, 0x1);
        align(4);

        L(l990);
        jmp(le60, T_NEAR);
        align(4);

        L(l998);
        cmp(N, 0x8);
        jl(lbac, T_NEAR);
        align(4);

        L(l9a4);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(la98, T_NEAR);
        align(4);

        L(l9c0);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(l9c0, T_NEAR);
        align(4);

        L(la98);
        test(M, 0x2);
        jle(lb2c, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(lb2c);
        test(M, 0x1);
        jle(lb9c, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(lb9c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l9a4, T_NEAR);
        align(4);

        L(lbac);
        cmp(N, 0x4);
        jl(lcd8, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lc3c, T_NEAR);
        align(4);

        L(lbcc);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(lbcc, T_NEAR);
        align(4);

        L(lc3c);
        test(M, 0x2);
        jle(lc90, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(lc90);
        test(M, 0x1);
        jle(lcd4, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(lcd4);
        sub(N, 0x4);
        align(4);

        L(lcd8);
        cmp(N, 0x2);
        jl(lda0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(ld38, T_NEAR);
        align(4);

        L(lcf8);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(lcf8, T_NEAR);
        align(4);

        L(ld38);
        test(M, 0x2);
        jle(ld6c, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(ld6c);
        test(M, 0x1);
        jle(ld9c, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(ld9c);
        sub(N, 0x2);
        align(4);

        L(lda0);
        cmp(N, 0x1);
        jl(le60, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(le00, T_NEAR);
        align(4);

        L(ldc0);
        movups(xmm0, xword[A1]);
        mulps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(ldc0, T_NEAR);
        align(4);

        L(le00);
        test(M, 0x2);
        jle(le34, T_NEAR);
        movsd(xmm0, qword[A1]);
        mulps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(le34);
        test(M, 0x1);
        jle(le5c, T_NEAR);
        movss(xmm0, dword[A1]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(le5c);
        sub(N, 0x1);
        align(4);

        L(le60);

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
