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

jit_sse41_f32_copy_bn_kern::jit_sse41_f32_copy_bn_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_bn_kern::generate() {

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

        Xbyak::Label l11c;
        Xbyak::Label l15c;
        Xbyak::Label l16c;
        Xbyak::Label l18c;
        Xbyak::Label l1c8;
        Xbyak::Label l1f8;
        Xbyak::Label l224;
        Xbyak::Label l228;
        Xbyak::Label l248;
        Xbyak::Label l284;
        Xbyak::Label l2b4;
        Xbyak::Label l2d8;
        Xbyak::Label l2dc;
        Xbyak::Label l2e4;
        Xbyak::Label l300;
        Xbyak::Label l318;
        Xbyak::Label l388;
        Xbyak::Label l3dc;
        Xbyak::Label l420;
        Xbyak::Label l430;
        Xbyak::Label l450;
        Xbyak::Label l490;
        Xbyak::Label l4c4;
        Xbyak::Label l4f4;
        Xbyak::Label l4f8;
        Xbyak::Label l518;
        Xbyak::Label l54;
        Xbyak::Label l558;
        Xbyak::Label l58c;
        Xbyak::Label l5b4;
        Xbyak::Label l5b8;
        Xbyak::Label l5c0;
        Xbyak::Label l5cc;
        Xbyak::Label l5e4;
        Xbyak::Label l654;
        Xbyak::Label l6a8;
        Xbyak::Label l6c;
        Xbyak::Label l6ec;
        Xbyak::Label l6fc;
        Xbyak::Label l71c;
        Xbyak::Label l75c;
        Xbyak::Label l790;
        Xbyak::Label l7c0;
        Xbyak::Label l7c4;
        Xbyak::Label l7e4;
        Xbyak::Label l824;
        Xbyak::Label l858;
        Xbyak::Label l880;
        Xbyak::Label l884;
        Xbyak::Label ld0;

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
        jne(l2e4, T_NEAR);
        cmp(N, 0x4);
        jl(l16c, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(ld0, T_NEAR);
        align(4);

        L(l6c);
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
        jg(l6c, T_NEAR);
        align(4);

        L(ld0);
        test(M, 0x2);
        jle(l11c, T_NEAR);
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

        L(l11c);
        test(M, 0x1);
        jle(l15c, T_NEAR);
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

        L(l15c);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l54, T_NEAR);
        align(4);

        L(l16c);
        cmp(N, 0x2);
        jl(l228, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l1c8, T_NEAR);
        align(4);

        L(l18c);
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
        jg(l18c, T_NEAR);
        align(4);

        L(l1c8);
        test(M, 0x2);
        jle(l1f8, T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l1f8);
        test(M, 0x1);
        jle(l224, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l224);
        sub(N, 0x2);
        align(4);

        L(l228);
        cmp(N, 0x1);
        jl(l2dc, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l284, T_NEAR);
        align(4);

        L(l248);
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
        jg(l248, T_NEAR);
        align(4);

        L(l284);
        test(M, 0x2);
        jle(l2b4, T_NEAR);
        movsd(xmm0, qword[A1]);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l2b4);
        test(M, 0x1);
        jle(l2d8, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l2d8);
        sub(N, 0x1);
        align(4);

        L(l2dc);
        jmp(l884, T_NEAR);
        align(4);

        L(l2e4);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(l5c0, T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x4);
        jl(l430, T_NEAR);
        align(4);

        L(l300);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l388, T_NEAR);
        align(4);

        L(l318);
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
        jg(l318, T_NEAR);
        align(4);

        L(l388);
        test(M, 0x2);
        jle(l3dc, T_NEAR);
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

        L(l3dc);
        test(M, 0x1);
        jle(l420, T_NEAR);
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

        L(l420);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l300, T_NEAR);
        align(4);

        L(l430);
        cmp(N, 0x2);
        jl(l4f8, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l490, T_NEAR);
        align(4);

        L(l450);
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
        jg(l450, T_NEAR);
        align(4);

        L(l490);
        test(M, 0x2);
        jle(l4c4, T_NEAR);
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

        L(l4c4);
        test(M, 0x1);
        jle(l4f4, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l4f4);
        sub(N, 0x2);
        align(4);

        L(l4f8);
        cmp(N, 0x1);
        jl(l5b8, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l558, T_NEAR);
        align(4);

        L(l518);
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
        jg(l518, T_NEAR);
        align(4);

        L(l558);
        test(M, 0x2);
        jle(l58c, T_NEAR);
        movsd(xmm0, qword[A1]);
        xorps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l58c);
        test(M, 0x1);
        jle(l5b4, T_NEAR);
        movss(xmm0, dword[A1]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l5b4);
        sub(N, 0x1);
        align(4);

        L(l5b8);
        jmp(l884, T_NEAR);
        align(4);

        L(l5c0);
        cmp(N, 0x4);
        jl(l6fc, T_NEAR);
        align(4);

        L(l5cc);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l654, T_NEAR);
        align(4);

        L(l5e4);
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
        jg(l5e4, T_NEAR);
        align(4);

        L(l654);
        test(M, 0x2);
        jle(l6a8, T_NEAR);
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

        L(l6a8);
        test(M, 0x1);
        jle(l6ec, T_NEAR);
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

        L(l6ec);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l5cc, T_NEAR);
        align(4);

        L(l6fc);
        cmp(N, 0x2);
        jl(l7c4, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l75c, T_NEAR);
        align(4);

        L(l71c);
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
        jg(l71c, T_NEAR);
        align(4);

        L(l75c);
        test(M, 0x2);
        jle(l790, T_NEAR);
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

        L(l790);
        test(M, 0x1);
        jle(l7c0, T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l7c0);
        sub(N, 0x2);
        align(4);

        L(l7c4);
        cmp(N, 0x1);
        jl(l884, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l824, T_NEAR);
        align(4);

        L(l7e4);
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
        jg(l7e4, T_NEAR);
        align(4);

        L(l824);
        test(M, 0x2);
        jle(l858, T_NEAR);
        movsd(xmm0, qword[A1]);
        mulps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l858);
        test(M, 0x1);
        jle(l880, T_NEAR);
        movss(xmm0, dword[A1]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l880);
        sub(N, 0x1);
        align(4);

        L(l884);

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
