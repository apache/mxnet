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

jit_sse41_f32_copy_bt_kern::jit_sse41_f32_copy_bt_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_bt_kern::generate() {

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

        Xbyak::Label l10c;
        Xbyak::Label l134;
        Xbyak::Label l14c;
        Xbyak::Label l15c;
        Xbyak::Label l17c;
        Xbyak::Label l1ec;
        Xbyak::Label l22c;
        Xbyak::Label l254;
        Xbyak::Label l26c;
        Xbyak::Label l270;
        Xbyak::Label l290;
        Xbyak::Label l308;
        Xbyak::Label l34c;
        Xbyak::Label l378;
        Xbyak::Label l394;
        Xbyak::Label l398;
        Xbyak::Label l3a0;
        Xbyak::Label l3bc;
        Xbyak::Label l3d4;
        Xbyak::Label l454;
        Xbyak::Label l49c;
        Xbyak::Label l4c8;
        Xbyak::Label l4e4;
        Xbyak::Label l4f4;
        Xbyak::Label l518;
        Xbyak::Label l54;
        Xbyak::Label l5a4;
        Xbyak::Label l5f0;
        Xbyak::Label l620;
        Xbyak::Label l63c;
        Xbyak::Label l640;
        Xbyak::Label l664;
        Xbyak::Label l68;
        Xbyak::Label l6f8;
        Xbyak::Label l748;
        Xbyak::Label l778;
        Xbyak::Label l794;
        Xbyak::Label l798;
        Xbyak::Label l7a0;
        Xbyak::Label l7ac;
        Xbyak::Label l7c4;
        Xbyak::Label l844;
        Xbyak::Label l88c;
        Xbyak::Label l8b8;
        Xbyak::Label l8d4;
        Xbyak::Label l8e4;
        Xbyak::Label l908;
        Xbyak::Label l994;
        Xbyak::Label l9e0;
        Xbyak::Label la10;
        Xbyak::Label la2c;
        Xbyak::Label la30;
        Xbyak::Label la54;
        Xbyak::Label lae8;
        Xbyak::Label lb38;
        Xbyak::Label lb68;
        Xbyak::Label lb84;
        Xbyak::Label lb88;
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
        sub(A, -128);
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
        jne(l3a0, T_NEAR);
        cmp(N, 0x4);
        jl(l15c, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(ld0, T_NEAR);
        align(4);

        L(l68);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(l68, T_NEAR);
        align(4);

        L(ld0);
        test(M, 0x4);
        jle(l10c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(l10c);
        test(M, 0x2);
        jle(l134, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l134);
        test(M, 0x1);
        jle(l14c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l14c);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l54, T_NEAR);
        align(4);

        L(l15c);
        cmp(N, 0x2);
        jl(l270, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l1ec, T_NEAR);
        align(4);

        L(l17c);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(l17c, T_NEAR);
        align(4);

        L(l1ec);
        test(M, 0x4);
        jle(l22c, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(l22c);
        test(M, 0x2);
        jle(l254, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l254);
        test(M, 0x1);
        jle(l26c, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l26c);
        sub(N, 0x2);
        align(4);

        L(l270);
        cmp(N, 0x1);
        jl(l398, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l308, T_NEAR);
        align(4);

        L(l290);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(l290, T_NEAR);
        align(4);

        L(l308);
        test(M, 0x4);
        jle(l34c, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(l34c);
        test(M, 0x2);
        jle(l378, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l378);
        test(M, 0x1);
        jle(l394, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l394);
        sub(N, 0x1);
        align(4);

        L(l398);
        jmp(lb88, T_NEAR);
        align(4);

        L(l3a0);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(l7a0, T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x4);
        jl(l4f4, T_NEAR);
        align(4);

        L(l3bc);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l454, T_NEAR);
        align(4);

        L(l3d4);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(l3d4, T_NEAR);
        align(4);

        L(l454);
        test(M, 0x4);
        jle(l49c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(l49c);
        test(M, 0x2);
        jle(l4c8, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l4c8);
        test(M, 0x1);
        jle(l4e4, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l4e4);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l3bc, T_NEAR);
        align(4);

        L(l4f4);
        cmp(N, 0x2);
        jl(l640, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l5a4, T_NEAR);
        align(4);

        L(l518);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(l518, T_NEAR);
        align(4);

        L(l5a4);
        test(M, 0x4);
        jle(l5f0, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(l5f0);
        test(M, 0x2);
        jle(l620, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l620);
        test(M, 0x1);
        jle(l63c, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l63c);
        sub(N, 0x2);
        align(4);

        L(l640);
        cmp(N, 0x1);
        jl(l798, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l6f8, T_NEAR);
        align(4);

        L(l664);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(l664, T_NEAR);
        align(4);

        L(l6f8);
        test(M, 0x4);
        jle(l748, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(l748);
        test(M, 0x2);
        jle(l778, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l778);
        test(M, 0x1);
        jle(l794, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l794);
        sub(N, 0x1);
        align(4);

        L(l798);
        jmp(lb88, T_NEAR);
        align(4);

        L(l7a0);
        cmp(N, 0x4);
        jl(l8e4, T_NEAR);
        align(4);

        L(l7ac);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l844, T_NEAR);
        align(4);

        L(l7c4);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(l7c4, T_NEAR);
        align(4);

        L(l844);
        test(M, 0x4);
        jle(l88c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(l88c);
        test(M, 0x2);
        jle(l8b8, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l8b8);
        test(M, 0x1);
        jle(l8d4, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l8d4);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l7ac, T_NEAR);
        align(4);

        L(l8e4);
        cmp(N, 0x2);
        jl(la30, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l994, T_NEAR);
        align(4);

        L(l908);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(l908, T_NEAR);
        align(4);

        L(l994);
        test(M, 0x4);
        jle(l9e0, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(l9e0);
        test(M, 0x2);
        jle(la10, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(la10);
        test(M, 0x1);
        jle(la2c, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(la2c);
        sub(N, 0x2);
        align(4);

        L(la30);
        cmp(N, 0x1);
        jl(lb88, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(lae8, T_NEAR);
        align(4);

        L(la54);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(la54, T_NEAR);
        align(4);

        L(lae8);
        test(M, 0x4);
        jle(lb38, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(lb38);
        test(M, 0x2);
        jle(lb68, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(lb68);
        test(M, 0x1);
        jle(lb84, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lb84);
        sub(N, 0x1);
        align(4);

        L(lb88);

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
