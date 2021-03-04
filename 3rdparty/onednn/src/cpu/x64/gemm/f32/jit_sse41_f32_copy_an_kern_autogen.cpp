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

jit_sse41_f32_copy_an_kern::jit_sse41_f32_copy_an_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_an_kern::generate() {

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

        Xbyak::Label l1020;
        Xbyak::Label l10b4;
        Xbyak::Label l1104;
        Xbyak::Label l1134;
        Xbyak::Label l1150;
        Xbyak::Label l1154;
        Xbyak::Label l130;
        Xbyak::Label l198;
        Xbyak::Label l1d4;
        Xbyak::Label l1f8;
        Xbyak::Label l208;
        Xbyak::Label l228;
        Xbyak::Label l290;
        Xbyak::Label l2cc;
        Xbyak::Label l2f4;
        Xbyak::Label l30c;
        Xbyak::Label l310;
        Xbyak::Label l330;
        Xbyak::Label l3a0;
        Xbyak::Label l3e0;
        Xbyak::Label l408;
        Xbyak::Label l420;
        Xbyak::Label l424;
        Xbyak::Label l444;
        Xbyak::Label l4bc;
        Xbyak::Label l500;
        Xbyak::Label l52c;
        Xbyak::Label l54;
        Xbyak::Label l548;
        Xbyak::Label l54c;
        Xbyak::Label l554;
        Xbyak::Label l570;
        Xbyak::Label l588;
        Xbyak::Label l67c;
        Xbyak::Label l6c;
        Xbyak::Label l6fc;
        Xbyak::Label l744;
        Xbyak::Label l76c;
        Xbyak::Label l77c;
        Xbyak::Label l7a0;
        Xbyak::Label l820;
        Xbyak::Label l868;
        Xbyak::Label l894;
        Xbyak::Label l8b0;
        Xbyak::Label l8b4;
        Xbyak::Label l8d8;
        Xbyak::Label l964;
        Xbyak::Label l9b0;
        Xbyak::Label l9e0;
        Xbyak::Label l9fc;
        Xbyak::Label la00;
        Xbyak::Label la24;
        Xbyak::Label lab8;
        Xbyak::Label lb08;
        Xbyak::Label lb38;
        Xbyak::Label lb54;
        Xbyak::Label lb58;
        Xbyak::Label lb60;
        Xbyak::Label lb6c;
        Xbyak::Label lb84;
        Xbyak::Label lc78;
        Xbyak::Label lcf8;
        Xbyak::Label ld40;
        Xbyak::Label ld68;
        Xbyak::Label ld78;
        Xbyak::Label ld9c;
        Xbyak::Label le1c;
        Xbyak::Label le64;
        Xbyak::Label le90;
        Xbyak::Label leac;
        Xbyak::Label leb0;
        Xbyak::Label led4;
        Xbyak::Label lf60;
        Xbyak::Label lfac;
        Xbyak::Label lfdc;
        Xbyak::Label lff8;
        Xbyak::Label lffc;

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
        jne(l554, T_NEAR);
        cmp(N, 0x8);
        jl(l208, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(l130, T_NEAR);
        align(4);

        L(l6c);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(l6c, T_NEAR);
        align(4);

        L(l130);
        test(M, 0x4);
        jle(l198, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(l198);
        test(M, 0x2);
        jle(l1d4, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l1d4);
        test(M, 0x1);
        jle(l1f8, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(l1f8);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l54, T_NEAR);
        align(4);

        L(l208);
        cmp(N, 0x4);
        jl(l310, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l290, T_NEAR);
        align(4);

        L(l228);
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
        jg(l228, T_NEAR);
        align(4);

        L(l290);
        test(M, 0x4);
        jle(l2cc, T_NEAR);
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

        L(l2cc);
        test(M, 0x2);
        jle(l2f4, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l2f4);
        test(M, 0x1);
        jle(l30c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l30c);
        sub(N, 0x4);
        align(4);

        L(l310);
        cmp(N, 0x2);
        jl(l424, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l3a0, T_NEAR);
        align(4);

        L(l330);
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
        jg(l330, T_NEAR);
        align(4);

        L(l3a0);
        test(M, 0x4);
        jle(l3e0, T_NEAR);
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

        L(l3e0);
        test(M, 0x2);
        jle(l408, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l408);
        test(M, 0x1);
        jle(l420, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l420);
        sub(N, 0x2);
        align(4);

        L(l424);
        cmp(N, 0x1);
        jl(l54c, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l4bc, T_NEAR);
        align(4);

        L(l444);
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
        jg(l444, T_NEAR);
        align(4);

        L(l4bc);
        test(M, 0x4);
        jle(l500, T_NEAR);
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

        L(l500);
        test(M, 0x2);
        jle(l52c, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l52c);
        test(M, 0x1);
        jle(l548, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l548);
        sub(N, 0x1);
        align(4);

        L(l54c);
        jmp(l1154, T_NEAR);
        align(4);

        L(l554);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(lb60, T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x8);
        jl(l77c, T_NEAR);
        align(4);

        L(l570);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(l67c, T_NEAR);
        align(4);

        L(l588);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(l588, T_NEAR);
        align(4);

        L(l67c);
        test(M, 0x4);
        jle(l6fc, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(l6fc);
        test(M, 0x2);
        jle(l744, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l744);
        test(M, 0x1);
        jle(l76c, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(l76c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l570, T_NEAR);
        align(4);

        L(l77c);
        cmp(N, 0x4);
        jl(l8b4, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l820, T_NEAR);
        align(4);

        L(l7a0);
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
        jg(l7a0, T_NEAR);
        align(4);

        L(l820);
        test(M, 0x4);
        jle(l868, T_NEAR);
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

        L(l868);
        test(M, 0x2);
        jle(l894, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l894);
        test(M, 0x1);
        jle(l8b0, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l8b0);
        sub(N, 0x4);
        align(4);

        L(l8b4);
        cmp(N, 0x2);
        jl(la00, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l964, T_NEAR);
        align(4);

        L(l8d8);
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
        jg(l8d8, T_NEAR);
        align(4);

        L(l964);
        test(M, 0x4);
        jle(l9b0, T_NEAR);
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

        L(l9b0);
        test(M, 0x2);
        jle(l9e0, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l9e0);
        test(M, 0x1);
        jle(l9fc, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l9fc);
        sub(N, 0x2);
        align(4);

        L(la00);
        cmp(N, 0x1);
        jl(lb58, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(lab8, T_NEAR);
        align(4);

        L(la24);
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
        jg(la24, T_NEAR);
        align(4);

        L(lab8);
        test(M, 0x4);
        jle(lb08, T_NEAR);
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

        L(lb08);
        test(M, 0x2);
        jle(lb38, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(lb38);
        test(M, 0x1);
        jle(lb54, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lb54);
        sub(N, 0x1);
        align(4);

        L(lb58);
        jmp(l1154, T_NEAR);
        align(4);

        L(lb60);
        cmp(N, 0x8);
        jl(ld78, T_NEAR);
        align(4);

        L(lb6c);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(lc78, T_NEAR);
        align(4);

        L(lb84);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(lb84, T_NEAR);
        align(4);

        L(lc78);
        test(M, 0x4);
        jle(lcf8, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(lcf8);
        test(M, 0x2);
        jle(ld40, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(ld40);
        test(M, 0x1);
        jle(ld68, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(ld68);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(lb6c, T_NEAR);
        align(4);

        L(ld78);
        cmp(N, 0x4);
        jl(leb0, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(le1c, T_NEAR);
        align(4);

        L(ld9c);
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
        jg(ld9c, T_NEAR);
        align(4);

        L(le1c);
        test(M, 0x4);
        jle(le64, T_NEAR);
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

        L(le64);
        test(M, 0x2);
        jle(le90, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(le90);
        test(M, 0x1);
        jle(leac, T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(leac);
        sub(N, 0x4);
        align(4);

        L(leb0);
        cmp(N, 0x2);
        jl(lffc, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(lf60, T_NEAR);
        align(4);

        L(led4);
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
        jg(led4, T_NEAR);
        align(4);

        L(lf60);
        test(M, 0x4);
        jle(lfac, T_NEAR);
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

        L(lfac);
        test(M, 0x2);
        jle(lfdc, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(lfdc);
        test(M, 0x1);
        jle(lff8, T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lff8);
        sub(N, 0x2);
        align(4);

        L(lffc);
        cmp(N, 0x1);
        jl(l1154, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l10b4, T_NEAR);
        align(4);

        L(l1020);
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
        jg(l1020, T_NEAR);
        align(4);

        L(l10b4);
        test(M, 0x4);
        jle(l1104, T_NEAR);
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

        L(l1104);
        test(M, 0x2);
        jle(l1134, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l1134);
        test(M, 0x1);
        jle(l1150, T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l1150);
        sub(N, 0x1);
        align(4);

        L(l1154);

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
