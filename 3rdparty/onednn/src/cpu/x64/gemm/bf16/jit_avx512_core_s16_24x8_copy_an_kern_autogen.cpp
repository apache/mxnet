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

jit_avx512_core_s16_24x8_copy_an_kern::jit_avx512_core_s16_24x8_copy_an_kern()
    : jit_generator(nullptr, S16_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_s16_24x8_copy_an_kern::generate() {

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

        Xbyak::Label l118;
        Xbyak::Label l18c;
        Xbyak::Label l1c0;
        Xbyak::Label l1d0;
        Xbyak::Label l1dc;
        Xbyak::Label l1f0;
        Xbyak::Label l24;
        Xbyak::Label l28c;
        Xbyak::Label l2e0;
        Xbyak::Label l308;
        Xbyak::Label l318;
        Xbyak::Label l324;
        Xbyak::Label l338;
        Xbyak::Label l38;
        Xbyak::Label l3e0;
        Xbyak::Label l43c;
        Xbyak::Label l470;
        Xbyak::Label l48c;
        Xbyak::Label l49c;
        Xbyak::Label l4a8;
        Xbyak::Label l4b8;
        Xbyak::Label l534;
        Xbyak::Label l57c;
        Xbyak::Label l5a8;
        Xbyak::Label l5c4;
        Xbyak::Label l5d4;
        Xbyak::Label l5e0;
        Xbyak::Label l5f0;
        Xbyak::Label l668;
        Xbyak::Label l6ac;
        Xbyak::Label l6d8;
        Xbyak::Label l6f4;
        Xbyak::Label l702;
        Xbyak::Label l70c;
        Xbyak::Label l71c;
        Xbyak::Label l794;
        Xbyak::Label l7dc;
        Xbyak::Label l80c;
        Xbyak::Label l824;
        Xbyak::Label l834;

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
        cmp(N, 0x18);
        jl(l1d0, T_NEAR);
        align(4);

        L(l24);
        mov(A1, A);
        add(A, 0x30);
        mov(I, M);
        sar(I, 0x2);
        jle(l118, T_NEAR);
        align(4);

        L(l38);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x60]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x60]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x40], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x60]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x60]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -192);
        dec(I);
        jg(l38, T_NEAR);
        align(4);

        L(l118);
        test(M, 0x2);
        jle(l18c, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x60]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x60]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x40], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -96);
        align(4);

        L(l18c);
        test(M, 0x1);
        jle(l1c0, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 - 0x70]);
        vmovdqu(xmm2, xword[A1 - 0x60]);
        add(A1, LDA);
        vmovdqu(xword[B - 0x80], xmm0);
        vmovdqu(xword[B - 0x70], xmm1);
        vmovdqu(xword[B - 0x60], xmm2);
        sub(B, -48);
        align(4);

        L(l1c0);
        sub(N, 0x18);
        cmp(N, 0x18);
        jge(l24, T_NEAR);
        align(4);

        L(l1d0);
        cmp(N, 0x10);
        jl(l318, T_NEAR);
        align(4);

        L(l1dc);
        mov(A1, A);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x2);
        jle(l28c, T_NEAR);
        align(4);

        L(l1f0);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x60], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x40], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -128);
        dec(I);
        jg(l1f0, T_NEAR);
        align(4);

        L(l28c);
        test(M, 0x2);
        jle(l2e0, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x60], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l2e0);
        test(M, 0x1);
        jle(l308, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 - 0x70]);
        vmovdqu(xword[B - 0x80], xmm0);
        vmovdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l308);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l1dc, T_NEAR);
        align(4);

        L(l318);
        cmp(N, 0x8);
        jl(l49c, T_NEAR);
        align(4);

        L(l324);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l3e0, T_NEAR);
        align(4);

        L(l338);
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
        jg(l338, T_NEAR);
        align(4);

        L(l3e0);
        test(M, 0x4);
        jle(l43c, T_NEAR);
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

        L(l43c);
        test(M, 0x2);
        jle(l470, T_NEAR);
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

        L(l470);
        test(M, 0x1);
        jle(l48c, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l48c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l324, T_NEAR);
        align(4);

        L(l49c);
        cmp(N, 0x4);
        jl(l5d4, T_NEAR);
        align(4);

        L(l4a8);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l534, T_NEAR);
        align(4);

        L(l4b8);
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
        jg(l4b8, T_NEAR);
        align(4);

        L(l534);
        test(M, 0x4);
        jle(l57c, T_NEAR);
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

        L(l57c);
        test(M, 0x2);
        jle(l5a8, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l5a8);
        test(M, 0x1);
        jle(l5c4, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l5c4);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l4a8, T_NEAR);
        align(4);

        L(l5d4);
        cmp(N, 0x2);
        jl(l702, T_NEAR);
        align(4);

        L(l5e0);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l668, T_NEAR);
        align(4);

        L(l5f0);
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
        jg(l5f0, T_NEAR);
        align(4);

        L(l668);
        test(M, 0x4);
        jle(l6ac, T_NEAR);
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

        L(l6ac);
        test(M, 0x2);
        jle(l6d8, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l6d8);
        test(M, 0x1);
        jle(l6f4, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l6f4);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l5e0, T_NEAR);
        align(4);

        L(l702);
        cmp(N, 0x1);
        jl(l834, T_NEAR);
        align(4);

        L(l70c);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l794, T_NEAR);
        align(4);

        L(l71c);
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
        jg(l71c, T_NEAR);
        align(4);

        L(l794);
        test(M, 0x4);
        jle(l7dc, T_NEAR);
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

        L(l7dc);
        test(M, 0x2);
        jle(l80c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l80c);
        test(M, 0x1);
        jle(l824, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l824);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l70c, T_NEAR);
        align(4);

        L(l834);
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
