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

#include "cpu/x64/gemm/bf16/common_s16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_avx512_core_s16_48x8_copy_an_kern::jit_avx512_core_s16_48x8_copy_an_kern()
    : jit_generator(nullptr, S16_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_s16_48x8_copy_an_kern::generate() {

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

        Xbyak::Label l1e8;
        Xbyak::Label l24;
        Xbyak::Label l2c8;
        Xbyak::Label l2fc;
        Xbyak::Label l30c;
        Xbyak::Label l318;
        Xbyak::Label l32c;
        Xbyak::Label l38;
        Xbyak::Label l44c;
        Xbyak::Label l4e8;
        Xbyak::Label l510;
        Xbyak::Label l520;
        Xbyak::Label l52c;
        Xbyak::Label l540;
        Xbyak::Label l5dc;
        Xbyak::Label l630;
        Xbyak::Label l658;
        Xbyak::Label l668;
        Xbyak::Label l674;
        Xbyak::Label l688;
        Xbyak::Label l730;
        Xbyak::Label l78c;
        Xbyak::Label l7c0;
        Xbyak::Label l7dc;
        Xbyak::Label l7ec;
        Xbyak::Label l7f8;
        Xbyak::Label l808;
        Xbyak::Label l884;
        Xbyak::Label l8cc;
        Xbyak::Label l8f8;
        Xbyak::Label l914;
        Xbyak::Label l924;
        Xbyak::Label l930;
        Xbyak::Label l940;
        Xbyak::Label l9b8;
        Xbyak::Label l9fc;
        Xbyak::Label la28;
        Xbyak::Label la44;
        Xbyak::Label la52;
        Xbyak::Label la5c;
        Xbyak::Label la6c;
        Xbyak::Label lae4;
        Xbyak::Label lb2c;
        Xbyak::Label lb5c;
        Xbyak::Label lb74;
        Xbyak::Label lb84;

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
        cmp(N, 0x30);
        jl(l30c, T_NEAR);
        align(4);

        L(l24);
        mov(A1, A);
        add(A, 0x60);
        mov(I, M);
        sar(I, 0x2);
        jle(l1e8, T_NEAR);
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
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x40]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x40]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x30]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x30]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x40], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x60], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x60]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x60]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x80], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0xa0], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x40]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x40]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0xc0], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x30]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x30]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0xe0], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -384);
        dec(I);
        jg(l38, T_NEAR);
        align(4);

        L(l1e8);
        test(M, 0x2);
        jle(l2c8, T_NEAR);
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
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x40]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x40]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x30]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x30]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -192);
        align(4);

        L(l2c8);
        test(M, 0x1);
        jle(l2fc, T_NEAR);
        vmovdqu(ymm0, yword[A1 - 0x80]);
        vmovdqu(ymm1, yword[A1 - 0x60]);
        vmovdqu(ymm2, yword[A1 - 0x40]);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm1);
        vmovdqu(yword[B - 0x40], ymm2);
        sub(B, -96);
        align(4);

        L(l2fc);
        sub(N, 0x30);
        cmp(N, 0x30);
        jge(l24, T_NEAR);
        align(4);

        L(l30c);
        cmp(N, 0x20);
        jl(l520, T_NEAR);
        align(4);

        L(l318);
        mov(A1, A);
        add(A, 0x40);
        mov(I, M);
        sar(I, 0x2);
        jle(l44c, T_NEAR);
        align(4);

        L(l32c);
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
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x70]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x70]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x20], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x60]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x60]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x40], ymm2);
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B + 0x60], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -256);
        dec(I);
        jg(l32c, T_NEAR);
        align(4);

        L(l44c);
        test(M, 0x2);
        jle(l4e8, T_NEAR);
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
        vmovdqu(xmm0, xword[A1 - 0x50]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x50]);
        vpunpcklwd(xmm2, xmm0, xmm1);
        vpunpckhwd(xmm3, xmm0, xmm1);
        vperm2f128(ymm2, ymm2, ymm3, 0x20);
        vmovdqu(yword[B - 0x20], ymm2);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -128);
        align(4);

        L(l4e8);
        test(M, 0x1);
        jle(l510, T_NEAR);
        vmovdqu(ymm0, yword[A1 - 0x80]);
        vmovdqu(ymm1, yword[A1 - 0x60]);
        add(A1, LDA);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm1);
        sub(B, -64);
        align(4);

        L(l510);
        sub(N, 0x20);
        cmp(N, 0x20);
        jge(l318, T_NEAR);
        align(4);

        L(l520);
        cmp(N, 0x10);
        jl(l668, T_NEAR);
        align(4);

        L(l52c);
        mov(A1, A);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x2);
        jle(l5dc, T_NEAR);
        align(4);

        L(l540);
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
        jg(l540, T_NEAR);
        align(4);

        L(l5dc);
        test(M, 0x2);
        jle(l630, T_NEAR);
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

        L(l630);
        test(M, 0x1);
        jle(l658, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 - 0x70]);
        vmovdqu(xword[B - 0x80], xmm0);
        vmovdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l658);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l52c, T_NEAR);
        align(4);

        L(l668);
        cmp(N, 0x8);
        jl(l7ec, T_NEAR);
        align(4);

        L(l674);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l730, T_NEAR);
        align(4);

        L(l688);
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
        jg(l688, T_NEAR);
        align(4);

        L(l730);
        test(M, 0x4);
        jle(l78c, T_NEAR);
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

        L(l78c);
        test(M, 0x2);
        jle(l7c0, T_NEAR);
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

        L(l7c0);
        test(M, 0x1);
        jle(l7dc, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l7dc);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l674, T_NEAR);
        align(4);

        L(l7ec);
        cmp(N, 0x4);
        jl(l924, T_NEAR);
        align(4);

        L(l7f8);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l884, T_NEAR);
        align(4);

        L(l808);
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
        jg(l808, T_NEAR);
        align(4);

        L(l884);
        test(M, 0x4);
        jle(l8cc, T_NEAR);
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

        L(l8cc);
        test(M, 0x2);
        jle(l8f8, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        vmovq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l8f8);
        test(M, 0x1);
        jle(l914, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l914);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l7f8, T_NEAR);
        align(4);

        L(l924);
        cmp(N, 0x2);
        jl(la52, T_NEAR);
        align(4);

        L(l930);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l9b8, T_NEAR);
        align(4);

        L(l940);
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
        jg(l940, T_NEAR);
        align(4);

        L(l9b8);
        test(M, 0x4);
        jle(l9fc, T_NEAR);
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

        L(l9fc);
        test(M, 0x2);
        jle(la28, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        vmovd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        vpunpcklwd(xmm0, xmm0, xmm1);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(la28);
        test(M, 0x1);
        jle(la44, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(la44);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l930, T_NEAR);
        align(4);

        L(la52);
        cmp(N, 0x1);
        jl(lb84, T_NEAR);
        align(4);

        L(la5c);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(lae4, T_NEAR);
        align(4);

        L(la6c);
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
        jg(la6c, T_NEAR);
        align(4);

        L(lae4);
        test(M, 0x4);
        jle(lb2c, T_NEAR);
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

        L(lb2c);
        test(M, 0x2);
        jle(lb5c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lb5c);
        test(M, 0x1);
        jle(lb74, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(lb74);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(la5c, T_NEAR);
        align(4);

        L(lb84);

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
