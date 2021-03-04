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

jit_avx_f32_copy_at_kern::jit_avx_f32_copy_at_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx_f32_copy_at_kern::generate() {

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

        Xbyak::Label l10fc;
        Xbyak::Label l11e0;
        Xbyak::Label l12b4;
        Xbyak::Label l12c4;
        Xbyak::Label l12e8;
        Xbyak::Label l1398;
        Xbyak::Label l1414;
        Xbyak::Label l1488;
        Xbyak::Label l148c;
        Xbyak::Label l14ac;
        Xbyak::Label l151c;
        Xbyak::Label l1570;
        Xbyak::Label l15b8;
        Xbyak::Label l15bc;
        Xbyak::Label l15dc;
        Xbyak::Label l1624;
        Xbyak::Label l165c;
        Xbyak::Label l1690;
        Xbyak::Label l1694;
        Xbyak::Label l16b4;
        Xbyak::Label l16f8;
        Xbyak::Label l172c;
        Xbyak::Label l1754;
        Xbyak::Label l1758;
        Xbyak::Label l1a0;
        Xbyak::Label l274;
        Xbyak::Label l340;
        Xbyak::Label l350;
        Xbyak::Label l374;
        Xbyak::Label l414;
        Xbyak::Label l488;
        Xbyak::Label l4f8;
        Xbyak::Label l4fc;
        Xbyak::Label l51c;
        Xbyak::Label l54;
        Xbyak::Label l57c;
        Xbyak::Label l5c8;
        Xbyak::Label l60c;
        Xbyak::Label l610;
        Xbyak::Label l630;
        Xbyak::Label l670;
        Xbyak::Label l6a4;
        Xbyak::Label l6d4;
        Xbyak::Label l6d8;
        Xbyak::Label l6f8;
        Xbyak::Label l70;
        Xbyak::Label l738;
        Xbyak::Label l768;
        Xbyak::Label l78c;
        Xbyak::Label l790;
        Xbyak::Label l798;
        Xbyak::Label l7b4;
        Xbyak::Label l7d0;
        Xbyak::Label l920;
        Xbyak::Label la04;
        Xbyak::Label lad8;
        Xbyak::Label lae8;
        Xbyak::Label lb0c;
        Xbyak::Label lbbc;
        Xbyak::Label lc38;
        Xbyak::Label lcac;
        Xbyak::Label lcb0;
        Xbyak::Label lcd0;
        Xbyak::Label ld40;
        Xbyak::Label ld94;
        Xbyak::Label lddc;
        Xbyak::Label lde0;
        Xbyak::Label le00;
        Xbyak::Label le48;
        Xbyak::Label le80;
        Xbyak::Label leb4;
        Xbyak::Label leb8;
        Xbyak::Label led8;
        Xbyak::Label lf1c;
        Xbyak::Label lf50;
        Xbyak::Label lf78;
        Xbyak::Label lf7c;
        Xbyak::Label lf84;
        Xbyak::Label lf90;
        Xbyak::Label lfac;

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
        vbroadcastss(ymm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vperm2f128(ymm4, ymm4, ymm4, 0x20);
        vucomiss(xmm6, xmm3);
        jne(l798, T_NEAR);
        cmp(N, 0x10);
        jl(l350, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l1a0, T_NEAR);
        align(4);

        L(l70);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(l70, T_NEAR);
        align(4);

        L(l1a0);
        test(M, 0x2);
        jle(l274, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(l274);
        test(M, 0x1);
        jle(l340, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(l340);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l54, T_NEAR);
        align(4);

        L(l350);
        cmp(N, 0x8);
        jl(l4fc, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l414, T_NEAR);
        align(4);

        L(l374);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(l374, T_NEAR);
        align(4);

        L(l414);
        test(M, 0x2);
        jle(l488, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(l488);
        test(M, 0x1);
        jle(l4f8, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(l4f8);
        sub(N, 0x8);
        align(4);

        L(l4fc);
        cmp(N, 0x4);
        jl(l610, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l57c, T_NEAR);
        align(4);

        L(l51c);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(l51c, T_NEAR);
        align(4);

        L(l57c);
        test(M, 0x2);
        jle(l5c8, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(l5c8);
        test(M, 0x1);
        jle(l60c, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(l60c);
        sub(N, 0x4);
        align(4);

        L(l610);
        cmp(N, 0x2);
        jl(l6d8, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l670, T_NEAR);
        align(4);

        L(l630);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(l630, T_NEAR);
        align(4);

        L(l670);
        test(M, 0x2);
        jle(l6a4, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l6a4);
        test(M, 0x1);
        jle(l6d4, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l6d4);
        sub(N, 0x2);
        align(4);

        L(l6d8);
        cmp(N, 0x1);
        jl(l790, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l738, T_NEAR);
        align(4);

        L(l6f8);
        vmovups(xmm0, xword[A1]);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(l6f8, T_NEAR);
        align(4);

        L(l738);
        test(M, 0x2);
        jle(l768, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l768);
        test(M, 0x1);
        jle(l78c, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l78c);
        sub(N, 0x1);
        align(4);

        L(l790);
        jmp(l1758, T_NEAR);
        align(4);

        L(l798);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(lf84, T_NEAR);
        vmovaps(ymm6, ymm4);
        cmp(N, 0x10);
        jl(lae8, T_NEAR);
        align(4);

        L(l7b4);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l920, T_NEAR);
        align(4);

        L(l7d0);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(l7d0, T_NEAR);
        align(4);

        L(l920);
        test(M, 0x2);
        jle(la04, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(la04);
        test(M, 0x1);
        jle(lad8, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(lad8);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l7b4, T_NEAR);
        align(4);

        L(lae8);
        cmp(N, 0x8);
        jl(lcb0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lbbc, T_NEAR);
        align(4);

        L(lb0c);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(lb0c, T_NEAR);
        align(4);

        L(lbbc);
        test(M, 0x2);
        jle(lc38, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(lc38);
        test(M, 0x1);
        jle(lcac, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(lcac);
        sub(N, 0x8);
        align(4);

        L(lcb0);
        cmp(N, 0x4);
        jl(lde0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(ld40, T_NEAR);
        align(4);

        L(lcd0);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vxorps(xmm2, xmm6, xmm2);
        vxorps(xmm3, xmm6, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(lcd0, T_NEAR);
        align(4);

        L(ld40);
        test(M, 0x2);
        jle(ld94, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(ld94);
        test(M, 0x1);
        jle(lddc, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(lddc);
        sub(N, 0x4);
        align(4);

        L(lde0);
        cmp(N, 0x2);
        jl(leb8, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(le48, T_NEAR);
        align(4);

        L(le00);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(le00, T_NEAR);
        align(4);

        L(le48);
        test(M, 0x2);
        jle(le80, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(le80);
        test(M, 0x1);
        jle(leb4, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(leb4);
        sub(N, 0x2);
        align(4);

        L(leb8);
        cmp(N, 0x1);
        jl(lf7c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lf1c, T_NEAR);
        align(4);

        L(led8);
        vmovups(xmm0, xword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(led8, T_NEAR);
        align(4);

        L(lf1c);
        test(M, 0x2);
        jle(lf50, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(lf50);
        test(M, 0x1);
        jle(lf78, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(lf78);
        sub(N, 0x1);
        align(4);

        L(lf7c);
        jmp(l1758, T_NEAR);
        align(4);

        L(lf84);
        cmp(N, 0x10);
        jl(l12c4, T_NEAR);
        align(4);

        L(lf90);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l10fc, T_NEAR);
        align(4);

        L(lfac);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(lfac, T_NEAR);
        align(4);

        L(l10fc);
        test(M, 0x2);
        jle(l11e0, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(l11e0);
        test(M, 0x1);
        jle(l12b4, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(l12b4);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(lf90, T_NEAR);
        align(4);

        L(l12c4);
        cmp(N, 0x8);
        jl(l148c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l1398, T_NEAR);
        align(4);

        L(l12e8);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(l12e8, T_NEAR);
        align(4);

        L(l1398);
        test(M, 0x2);
        jle(l1414, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(l1414);
        test(M, 0x1);
        jle(l1488, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(l1488);
        sub(N, 0x8);
        align(4);

        L(l148c);
        cmp(N, 0x4);
        jl(l15bc, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l151c, T_NEAR);
        align(4);

        L(l14ac);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmulps(xmm2, xmm6, xmm2);
        vmulps(xmm3, xmm6, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(l14ac, T_NEAR);
        align(4);

        L(l151c);
        test(M, 0x2);
        jle(l1570, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(l1570);
        test(M, 0x1);
        jle(l15b8, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(l15b8);
        sub(N, 0x4);
        align(4);

        L(l15bc);
        cmp(N, 0x2);
        jl(l1694, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l1624, T_NEAR);
        align(4);

        L(l15dc);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(l15dc, T_NEAR);
        align(4);

        L(l1624);
        test(M, 0x2);
        jle(l165c, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l165c);
        test(M, 0x1);
        jle(l1690, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l1690);
        sub(N, 0x2);
        align(4);

        L(l1694);
        cmp(N, 0x1);
        jl(l1758, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l16f8, T_NEAR);
        align(4);

        L(l16b4);
        vmovups(xmm0, xword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(l16b4, T_NEAR);
        align(4);

        L(l16f8);
        test(M, 0x2);
        jle(l172c, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l172c);
        test(M, 0x1);
        jle(l1754, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l1754);
        sub(N, 0x1);
        align(4);

        L(l1758);

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
