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

jit_avx512_core_s16_24x8_copy_at_kern::jit_avx512_core_s16_24x8_copy_at_kern()
    : jit_generator(nullptr, S16_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_s16_24x8_copy_at_kern::generate() {

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

        Xbyak::Label l208;
        Xbyak::Label l24;
        Xbyak::Label l390;
        Xbyak::Label l40;
        Xbyak::Label l4cc;
        Xbyak::Label l60c;
        Xbyak::Label l61c;
        Xbyak::Label l628;
        Xbyak::Label l644;
        Xbyak::Label l770;
        Xbyak::Label l87c;
        Xbyak::Label l958;
        Xbyak::Label la34;
        Xbyak::Label la44;
        Xbyak::Label la50;
        Xbyak::Label la6c;
        Xbyak::Label lb08;
        Xbyak::Label lb84;
        Xbyak::Label lbf4;
        Xbyak::Label lc68;
        Xbyak::Label lc78;
        Xbyak::Label lc84;
        Xbyak::Label lc9c;
        Xbyak::Label ld0c;
        Xbyak::Label ld58;
        Xbyak::Label ld9c;
        Xbyak::Label lde0;
        Xbyak::Label ldf0;
        Xbyak::Label ldfc;
        Xbyak::Label le14;
        Xbyak::Label le48;
        Xbyak::Label le74;
        Xbyak::Label lea0;
        Xbyak::Label lec8;
        Xbyak::Label led6;
        Xbyak::Label lee0;
        Xbyak::Label lef0;
        Xbyak::Label lf0c;
        Xbyak::Label lf2c;
        Xbyak::Label lf4c;
        Xbyak::Label lf6c;
        Xbyak::Label lf84;
        Xbyak::Label lf94;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        shl(LDA, 1);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x18);
        jl(l61c, T_NEAR);
        align(4);

        L(l24);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x18);
        add(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(l208, T_NEAR);
        align(4);

        L(l40);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vmovdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        vmovdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovdqu(xmm4, xword[A2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm5, 0x20);
        vmovdqu(xmm4, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm3, ymm3, ymm5, 0x20);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm2);
        vunpckhps(ymm5, ymm0, ymm2);
        vunpcklps(ymm0, ymm1, ymm3);
        vunpckhps(ymm1, ymm1, ymm3);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(yword[B - 0x20], ymm3);
        vmovdqu(yword[B + 0x40], ymm4);
        vmovdqu(yword[B + 0xa0], ymm5);
        vmovdqu(xmm0, xword[A2 - 0x80]);
        vmovdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        vmovdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        vmovdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xmm4, xword[A2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm5, 0x20);
        vmovdqu(xmm4, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm3, ymm3, ymm5, 0x20);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm2);
        vunpckhps(ymm5, ymm0, ymm2);
        vunpcklps(ymm0, ymm1, ymm3);
        vunpckhps(ymm1, ymm1, ymm3);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(yword[B], ymm3);
        vmovdqu(yword[B + 0x60], ymm4);
        vmovdqu(yword[B + 0xc0], ymm5);
        vmovdqu(xmm0, xword[A2 - 0x80]);
        vmovdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        vmovdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        vmovdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xmm4, xword[A2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm5, 0x20);
        vmovdqu(xmm4, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm3, ymm3, ymm5, 0x20);
        vunpcklps(ymm4, ymm0, ymm2);
        vunpckhps(ymm5, ymm0, ymm2);
        vunpcklps(ymm0, ymm1, ymm3);
        vunpckhps(ymm1, ymm1, ymm3);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vmovdqu(yword[B - 0x40], ymm2);
        vmovdqu(yword[B + 0x20], ymm3);
        vmovdqu(yword[B + 0x80], ymm4);
        vmovdqu(yword[B + 0xe0], ymm5);
        sub(A1, -16);
        sub(B, -384);
        dec(I);
        jg(l40, T_NEAR);
        align(4);

        L(l208);
        test(M, 0x4);
        jle(l390, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm2, xmm2, xmm4);
        vpunpcklqdq(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vpermilps(ymm0, ymm0, 0xd8);
        vpermilps(ymm1, ymm1, 0xd8);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(yword[B - 0x20], ymm3);
        vmovq(xmm0, qword[A2 - 0x80]);
        vmovq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm2, xmm2, xmm4);
        vpunpcklqdq(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vpermilps(ymm0, ymm0, 0xd8);
        vpermilps(ymm1, ymm1, 0xd8);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(yword[B], ymm3);
        vmovq(xmm0, qword[A2 - 0x80]);
        vmovq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm2, xmm2, xmm4);
        vpunpcklqdq(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vpermilps(ymm0, ymm0, 0xd8);
        vpermilps(ymm1, ymm1, 0xd8);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x40], ymm2);
        vmovdqu(yword[B + 0x20], ymm3);
        sub(A1, -8);
        sub(B, -192);
        align(4);

        L(l390);
        test(M, 0x2);
        jle(l4cc, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vperm2f128(ymm0, ymm0, ymm1, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovd(xmm0, dword[A2 - 0x80]);
        vmovd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vperm2f128(ymm0, ymm0, ymm1, 0x20);
        vmovdqu(yword[B - 0x60], ymm0);
        vmovd(xmm0, dword[A2 - 0x80]);
        vmovd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vperm2f128(ymm0, ymm0, ymm1, 0x20);
        vmovdqu(yword[B - 0x40], ymm0);
        sub(A1, -4);
        sub(B, -96);
        align(4);

        L(l4cc);
        test(M, 0x1);
        jle(l60c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A1 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A2 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x70], xmm0);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A2 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x60], xmm0);
        sub(B, -48);
        align(4);

        L(l60c);
        sub(N, 0x18);
        cmp(N, 0x18);
        jge(l24, T_NEAR);
        align(4);

        L(l61c);
        cmp(N, 0x10);
        jl(la44, T_NEAR);
        align(4);

        L(l628);
        mov(A1, A);
        mov(I, LDA);
        shl(I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(l770, T_NEAR);
        align(4);

        L(l644);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        vmovdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        vmovdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovdqu(xmm4, xword[A2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm5, 0x20);
        vmovdqu(xmm4, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm3, ymm3, ymm5, 0x20);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm2);
        vunpckhps(ymm5, ymm0, ymm2);
        vunpcklps(ymm0, ymm1, ymm3);
        vunpckhps(ymm1, ymm1, ymm3);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(yword[B - 0x40], ymm3);
        vmovdqu(yword[B], ymm4);
        vmovdqu(yword[B + 0x40], ymm5);
        vmovdqu(xmm0, xword[A2 - 0x80]);
        vmovdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        vmovdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        vmovdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xmm4, xword[A2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm5, 0x20);
        vmovdqu(xmm4, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        vmovdqu(xmm5, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm3, ymm3, ymm5, 0x20);
        vunpcklps(ymm4, ymm0, ymm2);
        vunpckhps(ymm5, ymm0, ymm2);
        vunpcklps(ymm0, ymm1, ymm3);
        vunpckhps(ymm1, ymm1, ymm3);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(yword[B - 0x20], ymm3);
        vmovdqu(yword[B + 0x20], ymm4);
        vmovdqu(yword[B + 0x60], ymm5);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(l644, T_NEAR);
        align(4);

        L(l770);
        test(M, 0x4);
        jle(l87c, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm2, xmm2, xmm4);
        vpunpcklqdq(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vpermilps(ymm0, ymm0, 0xd8);
        vpermilps(ymm1, ymm1, 0xd8);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(yword[B - 0x40], ymm3);
        vmovq(xmm0, qword[A2 - 0x80]);
        vmovq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vpunpcklqdq(xmm2, xmm2, xmm4);
        vpunpcklqdq(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vpermilps(ymm0, ymm0, 0xd8);
        vpermilps(ymm1, ymm1, 0xd8);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x60], ymm2);
        vmovdqu(yword[B - 0x20], ymm3);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(l87c);
        test(M, 0x2);
        jle(l958, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vperm2f128(ymm0, ymm0, ymm1, 0x20);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovd(xmm0, dword[A2 - 0x80]);
        vmovd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vperm2f128(ymm0, ymm0, ymm1, 0x20);
        vmovdqu(yword[B - 0x60], ymm0);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(l958);
        test(M, 0x1);
        jle(la34, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A1 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A2 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(la34);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l628, T_NEAR);
        align(4);

        L(la44);
        cmp(N, 0x8);
        jl(lc78, T_NEAR);
        align(4);

        L(la50);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        lea(I, ptr[A1 + LDA * 8]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(lb08, T_NEAR);
        align(4);

        L(la6c);
        vmovdqu(xmm4, xword[A1 - 0x80]);
        vmovdqu(xmm5, xword[A1 + LDA * 1 - 0x80]);
        vmovdqu(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -16);
        vmovdqu(xmm2, xword[A2 - 0x80]);
        vperm2f128(ymm4, ymm4, ymm2, 0x20);
        vmovdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        vperm2f128(ymm5, ymm5, ymm3, 0x20);
        vmovdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vmovdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        sub(A2, -16);
        vunpcklps(ymm2, ymm4, ymm0);
        vunpckhps(ymm3, ymm4, ymm0);
        vunpcklps(ymm4, ymm5, ymm1);
        vunpckhps(ymm5, ymm5, ymm1);
        vunpcklps(ymm0, ymm2, ymm4);
        vunpckhps(ymm1, ymm2, ymm4);
        vunpcklps(ymm2, ymm3, ymm5);
        vunpckhps(ymm3, ymm3, ymm5);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm1);
        vmovdqu(yword[B - 0x40], ymm2);
        vmovdqu(yword[B - 0x20], ymm3);
        sub(B, -128);
        dec(I);
        jg(la6c, T_NEAR);
        align(4);

        L(lb08);
        test(M, 0x4);
        jle(lb84, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        vmovq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        vmovq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -8);
        vunpcklps(xmm0, xmm0, xmm2);
        vunpcklps(xmm1, xmm1, xmm3);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        vmovq(xmm4, qword[A2 + LDA * 2 - 0x80]);
        vmovq(xmm5, qword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -8);
        vunpcklps(xmm2, xmm2, xmm4);
        vunpcklps(xmm3, xmm3, xmm5);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vmovdqu(yword[B - 0x80], ymm2);
        vmovdqu(yword[B - 0x60], ymm3);
        sub(B, -64);
        align(4);

        L(lb84);
        test(M, 0x2);
        jle(lbf4, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        vmovd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        vmovd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -4);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovd(xmm1, dword[A2 - 0x80]);
        vmovd(xmm2, dword[A2 + LDA * 1 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 2 - 0x80]);
        vmovd(xmm4, dword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -4);
        vunpcklps(xmm1, xmm1, xmm2);
        vunpcklps(xmm3, xmm3, xmm4);
        vpunpcklqdq(xmm1, xmm1, xmm3);
        vinsertf128(ymm0, ymm0, xmm1, 0x1);
        vmovdqu(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(lbf4);
        test(M, 0x1);
        jle(lc68, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A1 + LDA * 4]);
        mov(ax, word[A2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(lc68);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(la50, T_NEAR);
        align(4);

        L(lc78);
        cmp(N, 0x4);
        jl(ldf0, T_NEAR);
        align(4);

        L(lc84);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(ld0c, T_NEAR);
        align(4);

        L(lc9c);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        vmovdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        sub(A1, -16);
        vmovdqu(xmm2, xword[A2 - 0x80]);
        vmovdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        sub(A2, -16);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm2, ymm0, ymm1);
        vunpckhps(ymm3, ymm0, ymm1);
        vperm2f128(ymm0, ymm2, ymm2, 0x1);
        vperm2f128(ymm1, ymm3, ymm3, 0x1);
        vshufpd(ymm0, ymm2, ymm0, 0xc);
        vshufpd(ymm1, ymm3, ymm1, 0xc);
        vpermilpd(ymm0, ymm0, 0x6);
        vpermilpd(ymm1, ymm1, 0x6);
        vmovdqu(yword[B - 0x80], ymm0);
        vmovdqu(yword[B - 0x60], ymm1);
        sub(B, -64);
        dec(I);
        jg(lc9c, T_NEAR);
        align(4);

        L(ld0c);
        test(M, 0x4);
        jle(ld58, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        vmovq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        sub(A1, -8);
        vmovq(xmm2, qword[A2 - 0x80]);
        vmovq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        sub(A2, -8);
        vunpcklps(xmm0, xmm0, xmm2);
        vunpcklps(xmm1, xmm1, xmm3);
        vunpcklps(xmm2, xmm0, xmm1);
        vunpckhps(xmm3, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm2);
        vmovdqu(xword[B - 0x70], xmm3);
        sub(B, -32);
        align(4);

        L(ld58);
        test(M, 0x2);
        jle(ld9c, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        vmovd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        vmovd(xmm2, dword[A2 - 0x80]);
        vmovd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        vunpcklps(xmm0, xmm0, xmm1);
        vunpcklps(xmm2, xmm2, xmm3);
        vpunpcklqdq(xmm0, xmm0, xmm2);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(ld9c);
        test(M, 0x1);
        jle(lde0, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x3);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lde0);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(lc84, T_NEAR);
        align(4);

        L(ldf0);
        cmp(N, 0x2);
        jl(led6, T_NEAR);
        align(4);

        L(ldfc);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(le48, T_NEAR);
        align(4);

        L(le14);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        vmovdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        vunpcklps(xmm2, xmm0, xmm1);
        vunpckhps(xmm3, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm2);
        vmovdqu(xword[B - 0x70], xmm3);
        sub(B, -32);
        dec(I);
        jg(le14, T_NEAR);
        align(4);

        L(le48);
        test(M, 0x4);
        jle(le74, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        vmovq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(le74);
        test(M, 0x2);
        jle(lea0, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        vmovd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lea0);
        test(M, 0x1);
        jle(lec8, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lec8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(ldfc, T_NEAR);
        align(4);

        L(led6);
        cmp(N, 0x1);
        jl(lf94, T_NEAR);
        align(4);

        L(lee0);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(lf0c, T_NEAR);
        align(4);

        L(lef0);
        vmovdqu(ymm0, yword[A1 - 0x80]);
        sub(A1, -32);
        vmovdqu(yword[B - 0x80], ymm0);
        sub(B, -32);
        dec(I);
        jg(lef0, T_NEAR);
        align(4);

        L(lf0c);
        test(M, 0x8);
        jle(lf2c, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(lf2c);
        test(M, 0x4);
        jle(lf4c, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lf4c);
        test(M, 0x2);
        jle(lf6c, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lf6c);
        test(M, 0x1);
        jle(lf84, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(lf84);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(lee0, T_NEAR);
        align(4);

        L(lf94);
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
