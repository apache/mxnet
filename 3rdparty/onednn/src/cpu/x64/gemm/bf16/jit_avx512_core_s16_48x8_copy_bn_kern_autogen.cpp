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

jit_avx512_core_s16_48x8_copy_bn_kern::jit_avx512_core_s16_48x8_copy_bn_kern()
    : jit_generator(nullptr, S16_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_s16_48x8_copy_bn_kern::generate() {

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

        Xbyak::Label l158;
        Xbyak::Label l1c8;
        Xbyak::Label l23c;
        Xbyak::Label l24;
        Xbyak::Label l24c;
        Xbyak::Label l258;
        Xbyak::Label l270;
        Xbyak::Label l2e0;
        Xbyak::Label l32c;
        Xbyak::Label l370;
        Xbyak::Label l3b4;
        Xbyak::Label l3c4;
        Xbyak::Label l3d0;
        Xbyak::Label l3e8;
        Xbyak::Label l40;
        Xbyak::Label l41c;
        Xbyak::Label l448;
        Xbyak::Label l474;
        Xbyak::Label l49c;
        Xbyak::Label l4aa;
        Xbyak::Label l4b4;
        Xbyak::Label l4c4;
        Xbyak::Label l4e0;
        Xbyak::Label l500;
        Xbyak::Label l520;
        Xbyak::Label l540;
        Xbyak::Label l558;
        Xbyak::Label l568;
        Xbyak::Label ldc;

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
        cmp(N, 0x8);
        jl(l24c, T_NEAR);
        align(4);

        L(l24);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        lea(I, ptr[A1 + LDA * 8]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(ldc, T_NEAR);
        align(4);

        L(l40);
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
        jg(l40, T_NEAR);
        align(4);

        L(ldc);
        test(M, 0x4);
        jle(l158, T_NEAR);
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

        L(l158);
        test(M, 0x2);
        jle(l1c8, T_NEAR);
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

        L(l1c8);
        test(M, 0x1);
        jle(l23c, T_NEAR);
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

        L(l23c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l24, T_NEAR);
        align(4);

        L(l24c);
        cmp(N, 0x4);
        jl(l3c4, T_NEAR);
        align(4);

        L(l258);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(l2e0, T_NEAR);
        align(4);

        L(l270);
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
        jg(l270, T_NEAR);
        align(4);

        L(l2e0);
        test(M, 0x4);
        jle(l32c, T_NEAR);
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

        L(l32c);
        test(M, 0x2);
        jle(l370, T_NEAR);
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

        L(l370);
        test(M, 0x1);
        jle(l3b4, T_NEAR);
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

        L(l3b4);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l258, T_NEAR);
        align(4);

        L(l3c4);
        cmp(N, 0x2);
        jl(l4aa, T_NEAR);
        align(4);

        L(l3d0);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x3);
        jle(l41c, T_NEAR);
        align(4);

        L(l3e8);
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
        jg(l3e8, T_NEAR);
        align(4);

        L(l41c);
        test(M, 0x4);
        jle(l448, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        vmovq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l448);
        test(M, 0x2);
        jle(l474, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        vmovd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l474);
        test(M, 0x1);
        jle(l49c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        vpinsrw(xmm0, xmm0, eax, 0x1);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l49c);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l3d0, T_NEAR);
        align(4);

        L(l4aa);
        cmp(N, 0x1);
        jl(l568, T_NEAR);
        align(4);

        L(l4b4);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(l4e0, T_NEAR);
        align(4);

        L(l4c4);
        vmovdqu(ymm0, yword[A1 - 0x80]);
        sub(A1, -32);
        vmovdqu(yword[B - 0x80], ymm0);
        sub(B, -32);
        dec(I);
        jg(l4c4, T_NEAR);
        align(4);

        L(l4e0);
        test(M, 0x8);
        jle(l500, T_NEAR);
        vmovdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        vmovdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l500);
        test(M, 0x4);
        jle(l520, T_NEAR);
        vmovq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        vmovq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l520);
        test(M, 0x2);
        jle(l540, T_NEAR);
        vmovd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        vmovd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l540);
        test(M, 0x1);
        jle(l558, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l558);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l4b4, T_NEAR);
        align(4);

        L(l568);

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
