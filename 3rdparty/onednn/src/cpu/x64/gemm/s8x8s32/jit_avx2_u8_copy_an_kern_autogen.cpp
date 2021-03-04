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

jit_avx2_u8_copy_an_kern::jit_avx2_u8_copy_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_u8_copy_an_kern::generate() {

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

        Xbyak::Label l10c;
        Xbyak::Label l118;
        Xbyak::Label l12c;
        Xbyak::Label l1c4;
        Xbyak::Label l20;
        Xbyak::Label l218;
        Xbyak::Label l244;
        Xbyak::Label l260;
        Xbyak::Label l270;
        Xbyak::Label l27c;
        Xbyak::Label l28c;
        Xbyak::Label l30;
        Xbyak::Label l304;
        Xbyak::Label l348;
        Xbyak::Label l374;
        Xbyak::Label l390;
        Xbyak::Label l3a0;
        Xbyak::Label l3ac;
        Xbyak::Label l3c0;
        Xbyak::Label l458;
        Xbyak::Label l4ac;
        Xbyak::Label l4e0;
        Xbyak::Label l4f8;
        Xbyak::Label l506;
        Xbyak::Label l510;
        Xbyak::Label l520;
        Xbyak::Label l598;
        Xbyak::Label l5e0;
        Xbyak::Label l604;
        Xbyak::Label l61c;
        Xbyak::Label l62c;
        Xbyak::Label la8;
        Xbyak::Label le0;
        Xbyak::Label lfc;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x10);
        jl(l10c, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x2);
        jle(la8, T_NEAR);
        align(4);

        L(l30);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm4, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm4, xmm1);
        movdqa(xmm1, xmm2);
        punpcklbw(xmm2, xmm3);
        punpckhbw(xmm1, xmm3);
        movdqa(xmm3, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm3, xmm2);
        movdqa(xmm2, xmm4);
        punpcklwd(xmm4, xmm1);
        punpckhwd(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm3);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm2);
        sub(B, -64);
        dec(I);
        jg(l30, T_NEAR);
        align(4);

        L(la8);
        test(M, 0x2);
        jle(le0, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm2, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        align(4);

        L(le0);
        test(M, 0x1);
        jle(lfc, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(lfc);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l20, T_NEAR);
        align(4);

        L(l10c);
        cmp(N, 0x8);
        jl(l270, T_NEAR);
        align(4);

        L(l118);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l1c4, T_NEAR);
        align(4);

        L(l12c);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        dec(I);
        jg(l12c, T_NEAR);
        align(4);

        L(l1c4);
        test(M, 0x4);
        jle(l218, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l218);
        test(M, 0x2);
        jle(l244, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l244);
        test(M, 0x1);
        jle(l260, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l260);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l118, T_NEAR);
        align(4);

        L(l270);
        cmp(N, 0x4);
        jl(l3a0, T_NEAR);
        align(4);

        L(l27c);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l304, T_NEAR);
        align(4);

        L(l28c);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(l28c, T_NEAR);
        align(4);

        L(l304);
        test(M, 0x4);
        jle(l348, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l348);
        test(M, 0x2);
        jle(l374, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l374);
        test(M, 0x1);
        jle(l390, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l390);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l27c, T_NEAR);
        align(4);

        L(l3a0);
        cmp(N, 0x2);
        jl(l506, T_NEAR);
        align(4);

        L(l3ac);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l458, T_NEAR);
        align(4);

        L(l3c0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm4, eax, 0x0);
        punpcklbw(xmm1, xmm2);
        punpcklbw(xmm3, xmm4);
        punpcklwd(xmm1, xmm3);
        punpcklqdq(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(l3c0, T_NEAR);
        align(4);

        L(l458);
        test(M, 0x4);
        jle(l4ac, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l4ac);
        test(M, 0x2);
        jle(l4e0, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l4e0);
        test(M, 0x1);
        jle(l4f8, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l4f8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l3ac, T_NEAR);
        align(4);

        L(l506);
        cmp(N, 0x1);
        jl(l62c, T_NEAR);
        align(4);

        L(l510);
        mov(A1, A);
        add(A, 0x1);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l598, T_NEAR);
        align(4);

        L(l520);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x7);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        dec(LDA3);
        jg(l520, T_NEAR);
        align(4);

        L(l598);
        test(M, 0x4);
        jle(l5e0, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l5e0);
        test(M, 0x2);
        jle(l604, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(l604);
        test(M, 0x1);
        jle(l61c, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(l61c);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l510, T_NEAR);
        align(4);

        L(l62c);

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
