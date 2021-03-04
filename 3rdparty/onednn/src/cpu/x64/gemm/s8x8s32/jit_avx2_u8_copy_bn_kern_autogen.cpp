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

jit_avx2_u8_copy_bn_kern::jit_avx2_u8_copy_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_u8_copy_bn_kern::generate() {

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

        Xbyak::Label l140;
        Xbyak::Label l188;
        Xbyak::Label l1c8;
        Xbyak::Label l1d8;
        Xbyak::Label l1e4;
        Xbyak::Label l1fc;
        Xbyak::Label l20;
        Xbyak::Label l234;
        Xbyak::Label l260;
        Xbyak::Label l28c;
        Xbyak::Label l2bc;
        Xbyak::Label l2dc;
        Xbyak::Label l2ea;
        Xbyak::Label l2f4;
        Xbyak::Label l304;
        Xbyak::Label l320;
        Xbyak::Label l340;
        Xbyak::Label l360;
        Xbyak::Label l37c;
        Xbyak::Label l38;
        Xbyak::Label l394;
        Xbyak::Label l3a4;
        Xbyak::Label lac;
        Xbyak::Label lfc;

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
        sub(B, -128);
        lea(LDA3, ptr[LDA + LDA * 2]);
        cmp(N, 0x4);
        jl(l1d8, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(lac, T_NEAR);
        align(4);

        L(l38);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm2, xword[A2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm3);
        sub(B, -64);
        dec(I);
        jg(l38, T_NEAR);
        align(4);

        L(lac);
        test(M, 0x8);
        jle(lfc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        sub(A1, -8);
        movq(xmm2, qword[A2 - 0x80]);
        movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(lfc);
        test(M, 0x4);
        jle(l140, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        movd(xmm2, dword[A2 - 0x80]);
        movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l140);
        test(M, 0x2);
        jle(l188, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x3);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l188);
        test(M, 0x1);
        jle(l1c8, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l1c8);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l20, T_NEAR);
        align(4);

        L(l1d8);
        cmp(N, 0x2);
        jl(l2ea, T_NEAR);
        align(4);

        L(l1e4);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l234, T_NEAR);
        align(4);

        L(l1fc);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        movdqa(xmm2, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        dec(I);
        jg(l1fc, T_NEAR);
        align(4);

        L(l234);
        test(M, 0x8);
        jle(l260, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l260);
        test(M, 0x4);
        jle(l28c, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l28c);
        test(M, 0x2);
        jle(l2bc, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l2bc);
        test(M, 0x1);
        jle(l2dc, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(l2dc);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l1e4, T_NEAR);
        align(4);

        L(l2ea);
        cmp(N, 0x1);
        jl(l3a4, T_NEAR);
        align(4);

        L(l2f4);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(l320, T_NEAR);
        align(4);

        L(l304);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(l304, T_NEAR);
        align(4);

        L(l320);
        test(M, 0x8);
        jle(l340, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l340);
        test(M, 0x4);
        jle(l360, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l360);
        test(M, 0x2);
        jle(l37c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(l37c);
        test(M, 0x1);
        jle(l394, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(l394);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l2f4, T_NEAR);
        align(4);

        L(l3a4);

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
