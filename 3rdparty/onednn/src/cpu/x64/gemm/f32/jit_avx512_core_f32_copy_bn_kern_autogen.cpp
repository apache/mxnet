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

jit_avx512_core_f32_copy_bn_kern::jit_avx512_core_f32_copy_bn_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_f32_copy_bn_kern::generate() {

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
#define A1 rsi
#define A2 r10
#define LDA3 r11

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp

#endif

    inLocalLabel();
    {

        Xbyak::Label l114;
        Xbyak::Label l188;
        Xbyak::Label l1f8;
        Xbyak::Label l208;
        Xbyak::Label l228;
        Xbyak::Label l288;
        Xbyak::Label l2d4;
        Xbyak::Label l318;
        Xbyak::Label l31c;
        Xbyak::Label l33c;
        Xbyak::Label l37c;
        Xbyak::Label l3b0;
        Xbyak::Label l3e0;
        Xbyak::Label l3e4;
        Xbyak::Label l404;
        Xbyak::Label l444;
        Xbyak::Label l474;
        Xbyak::Label l498;
        Xbyak::Label l49c;
        Xbyak::Label l4a4;
        Xbyak::Label l4c4;
        Xbyak::Label l4e0;
        Xbyak::Label l58;
        Xbyak::Label l590;
        Xbyak::Label l60c;
        Xbyak::Label l680;
        Xbyak::Label l690;
        Xbyak::Label l6b0;
        Xbyak::Label l720;
        Xbyak::Label l74;
        Xbyak::Label l774;
        Xbyak::Label l7bc;
        Xbyak::Label l7c0;
        Xbyak::Label l7e0;
        Xbyak::Label l828;
        Xbyak::Label l860;
        Xbyak::Label l894;
        Xbyak::Label l898;
        Xbyak::Label l8b8;
        Xbyak::Label l8fc;
        Xbyak::Label l930;
        Xbyak::Label l958;
        Xbyak::Label l95c;
        Xbyak::Label l964;
        Xbyak::Label l970;
        Xbyak::Label l98c;
        Xbyak::Label la3c;
        Xbyak::Label lab8;
        Xbyak::Label lb2c;
        Xbyak::Label lb3c;
        Xbyak::Label lb5c;
        Xbyak::Label lbcc;
        Xbyak::Label lc20;
        Xbyak::Label lc68;
        Xbyak::Label lc6c;
        Xbyak::Label lc8c;
        Xbyak::Label lcd4;
        Xbyak::Label ld0c;
        Xbyak::Label ld40;
        Xbyak::Label ld44;
        Xbyak::Label ld64;
        Xbyak::Label lda8;
        Xbyak::Label lddc;
        Xbyak::Label le04;
        Xbyak::Label le08;

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
        vbroadcastss(zmm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vbroadcastss(zmm4, xmm4);
        vucomiss(xmm6, xmm3);
        jne(l4a4, T_NEAR);
        cmp(N, 0x8);
        jl(l208, T_NEAR);
        align(4);

        L(l58);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l114, T_NEAR);
        align(4);

        L(l74);
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
        jg(l74, T_NEAR);
        align(4);

        L(l114);
        test(M, 0x2);
        jle(l188, T_NEAR);
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

        L(l188);
        test(M, 0x1);
        jle(l1f8, T_NEAR);
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

        L(l1f8);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l58, T_NEAR);
        align(4);

        L(l208);
        cmp(N, 0x4);
        jl(l31c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l288, T_NEAR);
        align(4);

        L(l228);
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
        jg(l228, T_NEAR);
        align(4);

        L(l288);
        test(M, 0x2);
        jle(l2d4, T_NEAR);
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

        L(l2d4);
        test(M, 0x1);
        jle(l318, T_NEAR);
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

        L(l318);
        sub(N, 0x4);
        align(4);

        L(l31c);
        cmp(N, 0x2);
        jl(l3e4, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l37c, T_NEAR);
        align(4);

        L(l33c);
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
        jg(l33c, T_NEAR);
        align(4);

        L(l37c);
        test(M, 0x2);
        jle(l3b0, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(l3b0);
        test(M, 0x1);
        jle(l3e0, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l3e0);
        sub(N, 0x2);
        align(4);

        L(l3e4);
        cmp(N, 0x1);
        jl(l49c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l444, T_NEAR);
        align(4);

        L(l404);
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
        jg(l404, T_NEAR);
        align(4);

        L(l444);
        test(M, 0x2);
        jle(l474, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l474);
        test(M, 0x1);
        jle(l498, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l498);
        sub(N, 0x1);
        align(4);

        L(l49c);
        jmp(le08, T_NEAR);
        align(4);

        L(l4a4);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(l964, T_NEAR);
        vmovaps(zmm6, zmm4);
        cmp(N, 0x8);
        jl(l690, T_NEAR);
        align(4);

        L(l4c4);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l590, T_NEAR);
        align(4);

        L(l4e0);
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
        jg(l4e0, T_NEAR);
        align(4);

        L(l590);
        test(M, 0x2);
        jle(l60c, T_NEAR);
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

        L(l60c);
        test(M, 0x1);
        jle(l680, T_NEAR);
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

        L(l680);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l4c4, T_NEAR);
        align(4);

        L(l690);
        cmp(N, 0x4);
        jl(l7c0, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l720, T_NEAR);
        align(4);

        L(l6b0);
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
        jg(l6b0, T_NEAR);
        align(4);

        L(l720);
        test(M, 0x2);
        jle(l774, T_NEAR);
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

        L(l774);
        test(M, 0x1);
        jle(l7bc, T_NEAR);
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

        L(l7bc);
        sub(N, 0x4);
        align(4);

        L(l7c0);
        cmp(N, 0x2);
        jl(l898, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l828, T_NEAR);
        align(4);

        L(l7e0);
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
        jg(l7e0, T_NEAR);
        align(4);

        L(l828);
        test(M, 0x2);
        jle(l860, T_NEAR);
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

        L(l860);
        test(M, 0x1);
        jle(l894, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(l894);
        sub(N, 0x2);
        align(4);

        L(l898);
        cmp(N, 0x1);
        jl(l95c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(l8fc, T_NEAR);
        align(4);

        L(l8b8);
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
        jg(l8b8, T_NEAR);
        align(4);

        L(l8fc);
        test(M, 0x2);
        jle(l930, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(l930);
        test(M, 0x1);
        jle(l958, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(l958);
        sub(N, 0x1);
        align(4);

        L(l95c);
        jmp(le08, T_NEAR);
        align(4);

        L(l964);
        cmp(N, 0x8);
        jl(lb3c, T_NEAR);
        align(4);

        L(l970);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(la3c, T_NEAR);
        align(4);

        L(l98c);
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
        jg(l98c, T_NEAR);
        align(4);

        L(la3c);
        test(M, 0x2);
        jle(lab8, T_NEAR);
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

        L(lab8);
        test(M, 0x1);
        jle(lb2c, T_NEAR);
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

        L(lb2c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l970, T_NEAR);
        align(4);

        L(lb3c);
        cmp(N, 0x4);
        jl(lc6c, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lbcc, T_NEAR);
        align(4);

        L(lb5c);
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
        jg(lb5c, T_NEAR);
        align(4);

        L(lbcc);
        test(M, 0x2);
        jle(lc20, T_NEAR);
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

        L(lc20);
        test(M, 0x1);
        jle(lc68, T_NEAR);
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

        L(lc68);
        sub(N, 0x4);
        align(4);

        L(lc6c);
        cmp(N, 0x2);
        jl(ld44, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lcd4, T_NEAR);
        align(4);

        L(lc8c);
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
        jg(lc8c, T_NEAR);
        align(4);

        L(lcd4);
        test(M, 0x2);
        jle(ld0c, T_NEAR);
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

        L(ld0c);
        test(M, 0x1);
        jle(ld40, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(ld40);
        sub(N, 0x2);
        align(4);

        L(ld44);
        cmp(N, 0x1);
        jl(le08, T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(lda8, T_NEAR);
        align(4);

        L(ld64);
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
        jg(ld64, T_NEAR);
        align(4);

        L(lda8);
        test(M, 0x2);
        jle(lddc, T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(lddc);
        test(M, 0x1);
        jle(le04, T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(le04);
        sub(N, 0x1);
        align(4);

        L(le08);

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
