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

jit_avx_f32_copy_an_kern::jit_avx_f32_copy_an_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx_f32_copy_an_kern::generate() {

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

        Xbyak::Label l10cc;
        Xbyak::Label l116c;
        Xbyak::Label l11c0;
        Xbyak::Label l11f0;
        Xbyak::Label l1200;
        Xbyak::Label l1224;
        Xbyak::Label l12c4;
        Xbyak::Label l1318;
        Xbyak::Label l134c;
        Xbyak::Label l136c;
        Xbyak::Label l1370;
        Xbyak::Label l1394;
        Xbyak::Label l1430;
        Xbyak::Label l1484;
        Xbyak::Label l14b8;
        Xbyak::Label l14d8;
        Xbyak::Label l14dc;
        Xbyak::Label l1500;
        Xbyak::Label l159c;
        Xbyak::Label l15f0;
        Xbyak::Label l1624;
        Xbyak::Label l1644;
        Xbyak::Label l1648;
        Xbyak::Label l166c;
        Xbyak::Label l168;
        Xbyak::Label l1708;
        Xbyak::Label l175c;
        Xbyak::Label l1790;
        Xbyak::Label l17b0;
        Xbyak::Label l17b4;
        Xbyak::Label l1e4;
        Xbyak::Label l228;
        Xbyak::Label l250;
        Xbyak::Label l260;
        Xbyak::Label l280;
        Xbyak::Label l2fc;
        Xbyak::Label l340;
        Xbyak::Label l36c;
        Xbyak::Label l388;
        Xbyak::Label l38c;
        Xbyak::Label l3ac;
        Xbyak::Label l424;
        Xbyak::Label l468;
        Xbyak::Label l494;
        Xbyak::Label l4b0;
        Xbyak::Label l4b4;
        Xbyak::Label l4d4;
        Xbyak::Label l54;
        Xbyak::Label l54c;
        Xbyak::Label l590;
        Xbyak::Label l5bc;
        Xbyak::Label l5d8;
        Xbyak::Label l5dc;
        Xbyak::Label l5fc;
        Xbyak::Label l674;
        Xbyak::Label l6b8;
        Xbyak::Label l6c;
        Xbyak::Label l6e4;
        Xbyak::Label l700;
        Xbyak::Label l704;
        Xbyak::Label l70c;
        Xbyak::Label l728;
        Xbyak::Label l740;
        Xbyak::Label l87c;
        Xbyak::Label l91c;
        Xbyak::Label l970;
        Xbyak::Label l9a0;
        Xbyak::Label l9b0;
        Xbyak::Label l9d4;
        Xbyak::Label la74;
        Xbyak::Label lac8;
        Xbyak::Label lafc;
        Xbyak::Label lb1c;
        Xbyak::Label lb20;
        Xbyak::Label lb44;
        Xbyak::Label lbe0;
        Xbyak::Label lc34;
        Xbyak::Label lc68;
        Xbyak::Label lc88;
        Xbyak::Label lc8c;
        Xbyak::Label lcb0;
        Xbyak::Label ld4c;
        Xbyak::Label lda0;
        Xbyak::Label ldd4;
        Xbyak::Label ldf4;
        Xbyak::Label ldf8;
        Xbyak::Label le1c;
        Xbyak::Label leb8;
        Xbyak::Label lf0c;
        Xbyak::Label lf40;
        Xbyak::Label lf60;
        Xbyak::Label lf64;
        Xbyak::Label lf6c;
        Xbyak::Label lf78;
        Xbyak::Label lf90;

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
        vbroadcastss(ymm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vperm2f128(ymm4, ymm4, ymm4, 0x20);
        vucomiss(xmm6, xmm3);
        jne(l70c, T_NEAR);
        cmp(N, 0x10);
        jl(l260, T_NEAR);
        align(4);

        L(l54);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x40);
        mov(I, M);
        sar(I, 0x3);
        jle(l168, T_NEAR);
        align(4);

        L(l6c);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vmovups(yword[B + 0x60], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmovups(yword[B + 0x80], ymm0);
        vmovups(ymm0, yword[A2 - 0x60]);
        vmovups(yword[B + 0xa0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmovups(yword[B + 0xc0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x60]);
        vmovups(yword[B + 0xe0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmovups(yword[B + 0x100], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x60]);
        vmovups(yword[B + 0x120], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmovups(yword[B + 0x140], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x60]);
        vmovups(yword[B + 0x160], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -512);
        dec(I);
        jg(l6c, T_NEAR);
        align(4);

        L(l168);
        test(M, 0x4);
        jle(l1e4, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -256);
        align(4);

        L(l1e4);
        test(M, 0x2);
        jle(l228, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -128);
        align(4);

        L(l228);
        test(M, 0x1);
        jle(l250, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmovups(yword[B - 0x60], ymm0);
        sub(B, -64);
        align(4);

        L(l250);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l54, T_NEAR);
        align(4);

        L(l260);
        cmp(N, 0x8);
        jl(l38c, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(l2fc, T_NEAR);
        align(4);

        L(l280);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(l280, T_NEAR);
        align(4);

        L(l2fc);
        test(M, 0x4);
        jle(l340, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(l340);
        test(M, 0x2);
        jle(l36c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l36c);
        test(M, 0x1);
        jle(l388, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(l388);
        sub(N, 0x8);
        align(4);

        L(l38c);
        cmp(N, 0x4);
        jl(l4b4, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l424, T_NEAR);
        align(4);

        L(l3ac);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(l3ac, T_NEAR);
        align(4);

        L(l424);
        test(M, 0x4);
        jle(l468, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(l468);
        test(M, 0x2);
        jle(l494, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l494);
        test(M, 0x1);
        jle(l4b0, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l4b0);
        sub(N, 0x4);
        align(4);

        L(l4b4);
        cmp(N, 0x2);
        jl(l5dc, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l54c, T_NEAR);
        align(4);

        L(l4d4);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(l4d4, T_NEAR);
        align(4);

        L(l54c);
        test(M, 0x4);
        jle(l590, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(l590);
        test(M, 0x2);
        jle(l5bc, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l5bc);
        test(M, 0x1);
        jle(l5d8, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l5d8);
        sub(N, 0x2);
        align(4);

        L(l5dc);
        cmp(N, 0x1);
        jl(l704, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l674, T_NEAR);
        align(4);

        L(l5fc);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(l5fc, T_NEAR);
        align(4);

        L(l674);
        test(M, 0x4);
        jle(l6b8, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(l6b8);
        test(M, 0x2);
        jle(l6e4, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l6e4);
        test(M, 0x1);
        jle(l700, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l700);
        sub(N, 0x1);
        align(4);

        L(l704);
        jmp(l17b4, T_NEAR);
        align(4);

        L(l70c);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(lf6c, T_NEAR);
        vmovaps(ymm6, ymm4);
        cmp(N, 0x10);
        jl(l9b0, T_NEAR);
        align(4);

        L(l728);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x40);
        mov(I, M);
        sar(I, 0x3);
        jle(l87c, T_NEAR);
        align(4);

        L(l740);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x80], ymm0);
        vmovups(ymm0, yword[A2 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xa0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xc0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xe0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x100], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x120], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x140], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x160], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -512);
        dec(I);
        jg(l740, T_NEAR);
        align(4);

        L(l87c);
        test(M, 0x4);
        jle(l91c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -256);
        align(4);

        L(l91c);
        test(M, 0x2);
        jle(l970, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -128);
        align(4);

        L(l970);
        test(M, 0x1);
        jle(l9a0, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        sub(B, -64);
        align(4);

        L(l9a0);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l728, T_NEAR);
        align(4);

        L(l9b0);
        cmp(N, 0x8);
        jl(lb20, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(la74, T_NEAR);
        align(4);

        L(l9d4);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(l9d4, T_NEAR);
        align(4);

        L(la74);
        test(M, 0x4);
        jle(lac8, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(lac8);
        test(M, 0x2);
        jle(lafc, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(lafc);
        test(M, 0x1);
        jle(lb1c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(lb1c);
        sub(N, 0x8);
        align(4);

        L(lb20);
        cmp(N, 0x4);
        jl(lc8c, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(lbe0, T_NEAR);
        align(4);

        L(lb44);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(lb44, T_NEAR);
        align(4);

        L(lbe0);
        test(M, 0x4);
        jle(lc34, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(lc34);
        test(M, 0x2);
        jle(lc68, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(lc68);
        test(M, 0x1);
        jle(lc88, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(lc88);
        sub(N, 0x4);
        align(4);

        L(lc8c);
        cmp(N, 0x2);
        jl(ldf8, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(ld4c, T_NEAR);
        align(4);

        L(lcb0);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(lcb0, T_NEAR);
        align(4);

        L(ld4c);
        test(M, 0x4);
        jle(lda0, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(lda0);
        test(M, 0x2);
        jle(ldd4, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(ldd4);
        test(M, 0x1);
        jle(ldf4, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(ldf4);
        sub(N, 0x2);
        align(4);

        L(ldf8);
        cmp(N, 0x1);
        jl(lf64, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(leb8, T_NEAR);
        align(4);

        L(le1c);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(le1c, T_NEAR);
        align(4);

        L(leb8);
        test(M, 0x4);
        jle(lf0c, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(lf0c);
        test(M, 0x2);
        jle(lf40, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(lf40);
        test(M, 0x1);
        jle(lf60, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lf60);
        sub(N, 0x1);
        align(4);

        L(lf64);
        jmp(l17b4, T_NEAR);
        align(4);

        L(lf6c);
        cmp(N, 0x10);
        jl(l1200, T_NEAR);
        align(4);

        L(lf78);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x40);
        mov(I, M);
        sar(I, 0x3);
        jle(l10cc, T_NEAR);
        align(4);

        L(lf90);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x80], ymm0);
        vmovups(ymm0, yword[A2 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xa0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xc0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0xe0], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x100], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x120], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x140], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x160], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -512);
        dec(I);
        jg(lf90, T_NEAR);
        align(4);

        L(l10cc);
        test(M, 0x4);
        jle(l116c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -256);
        align(4);

        L(l116c);
        test(M, 0x2);
        jle(l11c0, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -128);
        align(4);

        L(l11c0);
        test(M, 0x1);
        jle(l11f0, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 - 0x60]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        sub(B, -64);
        align(4);

        L(l11f0);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(lf78, T_NEAR);
        align(4);

        L(l1200);
        cmp(N, 0x8);
        jl(l1370, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(l12c4, T_NEAR);
        align(4);

        L(l1224);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(l1224, T_NEAR);
        align(4);

        L(l12c4);
        test(M, 0x4);
        jle(l1318, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(l1318);
        test(M, 0x2);
        jle(l134c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l134c);
        test(M, 0x1);
        jle(l136c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(l136c);
        sub(N, 0x8);
        align(4);

        L(l1370);
        cmp(N, 0x4);
        jl(l14dc, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l1430, T_NEAR);
        align(4);

        L(l1394);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(l1394, T_NEAR);
        align(4);

        L(l1430);
        test(M, 0x4);
        jle(l1484, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(l1484);
        test(M, 0x2);
        jle(l14b8, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l14b8);
        test(M, 0x1);
        jle(l14d8, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l14d8);
        sub(N, 0x4);
        align(4);

        L(l14dc);
        cmp(N, 0x2);
        jl(l1648, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l159c, T_NEAR);
        align(4);

        L(l1500);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(l1500, T_NEAR);
        align(4);

        L(l159c);
        test(M, 0x4);
        jle(l15f0, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(l15f0);
        test(M, 0x2);
        jle(l1624, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l1624);
        test(M, 0x1);
        jle(l1644, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l1644);
        sub(N, 0x2);
        align(4);

        L(l1648);
        cmp(N, 0x1);
        jl(l17b4, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l1708, T_NEAR);
        align(4);

        L(l166c);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(l166c, T_NEAR);
        align(4);

        L(l1708);
        test(M, 0x4);
        jle(l175c, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(l175c);
        test(M, 0x2);
        jle(l1790, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l1790);
        test(M, 0x1);
        jle(l17b0, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l17b0);
        sub(N, 0x1);
        align(4);

        L(l17b4);

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
