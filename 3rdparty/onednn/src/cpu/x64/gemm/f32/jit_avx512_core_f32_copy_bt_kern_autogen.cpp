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

jit_avx512_core_f32_copy_bt_kern::jit_avx512_core_f32_copy_bt_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_f32_copy_bt_kern::generate() {

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

        Xbyak::Label l1048;
        Xbyak::Label l107c;
        Xbyak::Label l109c;
        Xbyak::Label l10a0;
        Xbyak::Label l12c;
        Xbyak::Label l158;
        Xbyak::Label l174;
        Xbyak::Label l184;
        Xbyak::Label l1a4;
        Xbyak::Label l21c;
        Xbyak::Label l260;
        Xbyak::Label l28c;
        Xbyak::Label l2a8;
        Xbyak::Label l2ac;
        Xbyak::Label l2cc;
        Xbyak::Label l344;
        Xbyak::Label l388;
        Xbyak::Label l3b4;
        Xbyak::Label l3d0;
        Xbyak::Label l3d4;
        Xbyak::Label l3f4;
        Xbyak::Label l46c;
        Xbyak::Label l4b0;
        Xbyak::Label l4dc;
        Xbyak::Label l4f8;
        Xbyak::Label l4fc;
        Xbyak::Label l504;
        Xbyak::Label l524;
        Xbyak::Label l53c;
        Xbyak::Label l58;
        Xbyak::Label l5dc;
        Xbyak::Label l630;
        Xbyak::Label l664;
        Xbyak::Label l684;
        Xbyak::Label l694;
        Xbyak::Label l6b8;
        Xbyak::Label l6c;
        Xbyak::Label l754;
        Xbyak::Label l7a8;
        Xbyak::Label l7dc;
        Xbyak::Label l7fc;
        Xbyak::Label l800;
        Xbyak::Label l824;
        Xbyak::Label l8c0;
        Xbyak::Label l914;
        Xbyak::Label l948;
        Xbyak::Label l968;
        Xbyak::Label l96c;
        Xbyak::Label l990;
        Xbyak::Label la2c;
        Xbyak::Label la80;
        Xbyak::Label lab4;
        Xbyak::Label lad4;
        Xbyak::Label lad8;
        Xbyak::Label lae0;
        Xbyak::Label laec;
        Xbyak::Label lb04;
        Xbyak::Label lba4;
        Xbyak::Label lbf8;
        Xbyak::Label lc2c;
        Xbyak::Label lc4c;
        Xbyak::Label lc5c;
        Xbyak::Label lc80;
        Xbyak::Label ld1c;
        Xbyak::Label ld70;
        Xbyak::Label lda4;
        Xbyak::Label ldc4;
        Xbyak::Label ldc8;
        Xbyak::Label ldec;
        Xbyak::Label le8;
        Xbyak::Label le88;
        Xbyak::Label ledc;
        Xbyak::Label lf10;
        Xbyak::Label lf30;
        Xbyak::Label lf34;
        Xbyak::Label lf58;
        Xbyak::Label lff4;

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
        vbroadcastss(zmm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vbroadcastss(zmm4, xmm4);
        vucomiss(xmm6, xmm3);
        jne(l504, T_NEAR);
        cmp(N, 0x8);
        jl(l184, T_NEAR);
        align(4);

        L(l58);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(le8, T_NEAR);
        align(4);

        L(l6c);
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
        jg(l6c, T_NEAR);
        align(4);

        L(le8);
        test(M, 0x4);
        jle(l12c, T_NEAR);
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

        L(l12c);
        test(M, 0x2);
        jle(l158, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l158);
        test(M, 0x1);
        jle(l174, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(l174);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l58, T_NEAR);
        align(4);

        L(l184);
        cmp(N, 0x4);
        jl(l2ac, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l21c, T_NEAR);
        align(4);

        L(l1a4);
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
        jg(l1a4, T_NEAR);
        align(4);

        L(l21c);
        test(M, 0x4);
        jle(l260, T_NEAR);
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

        L(l260);
        test(M, 0x2);
        jle(l28c, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l28c);
        test(M, 0x1);
        jle(l2a8, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l2a8);
        sub(N, 0x4);
        align(4);

        L(l2ac);
        cmp(N, 0x2);
        jl(l3d4, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l344, T_NEAR);
        align(4);

        L(l2cc);
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
        jg(l2cc, T_NEAR);
        align(4);

        L(l344);
        test(M, 0x4);
        jle(l388, T_NEAR);
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

        L(l388);
        test(M, 0x2);
        jle(l3b4, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l3b4);
        test(M, 0x1);
        jle(l3d0, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l3d0);
        sub(N, 0x2);
        align(4);

        L(l3d4);
        cmp(N, 0x1);
        jl(l4fc, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l46c, T_NEAR);
        align(4);

        L(l3f4);
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
        jg(l3f4, T_NEAR);
        align(4);

        L(l46c);
        test(M, 0x4);
        jle(l4b0, T_NEAR);
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

        L(l4b0);
        test(M, 0x2);
        jle(l4dc, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l4dc);
        test(M, 0x1);
        jle(l4f8, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l4f8);
        sub(N, 0x1);
        align(4);

        L(l4fc);
        jmp(l10a0, T_NEAR);
        align(4);

        L(l504);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(lae0, T_NEAR);
        vmovaps(zmm6, zmm4);
        cmp(N, 0x8);
        jl(l694, T_NEAR);
        align(4);

        L(l524);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(l5dc, T_NEAR);
        align(4);

        L(l53c);
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
        jg(l53c, T_NEAR);
        align(4);

        L(l5dc);
        test(M, 0x4);
        jle(l630, T_NEAR);
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

        L(l630);
        test(M, 0x2);
        jle(l664, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(l664);
        test(M, 0x1);
        jle(l684, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(l684);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l524, T_NEAR);
        align(4);

        L(l694);
        cmp(N, 0x4);
        jl(l800, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(l754, T_NEAR);
        align(4);

        L(l6b8);
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
        jg(l6b8, T_NEAR);
        align(4);

        L(l754);
        test(M, 0x4);
        jle(l7a8, T_NEAR);
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

        L(l7a8);
        test(M, 0x2);
        jle(l7dc, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(l7dc);
        test(M, 0x1);
        jle(l7fc, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l7fc);
        sub(N, 0x4);
        align(4);

        L(l800);
        cmp(N, 0x2);
        jl(l96c, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l8c0, T_NEAR);
        align(4);

        L(l824);
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
        jg(l824, T_NEAR);
        align(4);

        L(l8c0);
        test(M, 0x4);
        jle(l914, T_NEAR);
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

        L(l914);
        test(M, 0x2);
        jle(l948, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(l948);
        test(M, 0x1);
        jle(l968, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l968);
        sub(N, 0x2);
        align(4);

        L(l96c);
        cmp(N, 0x1);
        jl(lad8, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(la2c, T_NEAR);
        align(4);

        L(l990);
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
        jg(l990, T_NEAR);
        align(4);

        L(la2c);
        test(M, 0x4);
        jle(la80, T_NEAR);
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

        L(la80);
        test(M, 0x2);
        jle(lab4, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(lab4);
        test(M, 0x1);
        jle(lad4, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lad4);
        sub(N, 0x1);
        align(4);

        L(lad8);
        jmp(l10a0, T_NEAR);
        align(4);

        L(lae0);
        cmp(N, 0x8);
        jl(lc5c, T_NEAR);
        align(4);

        L(laec);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(lba4, T_NEAR);
        align(4);

        L(lb04);
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
        jg(lb04, T_NEAR);
        align(4);

        L(lba4);
        test(M, 0x4);
        jle(lbf8, T_NEAR);
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

        L(lbf8);
        test(M, 0x2);
        jle(lc2c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(lc2c);
        test(M, 0x1);
        jle(lc4c, T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(lc4c);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(laec, T_NEAR);
        align(4);

        L(lc5c);
        cmp(N, 0x4);
        jl(ldc8, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(ld1c, T_NEAR);
        align(4);

        L(lc80);
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
        jg(lc80, T_NEAR);
        align(4);

        L(ld1c);
        test(M, 0x4);
        jle(ld70, T_NEAR);
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

        L(ld70);
        test(M, 0x2);
        jle(lda4, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(lda4);
        test(M, 0x1);
        jle(ldc4, T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(ldc4);
        sub(N, 0x4);
        align(4);

        L(ldc8);
        cmp(N, 0x2);
        jl(lf34, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(le88, T_NEAR);
        align(4);

        L(ldec);
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
        jg(ldec, T_NEAR);
        align(4);

        L(le88);
        test(M, 0x4);
        jle(ledc, T_NEAR);
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

        L(ledc);
        test(M, 0x2);
        jle(lf10, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(lf10);
        test(M, 0x1);
        jle(lf30, T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lf30);
        sub(N, 0x2);
        align(4);

        L(lf34);
        cmp(N, 0x1);
        jl(l10a0, T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(lff4, T_NEAR);
        align(4);

        L(lf58);
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
        jg(lf58, T_NEAR);
        align(4);

        L(lff4);
        test(M, 0x4);
        jle(l1048, T_NEAR);
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

        L(l1048);
        test(M, 0x2);
        jle(l107c, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(l107c);
        test(M, 0x1);
        jle(l109c, T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l109c);
        sub(N, 0x1);
        align(4);

        L(l10a0);

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
