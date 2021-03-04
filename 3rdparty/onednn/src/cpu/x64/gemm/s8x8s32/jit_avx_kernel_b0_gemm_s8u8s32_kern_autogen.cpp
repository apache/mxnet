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

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_avx_kernel_b0_gemm_s8u8s32_kern::jit_avx_kernel_b0_gemm_s8u8s32_kern()
    : jit_generator(nullptr, S8U8S32_COMPUTE_KERNEL_CODE_SIZE) {}

void jit_avx_kernel_b0_gemm_s8u8s32_kern::generate() {

#ifndef _WIN32

#define M rdi
#define N rsi
#define K rdx
#define A r8
#define B r9
#define C r10
#define LDC r11

#define AA rcx
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#else

#define M rcx
#define N rdx
#define K r8
#define A rsi
#define B r9
#define C r10
#define LDC r11

#define AA rdi
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#endif

#ifdef _WIN32
#define ARG_A (args_offset - 16) + rsp
#define ARG_B (args_offset - 8) + rsp
#endif
#define ARG_C ((args_offset + 0) + rsp)
#define ARG_LDC ((args_offset + 8) + rsp)

    inLocalLabel();
    {

        Xbyak::Label l100c;
        Xbyak::Label l1094;
        Xbyak::Label l10a8;
        Xbyak::Label l1130;
        Xbyak::Label l1168;
        Xbyak::Label l11b8;
        Xbyak::Label l1210;
        Xbyak::Label l1238;
        Xbyak::Label l1268;
        Xbyak::Label l12c8;
        Xbyak::Label l12d4;
        Xbyak::Label l1334;
        Xbyak::Label l135c;
        Xbyak::Label l139c;
        Xbyak::Label l13e4;
        Xbyak::Label l13f4;
        Xbyak::Label l13f8;
        Xbyak::Label l1430;
        Xbyak::Label l1458;
        Xbyak::Label l14e0;
        Xbyak::Label l14f4;
        Xbyak::Label l157c;
        Xbyak::Label l15b4;
        Xbyak::Label l1604;
        Xbyak::Label l165c;
        Xbyak::Label l1684;
        Xbyak::Label l16b4;
        Xbyak::Label l1714;
        Xbyak::Label l1720;
        Xbyak::Label l1780;
        Xbyak::Label l17a8;
        Xbyak::Label l17e8;
        Xbyak::Label l1830;
        Xbyak::Label l1840;
        Xbyak::Label l1844;
        Xbyak::Label l187c;
        Xbyak::Label l18a4;
        Xbyak::Label l192c;
        Xbyak::Label l1940;
        Xbyak::Label l19c8;
        Xbyak::Label l1a00;
        Xbyak::Label l1a50;
        Xbyak::Label l1aa8;
        Xbyak::Label l1ad0;
        Xbyak::Label l1b00;
        Xbyak::Label l1b60;
        Xbyak::Label l1b6c;
        Xbyak::Label l1bcc;
        Xbyak::Label l1bf4;
        Xbyak::Label l1c34;
        Xbyak::Label l1c7c;
        Xbyak::Label l1c8c;
        Xbyak::Label l1c90;
        Xbyak::Label l240;
        Xbyak::Label l254;
        Xbyak::Label l3a0;
        Xbyak::Label l428;
        Xbyak::Label l4dc;
        Xbyak::Label l5b0;
        Xbyak::Label l618;
        Xbyak::Label l660;
        Xbyak::Label l738;
        Xbyak::Label l748;
        Xbyak::Label l820;
        Xbyak::Label l86c;
        Xbyak::Label l8e4;
        Xbyak::Label l94;
        Xbyak::Label l980;
        Xbyak::Label l9ac;
        Xbyak::Label l9c0;
        Xbyak::Label l9f8;
        Xbyak::Label la24;
        Xbyak::Label laec;
        Xbyak::Label lb00;
        Xbyak::Label lbc;
        Xbyak::Label lbc8;
        Xbyak::Label lc1c;
        Xbyak::Label lc8c;
        Xbyak::Label ld0c;
        Xbyak::Label ld48;
        Xbyak::Label ld84;
        Xbyak::Label le10;
        Xbyak::Label le20;
        Xbyak::Label leac;
        Xbyak::Label lee0;
        Xbyak::Label lf30;
        Xbyak::Label lf4;
        Xbyak::Label lf90;
        Xbyak::Label lfa8;
        Xbyak::Label lfac;
        Xbyak::Label lfe4;

        auto stack_alloc_size = 32;
        auto args_offset = stack_alloc_size + get_size_of_abi_save_regs() + 8;
#ifdef _WIN32
        args_offset += 48;
#endif
        preamble();
        sub(rsp, stack_alloc_size);
#ifdef _WIN32
        mov(A, ptr[ARG_A]);
        mov(B, ptr[ARG_B]);
#endif

        mov(C, qword[ARG_C]);
        mov(LDC, qword[ARG_LDC]);
        sub(A, -128);
        sub(B, -128);
        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(K, qword[K]);
        lea(LDC, ptr[LDC * 4 + 0x0]);
        vxorps(xmm8, xmm8, xmm8);
        vxorps(xmm9, xmm9, xmm9);
        vxorps(xmm10, xmm10, xmm10);
        vxorps(xmm11, xmm11, xmm11);
        vxorps(xmm12, xmm12, xmm12);
        vxorps(xmm13, xmm13, xmm13);
        vxorps(xmm14, xmm14, xmm14);
        vxorps(xmm15, xmm15, xmm15);
        mov(H, 0x10001);
        movq(xmm7, H);
        vpshufd(xmm7, xmm7, 0x0);
        mov(J, M);
        cmp(J, 0x10);
        jl(l9c0, T_NEAR);
        align(4);

        L(l94);
        mov(CO1, C);
        add(C, 0x40);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x20);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(l618, T_NEAR);
        align(4);

        L(lbc);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l3a0, T_NEAR);
        sub(H, 0x8);
        jle(l240, T_NEAR);
        align(4);

        L(lf4);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(lf4, T_NEAR);
        align(4);

        L(l240);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l3a0, T_NEAR);
        align(4);

        L(l254);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l254, T_NEAR);
        align(4);

        L(l3a0);
        mov(H, K);
        test(H, 0x4);
        je(l428, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x40);
        add(BO, 0x8);
        align(4);

        L(l428);
        mov(H, K);
        test(H, 0x2);
        je(l4dc, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vmovdqu(xmm3, xword[AO - 0x70]);
        vpunpcklwd(xmm2, xmm3, xmm6);
        vpunpckhwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(l4dc);
        mov(H, K);
        test(H, 0x1);
        je(l5b0, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm2, dword[AO - 0x78]);
        vpunpcklbw(xmm2, xmm2, xmm6);
        vpunpcklwd(xmm2, xmm2, xmm6);
        vbroadcastss(xmm3, dword[AO - 0x74]);
        vpunpcklbw(xmm3, xmm3, xmm6);
        vpunpcklwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(l5b0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xword[CO1 + 0x20], xmm12);
        vxorps(xmm12, xmm12, xmm12);
        vmovdqu(xword[CO1 + 0x30], xmm14);
        vxorps(xmm14, xmm14, xmm14);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        vmovdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        vxorps(xmm11, xmm11, xmm11);
        vmovdqu(xword[CO1 + LDC * 1 + 0x20], xmm13);
        vxorps(xmm13, xmm13, xmm13);
        vmovdqu(xword[CO1 + LDC * 1 + 0x30], xmm15);
        vxorps(xmm15, xmm15, xmm15);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(lbc, T_NEAR);
        align(4);

        L(l618);
        test(I, 0x1);
        jle(l9ac, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l820, T_NEAR);
        sub(H, 0x8);
        jle(l738, T_NEAR);
        align(4);

        L(l660);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l660, T_NEAR);
        align(4);

        L(l738);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l820, T_NEAR);
        align(4);

        L(l748);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l748, T_NEAR);
        align(4);

        L(l820);
        mov(H, K);
        test(H, 0x4);
        je(l86c, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x40);
        add(BO, 0x4);
        align(4);

        L(l86c);
        mov(H, K);
        test(H, 0x2);
        je(l8e4, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vmovdqu(xmm3, xword[AO - 0x70]);
        vpunpcklwd(xmm2, xmm3, xmm6);
        vpunpckhwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x20);
        add(BO, 0x2);
        align(4);

        L(l8e4);
        mov(H, K);
        test(H, 0x1);
        je(l980, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm2, dword[AO - 0x78]);
        vpunpcklbw(xmm2, xmm2, xmm6);
        vpunpcklwd(xmm2, xmm2, xmm6);
        vbroadcastss(xmm3, dword[AO - 0x74]);
        vpunpcklbw(xmm3, xmm3, xmm6);
        vpunpcklwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x10);
        add(BO, 0x1);
        align(4);

        L(l980);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xword[CO1 + 0x20], xmm12);
        vxorps(xmm12, xmm12, xmm12);
        vmovdqu(xword[CO1 + 0x30], xmm14);
        vxorps(xmm14, xmm14, xmm14);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l9ac);
        mov(A, AO);
        sub(J, 0x10);
        cmp(J, 0x10);
        jge(l94, T_NEAR);
        align(4);

        L(l9c0);
        test(J, 0x8);
        jle(lfac, T_NEAR);
        mov(CO1, C);
        add(C, 0x20);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x10);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(ld48, T_NEAR);
        align(4);

        L(l9f8);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(lbc8, T_NEAR);
        sub(H, 0x8);
        jle(laec, T_NEAR);
        align(4);

        L(la24);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(la24, T_NEAR);
        align(4);

        L(laec);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(lbc8, T_NEAR);
        align(4);

        L(lb00);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(lb00, T_NEAR);
        align(4);

        L(lbc8);
        mov(H, K);
        test(H, 0x4);
        je(lc1c, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x20);
        add(BO, 0x8);
        align(4);

        L(lc1c);
        mov(H, K);
        test(H, 0x2);
        je(lc8c, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(lc8c);
        mov(H, K);
        test(H, 0x1);
        je(ld0c, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(ld0c);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        vmovdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        vxorps(xmm11, xmm11, xmm11);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l9f8, T_NEAR);
        align(4);

        L(ld48);
        test(I, 0x1);
        jle(lfa8, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(leac, T_NEAR);
        sub(H, 0x8);
        jle(le10, T_NEAR);
        align(4);

        L(ld84);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(ld84, T_NEAR);
        align(4);

        L(le10);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(leac, T_NEAR);
        align(4);

        L(le20);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(le20, T_NEAR);
        align(4);

        L(leac);
        mov(H, K);
        test(H, 0x4);
        je(lee0, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(lee0);
        mov(H, K);
        test(H, 0x2);
        je(lf30, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(lf30);
        mov(H, K);
        test(H, 0x1);
        je(lf90, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x8);
        add(BO, 0x1);
        align(4);

        L(lf90);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(lfa8);
        mov(A, AO);
        align(4);

        L(lfac);
        test(J, 0x4);
        jle(l13f8, T_NEAR);
        mov(CO1, C);
        add(C, 0x10);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x8);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(l1238, T_NEAR);
        align(4);

        L(lfe4);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1130, T_NEAR);
        sub(H, 0x8);
        jle(l1094, T_NEAR);
        align(4);

        L(l100c);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l100c, T_NEAR);
        align(4);

        L(l1094);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l1130, T_NEAR);
        align(4);

        L(l10a8);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l10a8, T_NEAR);
        align(4);

        L(l1130);
        mov(H, K);
        test(H, 0x4);
        je(l1168, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x10);
        add(BO, 0x8);
        align(4);

        L(l1168);
        mov(H, K);
        test(H, 0x2);
        je(l11b8, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(l11b8);
        mov(H, K);
        test(H, 0x1);
        je(l1210, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(l1210);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(lfe4, T_NEAR);
        align(4);

        L(l1238);
        test(I, 0x1);
        jle(l13f4, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1334, T_NEAR);
        sub(H, 0x8);
        jle(l12c8, T_NEAR);
        align(4);

        L(l1268);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l1268, T_NEAR);
        align(4);

        L(l12c8);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l1334, T_NEAR);
        align(4);

        L(l12d4);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l12d4, T_NEAR);
        align(4);

        L(l1334);
        mov(H, K);
        test(H, 0x4);
        je(l135c, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(l135c);
        mov(H, K);
        test(H, 0x2);
        je(l139c, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(l139c);
        mov(H, K);
        test(H, 0x1);
        je(l13e4, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x1);
        align(4);

        L(l13e4);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l13f4);
        mov(A, AO);
        align(4);

        L(l13f8);
        test(J, 0x2);
        jle(l1844, T_NEAR);
        mov(CO1, C);
        add(C, 0x8);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x4);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(l1684, T_NEAR);
        align(4);

        L(l1430);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l157c, T_NEAR);
        sub(H, 0x8);
        jle(l14e0, T_NEAR);
        align(4);

        L(l1458);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l1458, T_NEAR);
        align(4);

        L(l14e0);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l157c, T_NEAR);
        align(4);

        L(l14f4);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l14f4, T_NEAR);
        align(4);

        L(l157c);
        mov(H, K);
        test(H, 0x4);
        je(l15b4, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x8);
        add(BO, 0x8);
        align(4);

        L(l15b4);
        mov(H, K);
        test(H, 0x2);
        je(l1604, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(l1604);
        mov(H, K);
        test(H, 0x1);
        je(l165c, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(l165c);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovlps(qword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l1430, T_NEAR);
        align(4);

        L(l1684);
        test(I, 0x1);
        jle(l1840, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1780, T_NEAR);
        sub(H, 0x8);
        jle(l1714, T_NEAR);
        align(4);

        L(l16b4);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l16b4, T_NEAR);
        align(4);

        L(l1714);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l1780, T_NEAR);
        align(4);

        L(l1720);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l1720, T_NEAR);
        align(4);

        L(l1780);
        mov(H, K);
        test(H, 0x4);
        je(l17a8, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(l17a8);
        mov(H, K);
        test(H, 0x2);
        je(l17e8, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(l17e8);
        mov(H, K);
        test(H, 0x1);
        je(l1830, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x2);
        add(BO, 0x1);
        align(4);

        L(l1830);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l1840);
        mov(A, AO);
        align(4);

        L(l1844);
        test(J, 0x1);
        jle(l1c90, T_NEAR);
        mov(CO1, C);
        add(C, 0x4);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x2);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(l1ad0, T_NEAR);
        align(4);

        L(l187c);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l19c8, T_NEAR);
        sub(H, 0x8);
        jle(l192c, T_NEAR);
        align(4);

        L(l18a4);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l18a4, T_NEAR);
        align(4);

        L(l192c);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l19c8, T_NEAR);
        align(4);

        L(l1940);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(l1940, T_NEAR);
        align(4);

        L(l19c8);
        mov(H, K);
        test(H, 0x4);
        je(l1a00, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x8);
        align(4);

        L(l1a00);
        mov(H, K);
        test(H, 0x2);
        je(l1a50, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x2);
        add(BO, 0x4);
        align(4);

        L(l1a50);
        mov(H, K);
        test(H, 0x1);
        je(l1aa8, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x1);
        add(BO, 0x2);
        align(4);

        L(l1aa8);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovss(dword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l187c, T_NEAR);
        align(4);

        L(l1ad0);
        test(I, 0x1);
        jle(l1c8c, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1bcc, T_NEAR);
        sub(H, 0x8);
        jle(l1b60, T_NEAR);
        align(4);

        L(l1b00);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l1b00, T_NEAR);
        align(4);

        L(l1b60);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l1bcc, T_NEAR);
        align(4);

        L(l1b6c);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(l1b6c, T_NEAR);
        align(4);

        L(l1bcc);
        mov(H, K);
        test(H, 0x4);
        je(l1bf4, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(l1bf4);
        mov(H, K);
        test(H, 0x2);
        je(l1c34, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(l1c34);
        mov(H, K);
        test(H, 0x1);
        je(l1c7c, T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x1);
        add(BO, 0x1);
        align(4);

        L(l1c7c);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l1c8c);
        mov(A, AO);
        align(4);

        L(l1c90);
        add(rsp, stack_alloc_size);
        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef K
#undef A
#undef B
#undef C
#undef LDC
#undef AA
#undef I
#undef J
#undef H
#undef AO
#undef BO
#undef CO1
#undef CO2
#ifdef _WIN32
#undef ARG_A
#undef ARG_B
#endif
#undef ARG_C
#undef ARG_LDC
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
