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

jit_avx_kernel_b0_c_gemm_s8u8s32_kern::jit_avx_kernel_b0_c_gemm_s8u8s32_kern()
    : jit_generator(nullptr, S8U8S32_COMPUTE_KERNEL_CODE_SIZE) {}

void jit_avx_kernel_b0_c_gemm_s8u8s32_kern::generate() {

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
#define ARG_COFFSET_C ((args_offset + 16) + rsp)

#define COFFSET_CX (0 + rsp)
#define COFFSET_CY (8 + rsp)

    inLocalLabel();
    {

        Xbyak::Label l1044;
        Xbyak::Label l1074;
        Xbyak::Label l1078;
        Xbyak::Label l10c4;
        Xbyak::Label l10ec;
        Xbyak::Label l114;
        Xbyak::Label l1174;
        Xbyak::Label l1188;
        Xbyak::Label l1210;
        Xbyak::Label l1248;
        Xbyak::Label l1298;
        Xbyak::Label l12f0;
        Xbyak::Label l1328;
        Xbyak::Label l1358;
        Xbyak::Label l13b8;
        Xbyak::Label l13c4;
        Xbyak::Label l1424;
        Xbyak::Label l144c;
        Xbyak::Label l148c;
        Xbyak::Label l14d4;
        Xbyak::Label l14f0;
        Xbyak::Label l14f4;
        Xbyak::Label l1540;
        Xbyak::Label l1568;
        Xbyak::Label l15f0;
        Xbyak::Label l1604;
        Xbyak::Label l168c;
        Xbyak::Label l16c4;
        Xbyak::Label l1714;
        Xbyak::Label l176c;
        Xbyak::Label l17a4;
        Xbyak::Label l17d4;
        Xbyak::Label l1834;
        Xbyak::Label l1840;
        Xbyak::Label l18a0;
        Xbyak::Label l18c8;
        Xbyak::Label l1908;
        Xbyak::Label l1950;
        Xbyak::Label l196c;
        Xbyak::Label l1970;
        Xbyak::Label l19bc;
        Xbyak::Label l19e4;
        Xbyak::Label l1a6c;
        Xbyak::Label l1a80;
        Xbyak::Label l1b08;
        Xbyak::Label l1b40;
        Xbyak::Label l1b90;
        Xbyak::Label l1be8;
        Xbyak::Label l1c20;
        Xbyak::Label l1c50;
        Xbyak::Label l1cb0;
        Xbyak::Label l1cbc;
        Xbyak::Label l1d1c;
        Xbyak::Label l1d44;
        Xbyak::Label l1d84;
        Xbyak::Label l1dcc;
        Xbyak::Label l1de8;
        Xbyak::Label l1dec;
        Xbyak::Label l260;
        Xbyak::Label l274;
        Xbyak::Label l3c0;
        Xbyak::Label l448;
        Xbyak::Label l4fc;
        Xbyak::Label l5d0;
        Xbyak::Label l670;
        Xbyak::Label l6b8;
        Xbyak::Label l790;
        Xbyak::Label l7a0;
        Xbyak::Label l878;
        Xbyak::Label l8c4;
        Xbyak::Label l93c;
        Xbyak::Label l9d8;
        Xbyak::Label la0;
        Xbyak::Label la2c;
        Xbyak::Label la40;
        Xbyak::Label la8c;
        Xbyak::Label lab8;
        Xbyak::Label lb80;
        Xbyak::Label lb94;
        Xbyak::Label lc5c;
        Xbyak::Label lcb0;
        Xbyak::Label ld20;
        Xbyak::Label lda0;
        Xbyak::Label ldc;
        Xbyak::Label ldfc;
        Xbyak::Label le38;
        Xbyak::Label lec4;
        Xbyak::Label led4;
        Xbyak::Label lf60;
        Xbyak::Label lf94;
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
        mov(H, qword[ARG_COFFSET_C]);
        mov(qword[COFFSET_CX], H);
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
        jl(la40, T_NEAR);
        align(4);

        L(la0);
        mov(CO1, C);
        add(C, 0x40);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x20);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(H, qword[COFFSET_CX]);
        mov(qword[COFFSET_CY], H);
        add(H, 0x40);
        mov(qword[COFFSET_CX], H);
        mov(I, N);
        cmp(I, 0x2);
        jl(l670, T_NEAR);
        align(4);

        L(ldc);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l3c0, T_NEAR);
        sub(H, 0x8);
        jle(l260, T_NEAR);
        align(4);

        L(l114);
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
        jg(l114, T_NEAR);
        align(4);

        L(l260);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l3c0, T_NEAR);
        align(4);

        L(l274);
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
        jg(l274, T_NEAR);
        align(4);

        L(l3c0);
        mov(H, K);
        test(H, 0x4);
        je(l448, T_NEAR);
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

        L(l448);
        mov(H, K);
        test(H, 0x2);
        je(l4fc, T_NEAR);
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

        L(l4fc);
        mov(H, K);
        test(H, 0x1);
        je(l5d0, T_NEAR);
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

        L(l5d0);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xmm0, xword[H + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vpaddd(xmm11, xmm11, xmm0);
        vmovdqu(xmm0, xword[H + 0x20]);
        vpaddd(xmm12, xmm12, xmm0);
        vpaddd(xmm13, xmm13, xmm0);
        vmovdqu(xmm0, xword[H + 0x30]);
        vpaddd(xmm14, xmm14, xmm0);
        vpaddd(xmm15, xmm15, xmm0);
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
        jge(ldc, T_NEAR);
        align(4);

        L(l670);
        test(I, 0x1);
        jle(la2c, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l878, T_NEAR);
        sub(H, 0x8);
        jle(l790, T_NEAR);
        align(4);

        L(l6b8);
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
        jg(l6b8, T_NEAR);
        align(4);

        L(l790);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l878, T_NEAR);
        align(4);

        L(l7a0);
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
        jg(l7a0, T_NEAR);
        align(4);

        L(l878);
        mov(H, K);
        test(H, 0x4);
        je(l8c4, T_NEAR);
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

        L(l8c4);
        mov(H, K);
        test(H, 0x2);
        je(l93c, T_NEAR);
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

        L(l93c);
        mov(H, K);
        test(H, 0x1);
        je(l9d8, T_NEAR);
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

        L(l9d8);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xmm0, xword[H + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xmm0, xword[H + 0x20]);
        vpaddd(xmm12, xmm12, xmm0);
        vmovdqu(xmm0, xword[H + 0x30]);
        vpaddd(xmm14, xmm14, xmm0);
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

        L(la2c);
        mov(A, AO);
        sub(J, 0x10);
        cmp(J, 0x10);
        jge(la0, T_NEAR);
        align(4);

        L(la40);
        test(J, 0x8);
        jle(l1078, T_NEAR);
        mov(CO1, C);
        add(C, 0x20);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x10);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(H, qword[COFFSET_CX]);
        mov(qword[COFFSET_CY], H);
        add(H, 0x20);
        mov(qword[COFFSET_CX], H);
        mov(I, N);
        cmp(I, 0x2);
        jl(ldfc, T_NEAR);
        align(4);

        L(la8c);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(lc5c, T_NEAR);
        sub(H, 0x8);
        jle(lb80, T_NEAR);
        align(4);

        L(lab8);
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
        jg(lab8, T_NEAR);
        align(4);

        L(lb80);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(lc5c, T_NEAR);
        align(4);

        L(lb94);
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
        jg(lb94, T_NEAR);
        align(4);

        L(lc5c);
        mov(H, K);
        test(H, 0x4);
        je(lcb0, T_NEAR);
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

        L(lcb0);
        mov(H, K);
        test(H, 0x2);
        je(ld20, T_NEAR);
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

        L(ld20);
        mov(H, K);
        test(H, 0x1);
        je(lda0, T_NEAR);
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

        L(lda0);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xmm0, xword[H + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vpaddd(xmm11, xmm11, xmm0);
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
        jge(la8c, T_NEAR);
        align(4);

        L(ldfc);
        test(I, 0x1);
        jle(l1074, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(lf60, T_NEAR);
        sub(H, 0x8);
        jle(lec4, T_NEAR);
        align(4);

        L(le38);
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
        jg(le38, T_NEAR);
        align(4);

        L(lec4);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(lf60, T_NEAR);
        align(4);

        L(led4);
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
        jg(led4, T_NEAR);
        align(4);

        L(lf60);
        mov(H, K);
        test(H, 0x4);
        je(lf94, T_NEAR);
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

        L(lf94);
        mov(H, K);
        test(H, 0x2);
        je(lfe4, T_NEAR);
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

        L(lfe4);
        mov(H, K);
        test(H, 0x1);
        je(l1044, T_NEAR);
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

        L(l1044);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xmm0, xword[H + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l1074);
        mov(A, AO);
        align(4);

        L(l1078);
        test(J, 0x4);
        jle(l14f4, T_NEAR);
        mov(CO1, C);
        add(C, 0x10);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x8);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(H, qword[COFFSET_CX]);
        mov(qword[COFFSET_CY], H);
        add(H, 0x10);
        mov(qword[COFFSET_CX], H);
        mov(I, N);
        cmp(I, 0x2);
        jl(l1328, T_NEAR);
        align(4);

        L(l10c4);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1210, T_NEAR);
        sub(H, 0x8);
        jle(l1174, T_NEAR);
        align(4);

        L(l10ec);
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
        jg(l10ec, T_NEAR);
        align(4);

        L(l1174);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l1210, T_NEAR);
        align(4);

        L(l1188);
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
        jg(l1188, T_NEAR);
        align(4);

        L(l1210);
        mov(H, K);
        test(H, 0x4);
        je(l1248, T_NEAR);
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

        L(l1248);
        mov(H, K);
        test(H, 0x2);
        je(l1298, T_NEAR);
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

        L(l1298);
        mov(H, K);
        test(H, 0x1);
        je(l12f0, T_NEAR);
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

        L(l12f0);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l10c4, T_NEAR);
        align(4);

        L(l1328);
        test(I, 0x1);
        jle(l14f0, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1424, T_NEAR);
        sub(H, 0x8);
        jle(l13b8, T_NEAR);
        align(4);

        L(l1358);
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
        jg(l1358, T_NEAR);
        align(4);

        L(l13b8);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l1424, T_NEAR);
        align(4);

        L(l13c4);
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
        jg(l13c4, T_NEAR);
        align(4);

        L(l1424);
        mov(H, K);
        test(H, 0x4);
        je(l144c, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(l144c);
        mov(H, K);
        test(H, 0x2);
        je(l148c, T_NEAR);
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

        L(l148c);
        mov(H, K);
        test(H, 0x1);
        je(l14d4, T_NEAR);
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

        L(l14d4);
        mov(H, qword[COFFSET_CY]);
        vmovdqu(xmm0, xword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l14f0);
        mov(A, AO);
        align(4);

        L(l14f4);
        test(J, 0x2);
        jle(l1970, T_NEAR);
        mov(CO1, C);
        add(C, 0x8);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x4);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(H, qword[COFFSET_CX]);
        mov(qword[COFFSET_CY], H);
        add(H, 0x8);
        mov(qword[COFFSET_CX], H);
        mov(I, N);
        cmp(I, 0x2);
        jl(l17a4, T_NEAR);
        align(4);

        L(l1540);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l168c, T_NEAR);
        sub(H, 0x8);
        jle(l15f0, T_NEAR);
        align(4);

        L(l1568);
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
        jg(l1568, T_NEAR);
        align(4);

        L(l15f0);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l168c, T_NEAR);
        align(4);

        L(l1604);
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
        jg(l1604, T_NEAR);
        align(4);

        L(l168c);
        mov(H, K);
        test(H, 0x4);
        je(l16c4, T_NEAR);
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

        L(l16c4);
        mov(H, K);
        test(H, 0x2);
        je(l1714, T_NEAR);
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

        L(l1714);
        mov(H, K);
        test(H, 0x1);
        je(l176c, T_NEAR);
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

        L(l176c);
        mov(H, qword[COFFSET_CY]);
        vmovsd(xmm0, qword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vpaddd(xmm9, xmm9, xmm0);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovlps(qword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l1540, T_NEAR);
        align(4);

        L(l17a4);
        test(I, 0x1);
        jle(l196c, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l18a0, T_NEAR);
        sub(H, 0x8);
        jle(l1834, T_NEAR);
        align(4);

        L(l17d4);
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
        jg(l17d4, T_NEAR);
        align(4);

        L(l1834);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l18a0, T_NEAR);
        align(4);

        L(l1840);
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
        jg(l1840, T_NEAR);
        align(4);

        L(l18a0);
        mov(H, K);
        test(H, 0x4);
        je(l18c8, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(l18c8);
        mov(H, K);
        test(H, 0x2);
        je(l1908, T_NEAR);
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

        L(l1908);
        mov(H, K);
        test(H, 0x1);
        je(l1950, T_NEAR);
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

        L(l1950);
        mov(H, qword[COFFSET_CY]);
        vmovsd(xmm0, qword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l196c);
        mov(A, AO);
        align(4);

        L(l1970);
        test(J, 0x1);
        jle(l1dec, T_NEAR);
        mov(CO1, C);
        add(C, 0x4);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x2);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(H, qword[COFFSET_CX]);
        mov(qword[COFFSET_CY], H);
        add(H, 0x4);
        mov(qword[COFFSET_CX], H);
        mov(I, N);
        cmp(I, 0x2);
        jl(l1c20, T_NEAR);
        align(4);

        L(l19bc);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1b08, T_NEAR);
        sub(H, 0x8);
        jle(l1a6c, T_NEAR);
        align(4);

        L(l19e4);
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
        jg(l19e4, T_NEAR);
        align(4);

        L(l1a6c);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(l1b08, T_NEAR);
        align(4);

        L(l1a80);
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
        jg(l1a80, T_NEAR);
        align(4);

        L(l1b08);
        mov(H, K);
        test(H, 0x4);
        je(l1b40, T_NEAR);
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

        L(l1b40);
        mov(H, K);
        test(H, 0x2);
        je(l1b90, T_NEAR);
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

        L(l1b90);
        mov(H, K);
        test(H, 0x1);
        je(l1be8, T_NEAR);
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

        L(l1be8);
        mov(H, qword[COFFSET_CY]);
        vmovss(xmm0, dword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vpaddd(xmm9, xmm9, xmm0);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovss(dword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(l19bc, T_NEAR);
        align(4);

        L(l1c20);
        test(I, 0x1);
        jle(l1de8, T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(l1d1c, T_NEAR);
        sub(H, 0x8);
        jle(l1cb0, T_NEAR);
        align(4);

        L(l1c50);
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
        jg(l1c50, T_NEAR);
        align(4);

        L(l1cb0);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(l1d1c, T_NEAR);
        align(4);

        L(l1cbc);
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
        jg(l1cbc, T_NEAR);
        align(4);

        L(l1d1c);
        mov(H, K);
        test(H, 0x4);
        je(l1d44, T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(l1d44);
        mov(H, K);
        test(H, 0x2);
        je(l1d84, T_NEAR);
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

        L(l1d84);
        mov(H, K);
        test(H, 0x1);
        je(l1dcc, T_NEAR);
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

        L(l1dcc);
        mov(H, qword[COFFSET_CY]);
        vmovss(xmm0, dword[H]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(l1de8);
        mov(A, AO);
        align(4);

        L(l1dec);
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
#undef ARG_COFFSET_C
#undef COFFSET_CX
#undef COFFSET_CY
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
