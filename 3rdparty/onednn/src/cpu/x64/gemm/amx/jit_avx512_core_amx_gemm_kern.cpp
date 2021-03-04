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

#include "cpu/x64/gemm/amx/jit_avx512_core_amx_gemm_kern.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define Bm rdi
#define Bn rsi
#define K rdx
#define AO rax
#define BO rbp
#define LDC r13
#define I rcx
#define FLAG r8
#define bm r9
#define bm0 r10
#define bm0w r10w
#define bm1 r11
#define bm1w r11w
#define bn r12
#define cc1 r14
#define cc2 r15
#define T0 rax
#define T0b al
#define T1 rbx
#define T1b bl
#define N qword[rsp + 0x40]
#define A qword[rsp + 0x48]
#define B qword[rsp + 0x50]
#define C qword[rsp + 0x58]
#define FinalM qword[rsp + 0x60]
#define FinalN qword[rsp + 0x68]
#define BackUp0 qword[rsp + 0x70]
#define BackUp1 qword[rsp + 0x78]
#define BackUp2 qword[rsp + 0x80]

#define ARG_X(x) qword[rsp + (STACKSIZE + (x))]

#define STACKSIZE 256
#define TILEB(X) byte[rsp + ((X) + 0xc0)]
#define TILEW(X) word[rsp + ((X) + 0xc0)]
#define TILED(X) dword[rsp + ((X) + 0xc0)]
#define TILEQ(X) qword[rsp + ((X) + 0xc0)]

void jit_avx512_core_amx_gemm_kern::generate() {

    int kerneltype = ((typea << 1) | typeb);

    dim_t SHIFT_UNROLL_M = 4, SHIFT_UNROLL_N = 4, SHIFT_UNROLL_K = 2;
    dim_t SHIFT_UNROLL_MM = 5, SHIFT_UNROLL_NN = 5, SHIFT_UNROLL_KK = 6;
    dim_t SHIFT_A = 0, SHIFT_B = 0, SHIFT_C = 2;

    dim_t UNROLL_M = 0, UNROLL_N = 0, UNROLL_K = 0;
    dim_t UNROLL_MM = 0, UNROLL_NN = 0, UNROLL_KK = 0;
    dim_t SIZE_A = 0, SIZE_B = 0, SIZE_C = 0;

    if (typec == 0) {
        // Floatingpoint Operation
        SHIFT_A = 1;
        SHIFT_B = 1;
        SHIFT_UNROLL_K = 1;
        SHIFT_UNROLL_KK = 5;

        kerneltype = 4;
    }

    UNROLL_M = (((dim_t)1) << SHIFT_UNROLL_M);
    UNROLL_N = (((dim_t)1) << SHIFT_UNROLL_N);
    UNROLL_K = (((dim_t)1) << SHIFT_UNROLL_K);

    UNROLL_MM = (((dim_t)1) << SHIFT_UNROLL_MM);
    UNROLL_NN = (((dim_t)1) << SHIFT_UNROLL_NN);
    UNROLL_KK = (((dim_t)1) << SHIFT_UNROLL_KK);

    SIZE_A = (((dim_t)1) << SHIFT_A);
    SIZE_B = (((dim_t)1) << SHIFT_B);
    SIZE_C = (((dim_t)1) << SHIFT_C);

    Xbyak::Label loopE, loopM, loopN, loopK;
    Xbyak::Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, la, lb;

    sub(rsp, STACKSIZE);

    mov(ptr[rsp + 0x00], rbx);
    mov(ptr[rsp + 0x08], rbp);
    mov(ptr[rsp + 0x10], r12);
    mov(ptr[rsp + 0x18], r13);
    mov(ptr[rsp + 0x20], r14);
    mov(ptr[rsp + 0x28], r15);

#ifdef _WIN32
    mov(ptr[rsp + 0x30], rsi);
    mov(ptr[rsp + 0x38], rdi);

    /* ABI conversion for Windows */
    mov(Bm, qword[rcx]);
    mov(Bn, qword[rdx]);
    mov(K, qword[r8]);

    mov(rcx, ARG_X(0x28));
    mov(B, ARG_X(0x30));
    mov(C, ARG_X(0x38));
    mov(LDC, ARG_X(0x40));
#else

    mov(Bm, qword[rdi]);
    mov(Bn, qword[rsi]);
    mov(K, qword[rdx]);
    mov(rcx, r8);
    mov(r8, r9);
    mov(r9, ARG_X(0x08));
    mov(LDC, ARG_X(0x10));
#endif

    /* Quick return if possible */
    test(Bm, Bm);
    jle(loopE, T_NEAR);
    test(Bn, Bn);
    jle(loopE, T_NEAR);
    test(K, K);
    jle(loopE, T_NEAR);

    /* Initializing tile. First value is pallate ID */
    mov(TILEQ(0x00), 1);
    mov(TILEQ(0x08), 0);
    mov(TILEQ(0x20), 0);
    mov(TILEQ(0x28), 0);
    mov(TILEQ(0x38), 0);

    /* Clear back up data */
    mov(BackUp0, 0);
    mov(BackUp1, 0);
    mov(BackUp2, 0);

    sal(LDC, SHIFT_C);

    /* K needs to be multiple of 4 */
    add(K, UNROLL_K - 1);
    and_(K, -UNROLL_K);

    mov(N, Bn);
    mov(A, rcx);
    mov(B, r8);
    mov(C, r9);

    /* Calculating last value for M loop */
    lea(T0, ptr[Bm - 1]);
    and_(T0, -UNROLL_MM);
    lea(T1, ptr[Bm - UNROLL_MM]);
    sub(T1, T0);
    mov(FinalM, T1);

    /* Calculating last value for N loop */
    lea(T0, ptr[Bn - 1]);
    and_(T0, -UNROLL_NN);
    lea(T1, ptr[Bn - UNROLL_NN]);
    sub(T1, T0);
    mov(FinalN, T1);
    align(4);

    L(loopM);
    /* Updating C address */
    mov(cc1, C);
    mov(cc2, LDC);
    sal(cc2, SHIFT_UNROLL_N);
    add(cc2, cc1);
    add(C, UNROLL_MM * SIZE_C);

    mov(bm, UNROLL_MM);
    cmp(Bm, UNROLL_MM);
    cmovle(bm, Bm);

    mov(bm0, UNROLL_M);
    cmp(bm, UNROLL_M);
    cmovle(bm0, bm);

    mov(bm1, bm);
    sub(bm1, bm0);

    /* Filling in tile information for M */
    mov(TILEB(0x30), UNROLL_KK / UNROLL_K);
    mov(TILEB(0x31), UNROLL_KK / UNROLL_K);

    sal(bm0, SHIFT_UNROLL_K + SHIFT_A);
    sal(bm1, SHIFT_UNROLL_K + SHIFT_A);

    mov(TILEW(0x10), bm0);
    mov(TILEW(0x12), bm1);

    sar(bm0, SHIFT_UNROLL_K + SHIFT_A - SHIFT_C);
    sar(bm1, SHIFT_UNROLL_K + SHIFT_A - SHIFT_C);

    mov(TILEW(0x18), bm0);
    mov(TILEW(0x1a), bm0);
    mov(TILEW(0x1c), bm1);
    mov(TILEW(0x1e), bm1);

    sal(bm, SHIFT_UNROLL_KK + SHIFT_A);

    xor_(FLAG, FLAG);
    mov(T0, 2);
    cmp(bm1, 0);
    cmovg(FLAG, T0);

    mov(Bn, N);
    mov(BO, B);
    align(4);

    L(loopN);
    mov(bn, UNROLL_NN);
    cmp(Bn, UNROLL_NN);
    cmovle(bn, Bn);

    mov(T0, UNROLL_N);
    cmp(bn, UNROLL_N);
    cmovle(T0, bn);

    mov(T1, bn);
    sub(T1, T0);

    /* Filling in tile information for N */
    mov(TILEW(0x14), UNROLL_KK * SIZE_B);
    mov(TILEW(0x16), UNROLL_KK * SIZE_B);

    mov(TILEB(0x32), T0);
    mov(TILEB(0x33), T1);

    mov(TILEB(0x34), T0);
    mov(TILEB(0x35), T1);
    mov(TILEB(0x36), T0);
    mov(TILEB(0x37), T1);

    sal(bn, SHIFT_UNROLL_KK + SHIFT_B);

    /* Disabling unnecessary tile */
    test(FLAG, 2);
    jg(l0);

    mov(TILEW(0x12), 0x00);
    mov(TILEW(0x1c), 0x00);
    mov(TILEW(0x1e), 0x00);
    mov(TILEB(0x31), 0x00);
    mov(TILEB(0x36), 0x00);
    mov(TILEB(0x37), 0x00);

    L(l0);
    or_(FLAG, 1);
    cmp(T1, 0);
    jg(l1);

    and_(FLAG, -2);

    mov(TILEW(0x16), 0x00);
    mov(TILEW(0x1a), 0x00);
    mov(TILEW(0x1e), 0x00);
    mov(TILEB(0x33), 0x00);
    mov(TILEB(0x35), 0x00);
    mov(TILEB(0x37), 0x00);

    L(l1);
    /* Configuring tile if tile has been changed */
    mov(T1, BackUp0);
    mov(T0, TILEQ(0x10));
    mov(BackUp0, T0);
    xor_(T1, T0);

    xor_(T1, BackUp1);
    mov(T0, TILEQ(0x18));
    mov(BackUp1, T0);
    xor_(T1, T0);

    xor_(T1, BackUp2);
    mov(T0, TILEQ(0x30));
    mov(BackUp2, T0);
    xor_(T1, T0);
    je(lb);

    ldtilecfg(TILEQ(0));

    L(lb);
    /* Load for C */
    if (isBetaZero) {
        tilezero(tmm4);
    } else {
        tileloadd(tmm4, ptr[cc1 + LDC + UNROLL_M * SIZE_C * 0]);
    }
    test(FLAG, 1);
    jle(l2);
    if (isBetaZero) {
        tilezero(tmm5);
    } else {
        tileloadd(tmm5, ptr[cc2 + LDC + UNROLL_M * SIZE_C * 0]);
    }
    L(l2);
    test(FLAG, 2);
    jle(l3);
    if (isBetaZero) {
        tilezero(tmm6);
    } else {
        tileloadd(tmm6, ptr[cc1 + LDC + UNROLL_M * SIZE_C * 1]);
    }
    L(l3);
    cmp(FLAG, 3);
    jne(l4);
    if (isBetaZero) {
        tilezero(tmm7);
    } else {
        tileloadd(tmm7, ptr[cc2 + LDC + UNROLL_M * SIZE_C * 1]);
    }
    L(l4);

    mov(AO, A);
    mov(T1, UNROLL_KK * SIZE_B);

    mov(I, K);
    align(4);

    /*
      Kernel Loop 4 load + 4 TMUL compute (2 x 2 tiles)
                     T5, T6, T7 may be skipped
          B0  B1
      A0  T4  T6
      A1  T5  T7
   */
    L(loopK);
    tileloadd(tmm0, ptr[AO + bm0 + UNROLL_M * UNROLL_KK * SIZE_A * 0]);
    tileloadd(tmm2, ptr[BO + T1 + UNROLL_N * UNROLL_KK * SIZE_B * 0]);

    switch (kerneltype) {
        case 0: tdpbuud(tmm4, tmm2, tmm0); break;
        case 1: tdpbsud(tmm4, tmm2, tmm0); break;
        case 2: tdpbusd(tmm4, tmm2, tmm0); break;
        case 3: tdpbssd(tmm4, tmm2, tmm0); break;
        case 4: tdpbf16ps(tmm4, tmm2, tmm0); break;
    }

    test(FLAG, 2);
    jle(l5);
    tileloadd(tmm1, ptr[AO + bm1 + UNROLL_M * UNROLL_KK * SIZE_A * 1]);

    switch (kerneltype) {
        case 0: tdpbuud(tmm6, tmm2, tmm1); break;
        case 1: tdpbsud(tmm6, tmm2, tmm1); break;
        case 2: tdpbusd(tmm6, tmm2, tmm1); break;
        case 3: tdpbssd(tmm6, tmm2, tmm1); break;
        case 4: tdpbf16ps(tmm6, tmm2, tmm1); break;
    }

    L(l5);
    test(FLAG, 1);
    jle(l6);
    tileloadd(tmm3, ptr[BO + T1 + UNROLL_N * UNROLL_KK * SIZE_B * 1]);

    switch (kerneltype) {
        case 0: tdpbuud(tmm5, tmm3, tmm0); break;
        case 1: tdpbsud(tmm5, tmm3, tmm0); break;
        case 2: tdpbusd(tmm5, tmm3, tmm0); break;
        case 3: tdpbssd(tmm5, tmm3, tmm0); break;
        case 4: tdpbf16ps(tmm5, tmm3, tmm0); break;
    }

    L(l6);
    cmp(FLAG, 3);
    jne(l7);

    switch (kerneltype) {
        case 0: tdpbuud(tmm7, tmm3, tmm1); break;
        case 1: tdpbsud(tmm7, tmm3, tmm1); break;
        case 2: tdpbusd(tmm7, tmm3, tmm1); break;
        case 3: tdpbssd(tmm7, tmm3, tmm1); break;
        case 4: tdpbf16ps(tmm7, tmm3, tmm1); break;
    }

    L(l7);

    add(AO, bm);
    add(BO, bn);
    sub(I, UNROLL_KK);
    jg(loopK);
    align(4);

    /* Store for C */
    tilestored(ptr[cc1 + LDC + UNROLL_M * SIZE_C * 0], tmm4);
    test(FLAG, 1);
    jle(l8);
    tilestored(ptr[cc2 + LDC + UNROLL_M * SIZE_C * 0], tmm5);
    L(l8);
    test(FLAG, 2);
    jle(l9);
    tilestored(ptr[cc1 + LDC + UNROLL_M * SIZE_C * 1], tmm6);
    L(l9);
    cmp(FLAG, 3);
    jne(la);
    tilestored(ptr[cc2 + LDC + UNROLL_M * SIZE_C * 1], tmm7);
    L(la);

    /* Updating C address */
    mov(T0, LDC);
    sal(T0, SHIFT_UNROLL_NN);
    add(cc1, T0);
    add(cc2, T0);

    /* Checking end of N loop */
    sub(Bn, UNROLL_NN);
    cmp(FinalN, Bn);
    jne(loopN);
    align(4);

    /* Updating for next A */
    lea(T0, ptr[K + UNROLL_KK - 1]);
    and_(T0, -UNROLL_KK);
    sal(T0, SHIFT_UNROLL_MM + SHIFT_A);
    add(A, T0);

    /* Checking end of M loop */
    sub(Bm, UNROLL_MM);
    cmp(FinalM, Bm);
    jne(loopM);
    align(4);

    L(loopE);
    tilerelease();

    mov(rbx, ptr[rsp + 0x00]);
    mov(rbp, ptr[rsp + 0x08]);
    mov(r12, ptr[rsp + 0x10]);
    mov(r13, ptr[rsp + 0x18]);
    mov(r14, ptr[rsp + 0x20]);
    mov(r15, ptr[rsp + 0x28]);

#ifdef _WIN32
    mov(rsi, ptr[rsp + 0x30]);
    mov(rdi, ptr[rsp + 0x38]);
#endif

    add(rsp, STACKSIZE);

    ret();
}

jit_avx512_core_amx_gemm_kern::jit_avx512_core_amx_gemm_kern(
        int typea, int typeb, int typec, int betaZero)
    : jit_generator(nullptr, 1024)
    , typea(typea)
    , typeb(typeb)
    , typec(typec)
    , isBetaZero(betaZero) {}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
