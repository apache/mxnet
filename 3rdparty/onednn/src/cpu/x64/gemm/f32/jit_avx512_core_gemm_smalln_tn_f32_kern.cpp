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

#include <atomic>
#include <mutex>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/gemm/f32/jit_avx512_core_gemm_smalln_tn_f32_kern.hpp"
#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define STACKSIZE 1024

// Convert between vector register lengths.
static inline Xbyak::Ymm make_ymm(const Xbyak::Zmm &v) {
    return Xbyak::Ymm(v.getIdx());
}

namespace avx512_core_gemm_smalln_tn_f32 {

struct xbyak_gemm_smalln_tn_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_gemm_smalln_tn_xbyak_gemm)

    xbyak_gemm_smalln_tn_t(int N, float beta, float alpha,
            void *code_ptr = nullptr,
            size_t code_size = 80 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size), N(N), beta(beta), alpha(alpha) {}

    void generate() override {
        using namespace Xbyak;
        /**
         * numN = 1 : 16 rows of A, 1x16 accumulators
         * numN = 2 : 8  rows of A, 2x8  accumulators
         * numN = 3 : 8  rows of A, 3x8  accumulators
         * numN = 4 : 8  rows of A, 4x8  accumulators + stack
         */
        numN = N; // Number of columns.
        isBeta0 = (beta == 0.0);
        isBetaN = (!isBeta0 && beta != 1.0);

        isAlpha0 = (alpha == 0.0);
        isAlphaN = (!isAlpha0 && alpha != 1.0);

        // various definitions for convenience
        auto ARG_M = abi_param1;
        auto ARG_K = abi_param2;
        auto ARG_ALPHA = abi_param3;
        auto ARG_BETA = abi_param4;
#ifdef _WIN32
        auto ARG_A = ptr[rsp + OFFSET_SHADOWSPACE + get_size_of_abi_save_regs()
                + STACKSIZE];
        auto ARG_LDA = qword[rsp + OFFSET_SHADOWSPACE + sizeof(float *)
                + get_size_of_abi_save_regs() + STACKSIZE];
        A = rsi;
        LDA = rdi;
        const auto stackOffset = get_size_of_abi_save_regs()
                + OFFSET_SHADOWSPACE + sizeof(float *) + STACKSIZE;
#else
        auto ARG_A = r8;
        auto ARG_LDA = r9;
        A = ARG_A;
        LDA = ARG_LDA;
        const auto stackOffset = get_size_of_abi_save_regs() + STACKSIZE;
#endif
        auto ARG_B = ptr[rsp + 8 + stackOffset];
        auto ARG_LDB = ptr[rsp + 16 + stackOffset];
        auto ARG_C = ptr[rsp + 24 + stackOffset];
        auto ARG_LDC = ptr[rsp + 32 + stackOffset];

        TEMP_REG = abi_param1;
        AO1 = abi_param2;
        BO2 = abi_param3; // numN == 4
        CO2 = abi_param4;
        JJ = rbx;
        II = rbp;
        BO1 = rax;
        TEMP_REG2 = r10;
        B = r11;
        LDB = r12; // numN > 1
        LDC = r13;
        CO1 = r14;
        AO2 = r15;
        // masks for load/store remainder handling
        k_rem = k1;
        m_rem = k2;

        auto M = qword[rsp + 64];
        auto K = qword[rsp + 72];
        auto ORIG_A = qword[rsp + 80];
        auto ORIG_B = qword[rsp + 88];
        auto KREM = qword[rsp + 96];
        auto MREM = qword[rsp + 104];
        auto ALPHA = ptr_b[rsp + 112];
        auto BETA = ptr_b[rsp + 120];

        const int perm_ab_offset = 128;
        const int perm_ba_offset = perm_ab_offset + 64;
        const int perm_ab1_offset = perm_ba_offset + 64;
        const int perm_ba1_offset = perm_ab1_offset + 64; // 384

        static Zmm zmmreg[] = {zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22,
                zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,
                zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9,
                zmm10, zmm11, zmm12, zmm13, zmm14, zmm15};

        static Address perm_[] = {dword[rsp + perm_ab_offset],
                dword[rsp + perm_ba_offset], dword[rsp + perm_ab1_offset],
                dword[rsp + perm_ba1_offset]};

        const int tempzmm_offset = 488;
        static Address TEMPZMM_[8] = {ptr[rsp + tempzmm_offset + 64 * 0],
                ptr[rsp + tempzmm_offset + 64 * 1],
                ptr[rsp + tempzmm_offset + 64 * 2],
                ptr[rsp + tempzmm_offset + 64 * 3],
                ptr[rsp + tempzmm_offset + 64 * 4],
                ptr[rsp + tempzmm_offset + 64 * 5],
                ptr[rsp + tempzmm_offset + 64 * 6],
                ptr[rsp + tempzmm_offset + 64 * 7]};

        int numMREM = (numN == 1) ? 16 : 8;
        Label label_mremT[17], label_kremT[16], label_k_loopT[16],
                label_no_k_remT[16];

        zmm_reg = zmmreg;
        TEMPZMM = TEMPZMM_;
        perm = perm_;

        // Start of kernel.
        preamble();
        sub(rsp, STACKSIZE);

#ifdef _WIN32
        mov(A, ARG_A);
        mov(LDA, ARG_LDA);
#endif

        // Back up all parameters
        mov(M, ARG_M);
        mov(K, ARG_K);
        mov(ORIG_A, A);
        mov(B, ARG_B);
        mov(ORIG_B, B);
        mov(LDC, ARG_LDC);
        mov(LDB, ARG_LDB);

        mov(ARG_ALPHA, dword[ARG_ALPHA]);
        mov(ARG_BETA, dword[ARG_BETA]);
        mov(ALPHA, ARG_ALPHA);
        mov(BETA, ARG_BETA);

        if (!isAlpha0) {
            for (int ii = 0; ii < 16; ii++) {
                mov(dword[rsp + perm_ab_offset + 4 * ii], permute_ab[ii]);
                mov(dword[rsp + perm_ba_offset + 4 * ii], permute_ba[ii]);
                mov(dword[rsp + perm_ab1_offset + 4 * ii], permute_ab1[ii]);
                mov(dword[rsp + perm_ba1_offset + 4 * ii], permute_ba1[ii]);
            }
        }

        mov(CO1, ARG_C);
        shl(LDA, 2); // sizeof(float) * LDA
        shl(LDC, 2); // sizeof(float) * LDC
        if (numN > 1) shl(LDB, 2); // sizeof(float) * LDB

        // Check for K remainder
        if (!isAlpha0) {
            mov(TEMP_REG, K);
            and_(K, ~15);
            sub(TEMP_REG, K);
            mov(KREM, TEMP_REG);
            mov(rax, 1);
            mov(rcx, TEMP_REG);
            shl(rax, cl);
            sub(rax, 1);
            kmovq(k_rem, rax);
        }

        // Check for M remainder
        mov(TEMP_REG2, M);
        and_(M, (numN > 1) ? ~7 : ~15);
        sub(TEMP_REG2, M);
        mov(MREM, TEMP_REG2);
        mov(rax, 1);
        mov(rcx, TEMP_REG2);
        shl(rax, cl);
        sub(rax, 1);
        kmovq(m_rem, rax);

        // If M < unroll skip M loop.
        mov(TEMP_REG2, M);
        test(TEMP_REG2, TEMP_REG2);
        jz(label_mrem, T_NEAR);

        // M LOOP
        mov(II, M);
        L_aligned(label_m_loop);

        if (!isAlpha0) {
            mov(AO1, ORIG_A);
            mov(BO1, ORIG_B);
            if (numN == 4) {
                lea(BO2, ptr[BO1 + LDB]);
                lea(BO2, ptr[BO2 + LDB * 2]);
            }
            mov(JJ, K);
            zero_accumulators();
            mov(TEMP_REG2, K);
            test(TEMP_REG2, TEMP_REG2);
            jz(label_krem, T_NEAR);
            // K LOOP
            L_aligned(label_k_loop);
            kloop(false, ALPHA, (numN == 1) ? 16 : 8);
            test(JJ, JJ);
            jne(label_k_loop);
            // Handle k remainder
            mov(TEMP_REG2, KREM);
            test(TEMP_REG2, TEMP_REG2);
            jz(label_no_k_rem, T_NEAR);
            L_aligned(label_krem);
            kloop(true, ALPHA, (numN == 1) ? 16 : 8);
            L_aligned(label_no_k_rem);
            reduction_16x16();
        } else
            zero_accumulators();

        updateC(false, BETA);

        sub(II, (numN < 2) ? 16 : 8);
        if (!isAlpha0) mov(AO1, ORIG_A);
        if (!isAlpha0) lea(AO1, ptr[AO1 + LDA * 8]);
        add(CO1, (numN < 2) ? 64 : 32);

        if (numN < 2 && !isAlpha0) lea(AO1, ptr[AO1 + LDA * 8]);
        if (!isAlpha0) mov(ORIG_A, AO1);
        test(II, II);
        jne(label_m_loop); // End of M loop

        mov(TEMP_REG, MREM);
        test(TEMP_REG, TEMP_REG);
        je(label_mremT[numMREM], T_NEAR); // No remainders for M, skip to end.

        L_aligned(label_mrem);
        if (!isAlpha0) {
            // Handling M remainders...
            mov(AO1, ORIG_A);
            mov(BO1, ORIG_B);
            if (numN == 4) {
                lea(BO2, ptr[BO1 + LDB]);
                lea(BO2, ptr[BO2 + LDB * 2]);
            }
            mov(JJ, K);
            zero_accumulators();

            mov(TEMP_REG, MREM);
            for (int ii = 0; ii < numMREM - 1; ii++) {
                cmp(TEMP_REG, ii + 1);
                je(label_mremT[ii], T_NEAR);
            }
            for (int ii = 0; ii < numMREM - 1; ii++) {
                L_aligned(label_mremT[ii]);
                mov(TEMP_REG, K);
                test(TEMP_REG, TEMP_REG);
                jz(label_kremT[ii], T_NEAR);
                // K LOOP
                L_aligned(label_k_loopT[ii]);
                kloop(false, ALPHA, ii + 1);
                test(JJ, JJ);
                jne(label_k_loopT[ii]);
                // Handle k remainder
                mov(TEMP_REG2, KREM);
                test(TEMP_REG2, TEMP_REG2);
                jz(label_no_k_remT[ii], T_NEAR);
                L_aligned(label_kremT[ii]);
                kloop(true, ALPHA, ii + 1);
                L_aligned(label_no_k_remT[ii]);
                jmp(label_mremT[numMREM - 1], T_NEAR);
            }
            L_aligned(label_mremT[numMREM - 1]);
            reduction_16x16();
        } else
            zero_accumulators();

        updateC(true, BETA);
        L_aligned(label_mremT[numMREM]);

        add(rsp, STACKSIZE);
        postamble();
    }

    void zero_accumulators() {
        // Set accumlators to zero.
        if (isAlpha0) { // zero all.
            for (int ii = 0; ii < 32; ii++) // 0-31
                vpxorq(zmm_reg[ii], zmm_reg[ii], zmm_reg[ii]);
            return;
        }
        for (int ii = 0; ii < 16; ii++) // 16-31
            vpxorq(zmm_reg[16 + ii], zmm_reg[16 + ii], zmm_reg[16 + ii]);

        if (numN < 5) {
            for (int ii = 0; ii < 8; ii++) // 0-7
                vpxorq(zmm_reg[ii], zmm_reg[ii], zmm_reg[ii]);
        }
        if (numN == 4) {
            for (int ii = 0; ii < 4; ii++) { // 8-11
                vpxorq(zmm_reg[ii + 8], zmm_reg[ii + 8], zmm_reg[ii + 8]);
                vmovups(TEMPZMM[2 * ii], zmm_reg[ii + 8]);
                vmovups(TEMPZMM[2 * ii + 1], zmm_reg[ii + 8]);
            }
        }
    }

    void updateC(bool mrem, Xbyak::Operand &BETA) {
        if (numN < 2) {
            if (!isBeta0) {
                vmovups(zmm_reg[25] | (mrem ? m_rem : k0), ptr[CO1]);
                if (isBetaN) vmulps(zmm_reg[25], zmm_reg[25], BETA);
                vaddps(zmm_reg[24], zmm_reg[24], zmm_reg[25]);
            }
            vmovups(ptr[CO1] | (mrem ? m_rem : k0), zmm_reg[24]);
        } else {
            if (!isBeta0) {
                vmovups(make_ymm(zmm_reg[1]) | (mrem ? m_rem : k0), ptr[CO1]);
                vmovups(make_ymm(zmm_reg[2]) | (mrem ? m_rem : k0),
                        ptr[CO1 + LDC]);
                if (isBetaN) {
                    vmulps(make_ymm(zmm_reg[1]), make_ymm(zmm_reg[1]), BETA);
                    vmulps(make_ymm(zmm_reg[2]), make_ymm(zmm_reg[2]), BETA);
                }
            }
            vshuff64x2(zmm_reg[4], zmm_reg[24], zmm_reg[24], 0b11101110);
            if (!isBeta0)
                vaddps(make_ymm(zmm_reg[24]), make_ymm(zmm_reg[24]),
                        make_ymm(zmm_reg[1]));
            if (!isBeta0)
                vaddps(make_ymm(zmm_reg[4]), make_ymm(zmm_reg[4]),
                        make_ymm(zmm_reg[2]));
            vmovups(ptr[CO1] | (mrem ? m_rem : k0), make_ymm(zmm_reg[24]));
            vmovups(ptr[CO1 + LDC] | (mrem ? m_rem : k0), make_ymm(zmm_reg[4]));

            if (numN == 3) {
                if (!isBeta0) {
                    vmovups(make_ymm(zmm_reg[5]) | (mrem ? m_rem : k0),
                            ptr[CO1 + LDC * 2]);
                    if (isBetaN)
                        vmulps(make_ymm(zmm_reg[5]), make_ymm(zmm_reg[5]),
                                BETA);
                    vaddps(make_ymm(zmm_reg[16]), make_ymm(zmm_reg[16]),
                            make_ymm(zmm_reg[5]));
                }
                vmovups(ptr[CO1 + LDC * 2] | (mrem ? m_rem : k0),
                        make_ymm(zmm_reg[16]));
            }
            if (numN == 4) {
                lea(CO2, ptr[CO1 + LDC]);
                lea(CO2, ptr[CO2 + LDC * 2]);
                if (!isBeta0) {
                    vmovups(make_ymm(zmm_reg[5]) | (mrem ? m_rem : k0),
                            ptr[CO1 + LDC * 2]);
                    vmovups(make_ymm(zmm_reg[6]) | (mrem ? m_rem : k0),
                            ptr[CO2]);
                    if (isBetaN) {
                        vmulps(make_ymm(zmm_reg[5]), make_ymm(zmm_reg[5]),
                                BETA);
                        vmulps(make_ymm(zmm_reg[6]), make_ymm(zmm_reg[6]),
                                BETA);
                    }
                }
                vshuff64x2(zmm_reg[10], zmm_reg[0], zmm_reg[0], 0b11101110);
                if (!isBeta0)
                    vaddps(make_ymm(zmm_reg[0]), make_ymm(zmm_reg[0]),
                            make_ymm(zmm_reg[5]));
                if (!isBeta0)
                    vaddps(make_ymm(zmm_reg[10]), make_ymm(zmm_reg[10]),
                            make_ymm(zmm_reg[6]));
                vmovups(ptr[CO1 + LDC * 2] | (mrem ? m_rem : k0),
                        make_ymm(zmm_reg[0]));
                vmovups(ptr[CO2] | (mrem ? m_rem : k0), make_ymm(zmm_reg[10]));
            }
        }
    }

    void reduction_16x16() {
        // 16-way reduction
        /**
         * Does not touch zmm_reg[0]-zmm_reg[7]
         * Final reductions stored in zmm_reg[24]
         */
        // [ A1 | A2 ] , [ B1 | B2 ]  --> [ A1 | B1 ], [ A2 | B2 ]
        for (int ii = 0; ii < 8; ii++) {
            vshuff32x4(zmm_reg[ii + 8], zmm_reg[ii + 16], zmm_reg[ii + 16 + 8],
                    0b01000100); // [ A1 | B1 ]
            vshuff32x4(zmm_reg[ii + 9], zmm_reg[ii + 16], zmm_reg[ii + 16 + 8],
                    0b11101110); // [ A2 | B2 ]
            vaddps(zmm_reg[ii + 24], zmm_reg[ii + 8],
                    zmm_reg[ii + 9]); // [ A1 + A2 | B1 + B2 ]
        }
        vmovups(zmm_reg[8], perm[0]);
        vmovups(zmm_reg[9], perm[1]);
        for (int ii = 0; ii < 4; ii++) {
            vmovaps(zmm_reg[10], zmm_reg[8]);
            vmovaps(zmm_reg[11], zmm_reg[9]);
            vpermi2ps(zmm_reg[10], zmm_reg[ii + 24], zmm_reg[ii + 24 + 4]);
            vpermi2ps(zmm_reg[11], zmm_reg[ii + 24], zmm_reg[ii + 24 + 4]);
            vaddps(zmm_reg[ii + 24], zmm_reg[10], zmm_reg[11]);
        }
        for (int ii = 0; ii < 2; ii++) {
            vshufpd(zmm_reg[10], zmm_reg[ii + 24], zmm_reg[ii + 24 + 2],
                    0b00000000);
            vshufpd(zmm_reg[11], zmm_reg[ii + 24], zmm_reg[ii + 24 + 2],
                    0b11111111);
            vaddps(zmm_reg[ii + 24], zmm_reg[10], zmm_reg[11]);
        }
        vmovups(zmm_reg[12], perm[2]);
        vmovups(zmm_reg[13], perm[3]);
        vpermi2ps(zmm_reg[12], zmm_reg[24], zmm_reg[25]);
        vpermi2ps(zmm_reg[13], zmm_reg[24], zmm_reg[25]);
        vaddps(zmm_reg[24], zmm_reg[12], zmm_reg[13]);

        if (numN == 4) {
            // Final reductions stored in zmm_reg[0]
            for (int ii = 0; ii < 8; ii++)
                vmovups(zmm_reg[ii + 8], TEMPZMM[ii]);

            for (int ii = 0; ii < 8; ii++) {
                vshuff32x4(zmm_reg[ii + 16], zmm_reg[ii], zmm_reg[ii + 8],
                        0b01000100);
                if (ii == 7) {
                    vshuff32x4(zmm_reg[30], zmm_reg[ii], zmm_reg[ii + 8],
                            0b11101110);
                    vaddps(zmm_reg[ii], zmm_reg[ii + 16], zmm_reg[30]);
                } else {
                    vshuff32x4(zmm_reg[ii + 16 + 1], zmm_reg[ii],
                            zmm_reg[ii + 8], 0b11101110);
                    vaddps(zmm_reg[ii], zmm_reg[ii + 16], zmm_reg[ii + 16 + 1]);
                }
            }
            vmovups(zmm_reg[8], perm[0]);
            vmovups(zmm_reg[9], perm[1]);
            for (int ii = 0; ii < 4; ii++) {
                vmovaps(zmm_reg[10], zmm_reg[8]);
                vmovaps(zmm_reg[11], zmm_reg[9]);
                vpermi2ps(zmm_reg[10], zmm_reg[ii], zmm_reg[ii + 4]);
                vpermi2ps(zmm_reg[11], zmm_reg[ii], zmm_reg[ii + 4]);
                vaddps(zmm_reg[ii], zmm_reg[10], zmm_reg[11]);
            }
            for (int ii = 0; ii < 2; ii++) {
                vshufpd(zmm_reg[10], zmm_reg[ii], zmm_reg[ii + 2], 0b00000000);
                vshufpd(zmm_reg[11], zmm_reg[ii], zmm_reg[ii + 2], 0b11111111);
                vaddps(zmm_reg[ii], zmm_reg[10], zmm_reg[11]);
            }
            vmovups(zmm_reg[12], perm[2]);
            vmovups(zmm_reg[13], perm[3]);
            vpermi2ps(zmm_reg[12], zmm_reg[0], zmm_reg[1]);
            vpermi2ps(zmm_reg[13], zmm_reg[0], zmm_reg[1]);
            vaddps(zmm_reg[0], zmm_reg[12], zmm_reg[13]);
        }

        if (numN == 3) {
            // 8-way reduction and store to zmm_reg[16]
            // zmm_reg[0-7] stores values to be reduced.
            for (int ii = 0; ii < 8; ii++) {
                if (ii < 7) {
                    vshuff32x4(zmm_reg[ii + 8], zmm_reg[ii], zmm_reg[ii],
                            0b01000100);
                    vshuff32x4(zmm_reg[ii + 9], zmm_reg[ii], zmm_reg[ii],
                            0b11101110);
                    vaddps(zmm_reg[ii + 16], zmm_reg[ii + 8], zmm_reg[ii + 9]);
                } else {
                    vshuff32x4(
                            zmm_reg[30], zmm_reg[ii], zmm_reg[ii], 0b01000100);
                    vshuff32x4(
                            zmm_reg[31], zmm_reg[ii], zmm_reg[ii], 0b11101110);
                    vaddps(zmm_reg[ii + 16], zmm_reg[30], zmm_reg[31]);
                }
            }
            vmovups(zmm_reg[8], perm[0]);
            vmovups(zmm_reg[9], perm[1]);
            for (int ii = 0; ii < 4; ii++) {
                vmovaps(zmm_reg[10], zmm_reg[8]);
                vmovaps(zmm_reg[11], zmm_reg[9]);
                vpermi2ps(zmm_reg[10], zmm_reg[ii + 16], zmm_reg[ii + 16 + 4]);
                vpermi2ps(zmm_reg[11], zmm_reg[ii + 16], zmm_reg[ii + 16 + 4]);
                vaddps(zmm_reg[ii + 16], zmm_reg[10], zmm_reg[11]);
            }
            for (int ii = 0; ii < 2; ii++) {
                vshufpd(zmm_reg[10], zmm_reg[ii + 16], zmm_reg[ii + 16 + 2],
                        0b00000000);
                vshufpd(zmm_reg[11], zmm_reg[ii + 16], zmm_reg[ii + 16 + 2],
                        0b11111111);
                vaddps(zmm_reg[ii + 16], zmm_reg[10], zmm_reg[11]);
            }
            vmovups(zmm_reg[12], perm[2]);
            vmovups(zmm_reg[13], perm[3]);
            vpermi2ps(zmm_reg[12], zmm_reg[16], zmm_reg[17]);
            vpermi2ps(zmm_reg[13], zmm_reg[16], zmm_reg[17]);
            vaddps(zmm_reg[16], zmm_reg[12], zmm_reg[13]);
        }
    }

    void kloop(bool krem, Xbyak::Address &ALPHA, int MROW) {
        mov(AO2, AO1); // AO2 temporary.
        // Load 1,2,3 or 4 column(s) of B
        if (krem) {
            vpxorq(zmm_reg[15], zmm_reg[15], zmm_reg[15]);
            if (numN > 1) vpxorq(zmm_reg[14], zmm_reg[14], zmm_reg[14]);
            if (numN > 2) vpxorq(zmm_reg[13], zmm_reg[13], zmm_reg[13]);
            if (numN > 3) vpxorq(zmm_reg[12], zmm_reg[12], zmm_reg[12]);
        }
        vmovups(zmm_reg[15] | (krem ? k_rem : k0), ptr[BO1]);
        if (numN > 1)
            vmovups(zmm_reg[14] | (krem ? k_rem : k0), ptr[BO1 + LDB]);
        if (numN > 2)
            vmovups(zmm_reg[13] | (krem ? k_rem : k0), ptr[BO1 + LDB * 2]);
        if (numN > 3) vmovups(zmm_reg[12] | (krem ? k_rem : k0), ptr[BO2]);

        if (isAlphaN) {
            vmulps(zmm_reg[15], zmm_reg[15], ALPHA);
            if (numN > 1) vmulps(zmm_reg[14], zmm_reg[14], ALPHA);
            if (numN > 2) vmulps(zmm_reg[13], zmm_reg[13], ALPHA);
            if (numN > 3) vmulps(zmm_reg[12], zmm_reg[12], ALPHA);
        }

        // Load 16 rows of A
        if (numN == 4) {
            /**
             * zmm_reg[12-15] stores B values
             * zmm_reg[8-11] stores accumulators values
             */
            int endval = (MROW < 4) ? MROW : 4;

            for (int ii = 0; ii < endval; ii++) {
                vmovups(zmm_reg[ii + 8], TEMPZMM[ii]);
            }
            for (int ii = 0; ii < endval; ii++) {
                vfmadd231ps(zmm_reg[ii + 8] | (krem ? k_rem : k0), zmm_reg[12],
                        ptr[AO2]);
                vfmadd231ps(zmm_reg[16 + ii] | (krem ? k_rem : k0), zmm_reg[15],
                        ptr[AO2]);
                vfmadd231ps(zmm_reg[16 + 8 + ii] | (krem ? k_rem : k0),
                        zmm_reg[14], ptr[AO2]);
                vfmadd231ps(zmm_reg[ii] | (krem ? k_rem : k0), zmm_reg[13],
                        ptr[AO2]);
                vmovups(TEMPZMM[ii], zmm_reg[ii + 8]);
                add(AO2, LDA);
            }
            if (MROW > 4) {
                MROW -= 4;
                for (int ii = 0; ii < MROW; ii++)
                    vmovups(zmm_reg[ii + 8], TEMPZMM[ii + 4]);
                for (int ii = 4; ii < 4 + MROW; ii++) {
                    vfmadd231ps(zmm_reg[ii + 4] | (krem ? k_rem : k0),
                            zmm_reg[12], ptr[AO2]);
                    vfmadd231ps(zmm_reg[16 + ii] | (krem ? k_rem : k0),
                            zmm_reg[15], ptr[AO2]);
                    vfmadd231ps(zmm_reg[16 + 8 + ii] | (krem ? k_rem : k0),
                            zmm_reg[14], ptr[AO2]);
                    vfmadd231ps(zmm_reg[ii] | (krem ? k_rem : k0), zmm_reg[13],
                            ptr[AO2]);
                    vmovups(TEMPZMM[ii], zmm_reg[ii + 4]);
                    add(AO2, LDA);
                }
            }
        }
        if (numN == 3) {
            /**
             * zmm_reg[13-15] stores B values
             * zmm_reg[8-12] stores A values
             */
            int endval = (MROW < 5) ? MROW : 5;
            for (int ii = 8; ii < 8 + endval; ii++) {
                // Storing A values in zmm_reg[8-12]
                vmovups(zmm_reg[ii] | (krem ? k_rem : k0) | T_z, ptr[AO2]);
                add(AO2, LDA);
            }
            for (int ii = 0; ii < endval; ii++) {
                vfmadd231ps(zmm_reg[16 + ii], zmm_reg[8 + ii], zmm_reg[15]);
                vfmadd231ps(zmm_reg[16 + 8 + ii], zmm_reg[8 + ii], zmm_reg[14]);
                vfmadd231ps(zmm_reg[ii], zmm_reg[8 + ii], zmm_reg[13]);
            }
            if (MROW > 5) {
                for (int ii = 5; ii < MROW; ii++) {
                    vfmadd231ps(zmm_reg[16 + ii] | (krem ? k_rem : k0),
                            zmm_reg[15], ptr[AO2]);
                    vfmadd231ps(zmm_reg[16 + 8 + ii] | (krem ? k_rem : k0),
                            zmm_reg[14], ptr[AO2]);
                    vfmadd231ps(zmm_reg[ii] | (krem ? k_rem : k0), zmm_reg[13],
                            ptr[AO2]);
                    add(AO2, LDA);
                }
            }
        }
        if (numN < 3) {
            int MROW2 = (MROW > 8)
                    ? 8
                    : MROW; // Do not process more than 8 rows here.
            for (int ii = 0; ii < MROW2; ii++) {
                vmovups(zmm_reg[ii] | (krem ? k_rem : k0) | T_z, ptr[AO2]);
                add(AO2, LDA);
            }
            for (int ii = 0; ii < MROW2; ii++) {
                vfmadd231ps(zmm_reg[16 + ii], zmm_reg[ii], zmm_reg[15]);
                if (numN == 2)
                    vfmadd231ps(zmm_reg[16 + 8 + ii], zmm_reg[ii], zmm_reg[14]);
            }
        }
        if (numN == 1) {
            if (MROW > 8) {
                vmovaps(zmm_reg[0], zmm_reg[15]);
                for (int ii = 8; ii < MROW; ii++) {
                    vmovups(zmm_reg[ii] | (krem ? k_rem : k0) | T_z, ptr[AO2]);
                    add(AO2, LDA);
                }
                for (int ii = 8; ii < MROW; ii++)
                    vfmadd231ps(zmm_reg[16 + ii], zmm_reg[ii], zmm_reg[0]);
            }
        }

        if (!krem) {
            sub(JJ, 16);
            add(AO1, 16 * 4);
            add(BO1, 16 * 4);
            if (numN == 4) add(BO2, 16 * 4);
        }
    }

private:
    uint32_t numN = 0;
    const int N = 0;
    const float beta = 0.0f;
    const float alpha = 0.0f;
    bool isBeta0 = false, isBetaN = false, isAlpha0 = false, isAlphaN = false;
    Xbyak::Zmm *zmm_reg = nullptr;
    Xbyak::Reg64 A, AO1, AO2, B, BO1, BO2, CO1, CO2;
    Xbyak::Reg64 LDA, LDB, LDC, II, JJ;
    Xbyak::Reg64 TEMP_REG, TEMP_REG2;
    Xbyak::Address *TEMPZMM = nullptr, *perm = nullptr;
    Xbyak::Opmask k_rem, m_rem;
    Xbyak::Label label_m_loop, label_k_loop, label_no_k_rem, label_krem,
            label_mrem;
    static uint32_t permute_ab[], permute_ba[], permute_ab1[], permute_ba1[];
};

uint32_t xbyak_gemm_smalln_tn_t::permute_ab[] = {0x00, 0x01, 0x02, 0x03, 0x10,
        0x11, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B};
uint32_t xbyak_gemm_smalln_tn_t::permute_ba[] = {0x04, 0x05, 0x06, 0x07, 0x14,
        0x15, 0x16, 0x17, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F};
uint32_t xbyak_gemm_smalln_tn_t::permute_ab1[] = {0x00, 0x10, 0x02, 0x12, 0x04,
        0x14, 0x06, 0x16, 0x08, 0x18, 0x0A, 0x1A, 0x0C, 0x1C, 0x0E, 0x1E};
uint32_t xbyak_gemm_smalln_tn_t::permute_ba1[] = {0x01, 0x11, 0x03, 0x13, 0x05,
        0x15, 0x07, 0x17, 0x09, 0x19, 0x0B, 0x1B, 0x0D, 0x1D, 0x0F, 0x1F};

/**
 * Currently N=4 kernels are not dispatched. For small/medium sizes using
 * 2x N=2 is better. For larger sizes, the N=4 kernel is better.
 * TODO: Investigate this further.
 */
static dim_t partitions[15][6] = {{0, 1, 1, 1, 1, 1}, {0, 2, 2, 2, 2, 2},
        {0, 3, 3, 3, 3, 3}, {0, 2, 4, 4, 4, 4}, {0, 3, 5, 5, 5, 5},
        {0, 3, 6, 6, 6, 6}, {0, 2, 4, 7, 7, 7}, {0, 3, 6, 8, 8, 8},
        {0, 3, 6, 9, 9, 9}, {0, 2, 4, 7, 10, 10}, {0, 3, 6, 9, 11, 11},
        {0, 3, 6, 9, 12, 12}, {0, 2, 4, 7, 10, 13}, {0, 3, 6, 9, 12, 14},
        {0, 3, 6, 9, 12, 15}};
} // namespace avx512_core_gemm_smalln_tn_f32

template <>
dnnl_status_t jump_to_gemm_smalln_tn(
        const gemm_info_t<float, float, float> *arg) {
    if ((arg->n < 16 && arg->n > 1 && arg->transa == do_trans
                && arg->transb != do_trans)
            && mayiuse(avx512_core) && arg->co == nullptr) {
        auto transa_char = (arg->transa != do_trans) ? "N" : "T";
        auto transb_char = (arg->transb != do_trans) ? "N" : "T";
        return jit_avx512_core_gemm_smalln_tn_f32(transa_char, transb_char,
                &arg->m, &arg->n, &arg->k, &arg->alpha, (float *)arg->a,
                &arg->lda, (float *)arg->b, &arg->ldb, &arg->beta,
                (float *)arg->c, &arg->ldc);
    }
    return dnnl_unimplemented;
}

template <>
dnnl_status_t jump_to_gemm_smalln_tn(
        const gemm_info_t<bfloat16_t, bfloat16_t, float> *arg) {
    return dnnl_unimplemented;
}

template <>
dnnl_status_t jump_to_gemm_smalln_tn(
        const gemm_info_t<int8_t, uint8_t, int32_t> *arg) {
    return dnnl_unimplemented;
}

template <>
dnnl_status_t jump_to_gemm_smalln_tn(
        const gemm_info_t<int8_t, int8_t, int32_t> *arg) {
    return dnnl_unimplemented;
}

dnnl_status_t sgemm_smalln_tn(const dim_t m, const dim_t n, const dim_t k,
        const float alpha, const float *A, const dim_t lda, const float *B,
        const dim_t ldb, const float beta, float *C, const dim_t ldc) {
    using namespace avx512_core_gemm_smalln_tn_f32;

    static xbyak_gemm_smalln_tn_t *kernels[4][3][3];
    static std::once_flag initialized;

    dnnl_status_t st = dnnl_success;
    std::call_once(initialized, [&] {
        for (dim_t N : {1, 2, 3, 4}) {
            for (float al : {0.0f, 1.0f, 2.0f}) {
                for (float be : {0.0f, 1.0f, 2.0f}) {
                    auto &kern = kernels[N - 1][(dim_t)al][(dim_t)be];
                    kern = new xbyak_gemm_smalln_tn_t(N, be, al);
                    st = kern->create_kernel();
                    if (st != dnnl_success) return;
                }
            }
        }
    });

    if (st != dnnl_success) return st;

    for (dim_t ii = 1; ii < 6; ii++) {
        dim_t nnval = partitions[n - 1][ii] - partitions[n - 1][ii - 1];
        dim_t nind = partitions[n - 1][ii - 1];
        if (nnval == 0 || m == 0) break;
        dim_t al_ind = (alpha == 0.0) ? 0 : (alpha == 1.0) ? 1 : 2;
        dim_t be_ind = (beta == 0.0) ? 0 : (beta == 1.0) ? 1 : 2;
        (*kernels[nnval - 1][al_ind][be_ind])(m, k, &alpha, &beta, A, lda,
                &B[nind * ldb], ldb, &C[nind * ldc], ldc);
    }

    return dnnl_success;
}

#define MROW_ALIGN 1
#define MINROWS 16 // Min rows each thread should process

static dim_t smalln_set_num_threads(dim_t m, dim_t k, dim_t nthr_to_use) {
    /**
     * Simple heuristics for determining num threads to use
     */

    // For very small sizes, don't parallelize
    if (m * k <= 8192) return 1;

    int nthr_16;
    if ((m & 15) == 0) { // Special handling if m is multiple of 16
        // nthr_16: number of threads such that each thread works on
        // 2^n * 16 number of rows.
        nthr_16 = m >> 4;
        while (nthr_16 > nthr_to_use && (nthr_16 & 1) == 0)
            nthr_16 = nthr_16 >> 1;
        // Ideal number of threads is more than what we can use.
        nthr_16 = (nthr_16 > nthr_to_use) ? nthr_to_use : nthr_16;
        /**
         * Check if nthr_16 should be used or nthr_to_use
         * If each thread is working on less than MINROWS rows (based on nthr_16)
         * use nthr_16. Each thread should be working on at least MINROWS rows.
         */
        bool use_orig_nthr = (m / nthr_16 <= MINROWS)
                ? !(nthr_16 <= nthr_to_use)
                : (4 * nthr_16 <= 3 * nthr_to_use);
        nthr_to_use = use_orig_nthr ? nthr_to_use : nthr_16;

    } else
        // Make sure each thread processes at least MINROWS rows.
        while (m / nthr_to_use < MINROWS && nthr_to_use > 1)
            nthr_to_use--;

    return nthr_to_use;
}

dnnl_status_t jit_avx512_core_gemm_smalln_tn_f32(const char *transa,
        const char *transb, const dim_t *p_m, const dim_t *p_n,
        const dim_t *p_k, const float *p_alpha, const float *A,
        const dim_t *p_lda, const float *B, const dim_t *p_ldb,
        const float *p_beta, float *C, const dim_t *p_ldc) {
    using namespace avx512_core_gemm_smalln_tn_f32;

    int max_num_threads = (dnnl_in_parallel()) ? 1 : dnnl_get_max_threads();

    dim_t m = *p_m;
    dim_t n = *p_n;
    dim_t k = *p_k;
    dim_t lda = *p_lda;
    dim_t ldb = *p_ldb;
    dim_t ldc = *p_ldc;
    float beta = *p_beta;
    float alpha = *p_alpha;

    if (n <= 0 || m <= 0) return dnnl_success;

    max_num_threads = smalln_set_num_threads(m, k, max_num_threads);

    if (max_num_threads == 1) {
        return sgemm_smalln_tn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    std::atomic<dnnl_status_t> st(dnnl_success);
    parallel(max_num_threads, [&](int ithr, int nthr) {
        dim_t mid = (m / nthr) & (~(MROW_ALIGN - 1));
        dim_t mpart = (ithr < nthr - 1) ? mid : m - mid * (nthr - 1);
        auto st_thr = sgemm_smalln_tn(mpart, n, k, alpha, &A[ithr * lda * mid],
                lda, B, ldb, beta, &C[ithr * mid], ldc);
        if (st_thr != dnnl_success) st = st_thr;
    });
    return st;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
