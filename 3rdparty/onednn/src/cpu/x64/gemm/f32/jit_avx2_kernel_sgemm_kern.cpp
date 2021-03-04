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

#include "cpu/x64/gemm/f32/jit_avx2_kernel_sgemm_kern.hpp"

#ifdef _WIN32
static const bool is_windows = true;
#else
static const bool is_windows = false;
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

int jit_avx2_kernel_sgemm_kern::next_acc(int idx, int um, int un) {
    while (!(((idx / unroll_n_) < std::max(1, um / nelt_per_vecreg_))
            || ((idx % unroll_n_) < un)))
        idx++;
    return idx;
}

void jit_avx2_kernel_sgemm_kern::prefetchB_beforeBload(
        int um, int un, int k_idx, int n_idx) {
    if (!mayiuse(avx512_core)) {
        if ((n_idx == 0) && (k_idx == 0) && (un == unroll_n_) && (um != 16)) {
            prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
            offb_ += 16;
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchB_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    if (!mayiuse(avx512_core)) {
        if ((um == 16) || (un < unroll_n_)) {
            if ((k_idx + m_idx + n_idx) == 0) {
                prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
                offb_ += 16;
            }
            if ((um == 16) && (un == 4) && (k_idx == 2)
                    && ((m_idx + n_idx) == 0)) {
                prefetcht0(ptr[BO_ + elt_size_ * (PREFETCHSIZEB_ + offb_)]);
                offb_ += 16;
            }
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchA_afterFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    if (mayiuse(avx512_core)) {
        if ((um < unroll_m_) && (m_idx == 0)) {
            if (((k_idx % (nb_zmm_a_ / unroll_m_reg_) == 0) && (n_idx % 6 == 0))
                    || ((k_idx % (nb_zmm_a_ / unroll_m_reg_) == 1)
                            && (n_idx == 3))) {
                prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                off_ += 16;
            }
        }
    } else {
        if (un == unroll_n_) {
            if (((um < nelt_per_vecreg_) && (n_idx == 0)
                        && (k_idx == std::min(2, nelt_per_vecreg_ / um - 1)))
                    || ((um == nelt_per_vecreg_) && (un == unroll_n_)
                            && (n_idx == 1) && (k_idx == 0))) {
                prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                off_ += 16;
            }
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchA_afterBload(
        int um, int un, int k_idx, int n_idx) {
    if (!mayiuse(avx512_core)) {
        if ((um == unroll_m_) && (un == 2)) {
            if (k_idx % 3 == 0) {
                if (n_idx == 1) {
                    if (k_idx == 0) off_ += 16;
                    prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    off_ += 16;
                }
                if ((k_idx == 0) && (n_idx == 0)) {
                    prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    off_ += 16;
                }
            } else {
                if (n_idx == 1) {
                    prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                    off_ += 16;
                }
            }
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchB_afterFMA(
        int k_idx, int n_idx, int m_idx) {
    if (mayiuse(avx512_core)) {
        if (((m_idx + (k_idx % (nb_zmm_a_ / unroll_m_reg_)) * unroll_m_reg_)
                    == 0)
                && (n_idx == 1)) {
            prefetcht0(ptr[BO_
                    + elt_size_
                            * (PREFETCHSIZEB_
                                    + nelt_per_vecreg_ * k_idx
                                            / (nb_zmm_a_ / unroll_m_reg_))]);
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchA_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    if (!mayiuse(avx512_core)) {
        if ((um == unroll_m_) && (un == unroll_n_)) {
            if (((k_idx == 0) && (n_idx % 2 == 1) && (m_idx == 0))
                    || ((k_idx == 1) && (n_idx == 2) && (m_idx == 0))
                    || ((k_idx == 2) && (n_idx == 0) && (m_idx == 2))
                    || ((k_idx == 2) && (n_idx == 3) && (m_idx == 0))
                    || ((k_idx == 3) && (n_idx == 1) && (m_idx == 0))) {
                prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                off_ += 16;
            }
        }
        if ((um == unroll_m_) && (un == 1)) {
            if (m_idx == 2) {
                prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                off_ += 16;
            } else if ((m_idx == 0) && ((k_idx == 1) || (k_idx == 2))) {
                prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
                off_ += 16;
            }
        }
        if ((um == 16) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 2)) {
            prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
            off_ += 16;
        }
        if ((um == 8) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 1)
                && (k_idx == 2)) {
            prefetcht0(ptr[AO_ + elt_size_ * (PREFETCHSIZEA_ + off_)]);
            off_ += 16;
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchC_afterBload(
        int um, int un, int k_idx, int n_idx) {
    if (mayiuse(avx512_core)) {
        if (um == unroll_m_) {
            if (n_idx == std::min(1, un - 1)) {
                if (k_idx == unroll_k_ - 1)
                    lea(CO2_, ptr[CO2_ + LDC_]);
                else
                    prefetchw(ptr[CO2_ + elt_size_ * k_idx * nelt_per_vecreg_]);
            }
        }
    }
}

void jit_avx2_kernel_sgemm_kern::prefetchC_beforeKloop(int um) {
    if (mayiuse(avx512_core)) {
        if (um < unroll_m_) {
            prefetchw(ptr[CO2_ + elt_size_ * 0]);
            prefetchw(ptr[CO2_ + elt_size_ * 8]);
            if (um <= 16) prefetchw(ptr[CO2_ + elt_size_ * 16]);
            lea(CO2_, ptr[CO2_ + LDC_]);
        }
    } else {
        prefetcht2(ptr[AA_ - 16 * elt_size_]);

        prefetcht0(ptr[CO1_ + 7 * elt_size_]);
        prefetcht0(ptr[CO1_ + LDC_ + 7 * elt_size_]);
        prefetcht0(ptr[CO2_ + 7 * elt_size_]);
        prefetcht0(ptr[CO2_ + LDC_ + 7 * elt_size_]);

        prefetcht0(ptr[CO1_ + 23 * elt_size_]);
        prefetcht0(ptr[CO1_ + LDC_ + 23 * elt_size_]);
        prefetcht0(ptr[CO2_ + 23 * elt_size_]);
        prefetcht0(ptr[CO2_ + LDC_ + 23 * elt_size_]);

        add(LL_, second_fetch_);

        prefetcht2(ptr[AA_]);
    }
}

void jit_avx2_kernel_sgemm_kern::generate() {

    int i, unroll_x, unroll_y, uy_bin, ux_bin;
    int C_off = is_windows ? 56 : 8;
    int LDC_off = is_windows ? 64 : 16;
    int sepload = 0;

    Xbyak::Label unroll_x_label[MAX_UNROLL_M],
            unroll_y_label[(MAX_UNROLL_N_BIN + 1) * MAX_UNROLL_M];
    Xbyak::Label end_n_loop_label[MAX_UNROLL_M], end_m_loop_label;

    preamble();

    if (is_windows) {
        mov(M_, ptr[rcx]);
        mov(N_, ptr[rdx]);
        mov(K_, ptr[r8]);
        mov(A_, ptr[rsp + get_size_of_abi_save_regs() + 40]);
        mov(B_, ptr[rsp + get_size_of_abi_save_regs() + 48]);
    } else {
        mov(M_, ptr[M_]);
        mov(N_, ptr[N_]);
        mov(K_, ptr[K_]);
    }

    mov(C_, ptr[rsp + get_size_of_abi_save_regs() + C_off]);
    mov(LDC_, ptr[rsp + get_size_of_abi_save_regs() + LDC_off]);

    if (mayiuse(avx512_core)) {
        for (i = zmm_acc_idx_; i < unroll_m_reg_ * unroll_n_ + zmm_acc_idx_;
                i++)
            vpxorq(Xbyak::Zmm(i), Xbyak::Zmm(i), Xbyak::Zmm(i));
    }

    sub(A_, -addr_off_ * elt_size_);
    sub(B_, -addr_off_ * elt_size_);

    sal(LDC_, elt_size_bin_);

    for (unroll_x = unroll_m_, i = 0, ux_bin = unroll_m_bin_; unroll_x >= 1;
            unroll_x -= std::min(nelt_per_vecreg_, std::max(1, unroll_x / 2)),
        i++, ux_bin--) {

        if (unroll_x == unroll_m_) {
            mov(J_, M_);
            cmp(J_, unroll_m_);
            jl(unroll_x_label[i + 1], T_NEAR);
            L_aligned(unroll_x_label[i]);
        } else {
            L_aligned(unroll_x_label[i]);
            test(J_, unroll_x);
            if (unroll_x > 1)
                jle(unroll_x_label[i + 1], T_NEAR);
            else
                jle(end_m_loop_label, T_NEAR);
        }

        mov(AA_, KK_);

        if ((1 << ux_bin) > unroll_x)
            imul(AA_, AA_, unroll_x * elt_size_);
        else
            sal(AA_, elt_size_bin_ + ux_bin);

        add(AA_, A_);
        mov(CO1_, C_);

        if ((unroll_x == unroll_m_) || (!mayiuse(avx512_core)))
            lea(CO2_, ptr[C_ + LDC_ * 2]);

        add(C_, unroll_x * elt_size_);
        mov(BO_, B_);

        for (unroll_y = unroll_n_, uy_bin = unroll_n_bin_; unroll_y >= 1;
                unroll_y /= 2, uy_bin--) {

            if (unroll_y == unroll_n_) {
                mov(I_, N_);
                sar(I_, uy_bin);
                jle(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1],
                        T_NEAR);
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
            } else {
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
                test(N_, unroll_y);
                if (uy_bin == 0)
                    jle(end_n_loop_label[i], T_NEAR);
                else
                    jle(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1],
                            T_NEAR);
            }

            if (!mayiuse(avx512_core))
                prefetcht2(ptr[AA_ - addr_off_ * elt_size_]);

            switch (unroll_x) {
                case 8:
                    if (mayiuse(avx512_core)) {
                        loop<Xbyak::Zmm, Xbyak::Zmm, Xbyak::Address, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vbroadcastf64x4,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        update<Xbyak::Ymm, Xbyak::Operand>(unroll_x, unroll_y,
                                0, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vmovups);
                    } else {
                        loop<Xbyak::Ymm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        update<Xbyak::Ymm, Xbyak::Operand>(unroll_x, unroll_y,
                                1, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vmovups);
                    }

                    break;
                case 4:
                    if (mayiuse(avx512_core)) {
                        loop<Xbyak::Zmm, Xbyak::Ymm, Xbyak::Address, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vbroadcastf32x4,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        sepload = 0;
                    } else {
                        loop<Xbyak::Xmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        sepload = 1;
                    }

                    update<Xbyak::Xmm, Xbyak::Operand>(unroll_x, unroll_y,
                            sepload, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                            &Xbyak::CodeGenerator::vmovups,
                            &Xbyak::CodeGenerator::vmovups);

                    break;
                case 2:
                    if (mayiuse(avx512_core)) {
                        loop<Xbyak::Zmm, Xbyak::Ymm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vbroadcastsd,
                                &Xbyak::CodeGenerator::vbroadcastss);
                    } else {
                        loop<Xbyak::Xmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovddup,
                                &Xbyak::CodeGenerator::vbroadcastss);
                    }
                    update<Xbyak::Xmm, Xbyak::Address>(unroll_x, unroll_y, 1,
                            beta_zero_, &Xbyak::CodeGenerator::vaddps,
                            &Xbyak::CodeGenerator::vmovlps,
                            &Xbyak::CodeGenerator::vmovsd);
                    break;
                case 1:
                    if (mayiuse(avx512_core)) {
                        loop<Xbyak::Zmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vbroadcastss,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        sepload = 0;
                    } else {
                        loop<Xbyak::Xmm, Xbyak::Xmm, Xbyak::Address, Xbyak::Xmm,
                                Xbyak::Address>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovss,
                                &Xbyak::CodeGenerator::vmovss);
                        sepload = 1;
                    }
                    update<Xbyak::Xmm, Xbyak::Address>(unroll_x, unroll_y,
                            sepload, beta_zero_, &Xbyak::CodeGenerator::vaddss,
                            &Xbyak::CodeGenerator::vmovss,
                            &Xbyak::CodeGenerator::vmovss);

                    break;
                default:
                    if (mayiuse(avx512_core)) {
                        loop<Xbyak::Zmm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        update<Xbyak::Zmm, Xbyak::Operand>(unroll_x, unroll_y,
                                0, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vmovups);
                    } else {
                        loop<Xbyak::Ymm, Xbyak::Xmm, Xbyak::Operand, Xbyak::Xmm,
                                Xbyak::Operand>(unroll_x, unroll_y,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vbroadcastss);
                        update<Xbyak::Ymm, Xbyak::Operand>(unroll_x, unroll_y,
                                1, beta_zero_, &Xbyak::CodeGenerator::vaddps,
                                &Xbyak::CodeGenerator::vmovups,
                                &Xbyak::CodeGenerator::vmovups);
                    }

                    break;
            }

            if (mayiuse(avx512_core)) {
                sub(AA_, -16 * elt_size_);
            } else {
                if ((unroll_y != unroll_n_) || (unroll_x <= 4)) {
                    if (unroll_x == unroll_m_)
                        sub(AA_, -16 * elt_size_);
                    else
                        sub(AA_, -32 * elt_size_);
                } else
                    sub(AA_, -48 * elt_size_);
            }

            if (unroll_y == unroll_n_) {
                dec(I_);
                jg(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin], T_NEAR);
            }
        }

        L_aligned(end_n_loop_label[i]);

        mov(A_, AO_);

        if (unroll_x == unroll_m_) {
            sub(J_, unroll_x);
            cmp(J_, unroll_x);
            jge(unroll_x_label[i], T_NEAR);
        }
    }

    L_aligned(end_m_loop_label);

    postamble();
}

jit_avx2_kernel_sgemm_kern::jit_avx2_kernel_sgemm_kern(bool beta_zero)
    : jit_generator(nullptr, 65536) {

    beta_zero_ = beta_zero;
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
