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

#include "cpu/x64/gemm/s8x8s32/jit_avx512_core_kernel_gemv_s8x8s32_kern.hpp"

#ifdef _WIN32
#define is_windows 1
#else
#define is_windows 0
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

void jit_avx512_core_gemv_s8x8s32_kern::vnni(
        Zmm acc, Zmm a, Zmm b, vnni_op_t op) {
    if (isa == avx512_core_vnni) {
        if (op == vnni_op_t::sub) vxorps(acc, acc, zmm_1_u1); // acc = -acc

        if (ver == ver_t::u8s8)
            vpdpbusd(acc, a, b);
        else
            vpdpbusd(acc, b, a);

        if (op == vnni_op_t::sub) vxorps(acc, acc, zmm_1_u1); // acc = -acc
    } else {
        assert(isa == avx512_core);

        if (ver == ver_t::u8s8)
            vpmaddubsw(zmm_tmp, a, b);
        else
            vpmaddubsw(zmm_tmp, b, a);
        vpmaddwd(zmm_tmp, zmm_tmp, zmm_1_s16);

        if (op == vnni_op_t::sub)
            vpsubd(acc, acc, zmm_tmp);
        else
            vpaddd(acc, zmm_tmp, acc);
    }
}

void jit_avx512_core_gemv_s8x8s32_kern::n_loop_body(int nreg_acc, Reg64 A,
        Reg64 lda, Reg64 X, int use_mask, Opmask mask_n) {
    const int nreg_A = nreg_acc / 2 + (nreg_acc % 2);

    // load X + j
    if (use_mask)
        vmovdqu8(zmm_b | mask_n | T_z, ptr[X]);
    else
        vmovdqu8(zmm_b, ptr[X]);
    if (ver == ver_t::s8s8) vxorps(zmm_b, zmm_b, zmm_128_u8);

    xor_(r14, r14);
    // load values of A
    for (int i = 0; i < nreg_A; i++) {
        if (use_mask)
            vmovdqu8(zmm_a(i) | mask_n | T_z, ptr[A + r14]);
        else
            vmovdqu8(zmm_a(i), ptr[A + r14]);
        add(r14, lda);
    }

    for (int i = 0; i < nreg_A; i++)
        vnni(zmm_acc(i), zmm_a(i), zmm_b, vnni_op_t::add);

    if (ver == ver_t::s8s8)
        for (int i = 0; i < nreg_A; i++)
            vnni(zmm_acc(i), zmm_a(i), zmm_128_u8, vnni_op_t::sub);

    for (int i = 0; i < nreg_A - (nreg_acc % 2); i++) {
        if (use_mask)
            vmovdqu8(zmm_a(i) | mask_n | T_z, ptr[A + r14]);
        else
            vmovdqu8(zmm_a(i), ptr[A + r14]);
        add(r14, lda);
    }

    for (int i = 0; i < nreg_A - (nreg_acc % 2); i++)
        vnni(zmm_acc(nreg_A + i), zmm_a(i), zmm_b, vnni_op_t::add);

    if (ver == ver_t::s8s8)
        for (int i = 0; i < nreg_A - (nreg_acc % 2); i++)
            vnni(zmm_acc(nreg_A + i), zmm_a(i), zmm_128_u8, vnni_op_t::sub);
}

void jit_avx512_core_gemv_s8x8s32_kern::shuffle_and_add(
        Zmm dest, Zmm A, Zmm B, Zmm C, Zmm D) {
    vshufi32x4(dest, A, C, 0x44);
    vshufi32x4(A, A, C, 0xEE);
    vpaddd(C, dest, A); // C = A0 + A2|A1 + A3|C0 + C2|C1 + C3

    vshufi32x4(dest, B, D, 0x44);
    vshufi32x4(B, B, D, 0xEE);
    vpaddd(D, dest, B); // D = B0 + B2|B1 + B3|D0 + D2|D1 + D3

    vshufi32x4(A, C, D, 0x88);
    vshufi32x4(B, C, D, 0xDD);
    vpaddd(dest, A, B); // dest = SAi|SBi|SCi|SDi
}

void jit_avx512_core_gemv_s8x8s32_kern::update_c(
        int nreg_acc, Reg64 Y, int use_mask, Opmask mask_m) {
    int l, i, k, j, last_it;
    Label store_label;

    auto ymm_c = [&](int i) { return Ymm(zmm_a(i).getIdx()); };
    auto zmm_c = [&](int i) { return zmm_a(i); };

    l = 0;
    for (k = 0; k < nreg_acc; k += 8) {
        for (i = 0, j = k; i < 8; i += 4, j += 2) {
            if (j < nreg_acc) {
                // shuffle per block of 4 registers
                shuffle_and_add(zmm_c(l), // desc
                        zmm_acc(j), // A = acc0
                        zmm_acc(1 + j), // B = acc1
                        zmm_acc(4 + j), // C = acc4
                        zmm_acc(5 + j)); // D = acc5

                // extract low and high from dest and hadd
                vextracti32x8(ymm_c(l + 1), zmm_c(l), 0);
                vextracti32x8(ymm_c(l + 2), zmm_c(l), 1);
                vphaddd(ymm_c(l), ymm_c(l + 1), ymm_c(l + 2));
            }
            l++;
        }

        vphaddd(ymm_c(l), ymm_c(l - 2), ymm_c(l - 1));
        l++;
    }

    // eventually add with C and store new value
    Xmm xmm_zero = Xmm(zmm_tmp.getIdx());
    vxorps(xmm_zero, xmm_zero, xmm_zero);
    vucomiss(xmm_beta, xmm_zero);
    je(store_label, T_NEAR);

    // beta = 1
    for (k = 0, l = 2; k < nreg_acc; k += 8, l += 3) {
        // load Y and add
        last_it = (k + 8) > nreg_acc;
        if (use_mask && last_it)
            vmovdqu32(ymm_c(k / 8) | mask_m | T_z, ptr[Y + (k / 8) * 32]);
        else
            vmovdqu32(ymm_c(k / 8), ptr[Y + (k / 8) * 32]);

        vpaddd(ymm_c(l), ymm_c(l), ymm_c(k / 8));
    }

    // store
    L_aligned(store_label);
    for (k = 0, l = 2; k < nreg_acc; k += 8, l += 3) {
        last_it = (k + 8) > nreg_acc;
        if (use_mask && last_it)
            vmovdqu32(ptr[Y + (k / 8) * 32], ymm_c(l) | mask_m);
        else
            vmovdqu32(ptr[Y + (k / 8) * 32], ymm_c(l));
    }
}

void jit_avx512_core_gemv_s8x8s32_kern::generate() {

    const int vec_len = 64; // bytes

    isa = mayiuse(avx512_core_vnni) ? avx512_core_vnni : avx512_core;

    assert(ver != ver_t::undef);

    Opmask mask_n = k1, mask_m = k2;
    Label one_label, m_tail_label, m_loop_label, n_loop_label;
    Label n_tail_label, update_c_label, end_label;
    constexpr unsigned int n_labels = (1 << unroll_m_) - 1;
    Label m_tail_label_case[n_labels];
    Label n_loop_label_case[n_labels];
    Label n_tail_label_case[n_labels];
    Label update_c_label_case[n_labels];

    Reg64 n = abi_param2, m = abi_param1;
    Reg64 A = is_windows ? abi_param4 : abi_param3;
    Reg64 lda = is_windows ? abi_param3 : abi_param4;
    Reg64 X = is_windows ? rdi : r8;
    Reg64 Y = is_windows ? rsi : r9;

    preamble();

    if (is_windows) {
        // Windows: read on the stack lda, X, beta, Y
        mov(lda, ptr[rsp + get_size_of_abi_save_regs() + 40]);
        mov(X, ptr[rsp + get_size_of_abi_save_regs() + 48]);
        movss(xmm_beta, ptr[rsp + get_size_of_abi_save_regs() + 56]);
        mov(Y, ptr[rsp + get_size_of_abi_save_regs() + 64]);
    }

    mov(rax, (1 << unroll_n_) - 1);
    kmovq(k3, rax);

    and_(rax, n); // rax contains n & ((1 << unroll_n_) - 1)
    mov(rbx, 1);
    shlx(rbx, rbx, rax);
    sub(rbx, 1);
    kmovq(mask_n, rbx);
    // mask_n set (AVX512 only), can use rax and rbx again

    // set mask_m for update of the C matrix
    // load/store on the C matrix use Ymm so tail according to Ymm size
    mov(rax, 7); // 8 * 32 = 256 Ymm size
    and_(rax, m); // rax contains m & 7
    mov(rbx, 1);
    shlx(rbx, rbx, rax);
    sub(rbx, 1);
    kmovq(mask_m, rbx);
    // mask_m set (AVX512 only), can use rax and rbx again

    // setup const registers
    if (isa == avx512_core)
        vmovdqu16(zmm_1_s16, ptr[rip + one_label + 0 * vec_len]);
    if (isa == avx512_core_vnni && ver == ver_t::s8s8)
        vmovdqu16(zmm_1_u1, ptr[rip + one_label + 1 * vec_len]);
    if (ver == ver_t::s8s8)
        vmovdqu16(zmm_128_u8, ptr[rip + one_label + 2 * vec_len]);

    assert(zmm_1_s16 == zmm_1_u1);

    // M loop
    // base pointer for A rax contains a + i * lda
    // Loop stop when rax >= a + (m & mask_um) * lda = rbx
    // loop increment r10 = um * lda
    // rbp = Y + i
    const int mask_um = 0xFFFFFFF0;
    mov(rax, A); // i = 0
    mov(rbx, m);
    and_(rbx, mask_um);
    imul(rbx, lda);
    add(rbx, A);
    mov(r10, lda);
    sal(r10, unroll_m_);
    mov(rbp, Y);

    // N loop
    // base pointer for X r11 contains x + j
    // Loop stop when r11 >= x + n & mask_un = r12
    // loop increment un
    // r13 = rax + j = A + i * lda + j
    const int mask_un = 0xFFFFFFC0;
    mov(r12, n);
    and_(r12, mask_un);
    add(r12, X);

    // M loop
    L_aligned(m_loop_label);
    cmp(rax, rbx);
    jge(m_tail_label, T_NEAR);

    // enter M loop
    for (int i = 0; i < zmm_acc_idx_count; i++)
        vpxorq(zmm_acc(i), zmm_acc(i), zmm_acc(i));

    // N loop
    mov(r11, X); // j = 0
    mov(r13, rax);
    L_aligned(n_loop_label);
    cmp(r11, r12);
    jge(n_tail_label, T_NEAR);

    // enter N loop

    n_loop_body(zmm_acc_idx_count, r13, lda, r11, 0, mask_n);

    // increment rax with un
    add(r11, 1 << unroll_n_);
    add(r13, 1 << unroll_n_);
    jmp(n_loop_label, T_NEAR);
    // end N loop

    // N tail
    L_aligned(n_tail_label);

    ktestq(mask_n, k3);
    je(update_c_label, T_NEAR);
    n_loop_body(zmm_acc_idx_count, r13, lda, r11, 1, mask_n);

    // update C matrix
    L_aligned(update_c_label);

    update_c(zmm_acc_idx_count, rbp, 0, mask_m);

    // increment rax with um * lda
    add(rax, r10);
    add(rbp, 1 << (unroll_m_ + 2));
    jmp(m_loop_label, T_NEAR);
    // end M loop

    // M tail
    L_aligned(m_tail_label);

    // r10 will contain m_tail = m % unroll_m_ = m & (1 << unroll_m_) - 1
    mov(r10, m);
    and_(r10, (1 << unroll_m_) - 1);
    for (int ii = 1; ii < 1 << unroll_m_; ii++) {
        L_aligned(m_tail_label_case[ii - 1]);
        cmp(r10, ii);
        if (ii == (1 << unroll_m_) - 1)
            jne(end_label, T_NEAR);
        else
            jne(m_tail_label_case[ii], T_NEAR);

        // m_tail = i, use i accumulators

        for (int i = 0; i < ii; i++)
            vpxorq(zmm_acc(i), zmm_acc(i), zmm_acc(i));

        // N loop
        mov(r11, X); // j = 0
        mov(r13, rax);
        L_aligned(n_loop_label_case[ii - 1]);
        cmp(r11, r12);
        jge(n_tail_label_case[ii - 1], T_NEAR);

        n_loop_body(ii, r13, lda, r11, 0, mask_n);

        // increment rax with un
        add(r11, 1 << unroll_n_);
        add(r13, 1 << unroll_n_);
        jmp(n_loop_label_case[ii - 1], T_NEAR);
        // end N loop

        // N tail
        L_aligned(n_tail_label_case[ii - 1]);
        ktestq(mask_n, k3);
        je(update_c_label_case[ii - 1], T_NEAR);
        n_loop_body(ii, r13, lda, r11, 1, mask_n);

        // update C matrix
        L_aligned(update_c_label_case[ii - 1]);
        update_c(ii, rbp, 1, mask_m);

        if (ii < ((1 << unroll_m_) - 1)) jmp(end_label, T_NEAR);
    }

    L_aligned(end_label);

    postamble();

    L_aligned(one_label);
    for (int i = 0; i < vec_len / 2; i++) // 1_s16
        dw((int16_t)0x0001);
    for (int i = 0; i < vec_len / 2; i++) // 1_u1
        dw((int16_t)0xffff);
    for (int i = 0; i < vec_len / 2; i++) // 128_u8
        dw((int16_t)0x8080);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
