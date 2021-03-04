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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    using namespace Xbyak::util;
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count();
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();

    // For avx we need a register to save the upper part of Ymm
    preserve_vec_for_avx = isa == avx
            && utils::one_of(alg_, eltwise_tanh, eltwise_elu, eltwise_abs,
                    eltwise_soft_relu, eltwise_logistic, eltwise_exp,
                    eltwise_gelu_tanh, eltwise_swish, eltwise_gelu_erf,
                    eltwise_tanh_use_dst_for_bwd, eltwise_elu_use_dst_for_bwd,
                    eltwise_logistic_use_dst_for_bwd,
                    eltwise_exp_use_dst_for_bwd);
    if (preserve_vec_for_avx) vecs_to_preserve++;

    // For sse41 mask register has to be Xmm(0)
    if (isa == sse41 && vecs_to_preserve > 0) {
        size_t idx = 0;
        assert(idx < start_idx);
        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    // Same logic but to allocate gprs
    size_t preserved_gprs_count = 0;
    for (size_t gpr_idx = 0; gpr_idx <= Operand::R15; ++gpr_idx) {
        int _idx = Operand::R15 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && !utils::one_of(_idx, p_table.getIdx(), Operand::RSP))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
    assert(preserved_gprs_count == aux_gprs_count());

    if (save_state_) {
        h->push(p_table);
        for (size_t i = 0; i < preserved_gprs_count; ++i)
            h->push(Reg64(preserved_gpr_idxs[i]));

        if (preserved_vecs_count) h->sub(h->rsp, preserved_vecs_count * vlen);

        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(
                    h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[i]));

        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
        const injector_utils::vmm_index_set_iterator_t start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off) h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));

        if (idx_off) h->sub(h->rsp, idx_off * vlen);
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    using namespace Xbyak::util;
    if (!save_state_) return;

    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]), h->ptr[h->rsp + i * vlen]);

    if (preserved_vecs_count) h->add(h->rsp, preserved_vecs_count * vlen);

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->pop(Reg64(preserved_gpr_idxs[i]));
    h->pop(p_table);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_aux2 = Vmm(preserved_vec_idxs[2]);
    vmm_aux3 = Vmm(preserved_vec_idxs[3]);
    vmm_aux4 = Vmm(preserved_vec_idxs[4]);
    if (preserve_vec_for_avx) {
        vmm_tmp = Vmm(preserved_vec_idxs[vecs_to_preserve - 1]);
        ymm_tmp = Ymm(preserved_vec_idxs[vecs_to_preserve - 1]);
        xmm_tmp = Xmm(preserved_vec_idxs[vecs_to_preserve - 1]);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::vec_shift(const Vmm &vmm_dst,
        const Vmm &vmm_src, bool shift_left, const int imm) {
    if (isa != avx) {
        if (shift_left)
            h->uni_vpslld(vmm_dst, vmm_src, imm);
        else
            h->uni_vpsrld(vmm_dst, vmm_src, imm);
    } else {
        // Declare appropriate vectors to use non-uni instructions
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Ymm ymm_src = Ymm(vmm_src.getIdx());
        if (vmm_dst.getIdx() != vmm_src.getIdx()) h->vmovups(ymm_dst, ymm_src);
        h->vextractf128(xmm_tmp, ymm_dst, 1);
        if (shift_left) {
            h->vpslld(xmm_dst, xmm_dst, imm);
            h->vpslld(xmm_tmp, xmm_tmp, imm);
        } else {
            h->vpsrld(xmm_dst, xmm_dst, imm);
            h->vpsrld(xmm_tmp, xmm_tmp, imm);
        }
        h->vinsertf128(ymm_dst, ymm_dst, xmm_tmp, 1);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Stores a mask by applying cmpps on two inputs w/ a given predicate.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(const Vmm &vmm_src,
        const Xbyak::Operand &compare_operand, int cmp_predicate) {
    if (has_avx512()) {
        h->vcmpps(k_mask, vmm_src, compare_operand, cmp_predicate);
    } else {
        h->uni_vcmpps(vmm_mask, vmm_src, compare_operand, cmp_predicate);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const Vmm &vmm_dst, const Xbyak::Operand &src) {
    if (has_avx512()) {
        h->vblendmps(vmm_dst | k_mask, vmm_dst, src);
    } else {
        h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Tests a mask for all zeros. If all zeroes occur, set ZF = 1.
// Nicely combines with jump_if_zero (jz).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::test_mask() {
    if (has_avx512()) {
        h->kortestw(k_mask, k_mask);
    } else {
        h->uni_vtestps(vmm_mask, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2);

    // x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, table_val(ln2f));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    if (isa != avx)
        h->uni_vpaddd(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    else {
        Ymm ymm_aux2 = Ymm(vmm_aux2.getIdx());
        Xmm xmm_aux2 = Xmm(vmm_aux2.getIdx());
        h->vextractf128(xmm_tmp, ymm_aux2, 1);
        h->vpaddd(xmm_tmp, xmm_tmp, table_val(exponent_bias));
        h->vpaddd(xmm_aux2, xmm_aux2, table_val(exponent_bias));
        h->vinsertf128(ymm_aux2, ymm_aux2, xmm_tmp, 1);
    }
    vec_shift(vmm_aux2, vmm_aux2, true, n_mantissa_bits); //Vmm(6) = 2^-fx
    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vxorps(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // compute polynomial
    h->uni_vmovups(vmm_src, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux1, vmm_src);
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));

    // combine with mask
    compute_cmp_mask(vmm_aux3, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // we add a check as the avx2 code cannot be used for avx
    assert(IMPLICATION(isa == avx2, mayiuse(avx2)));

    using namespace Xbyak::util;
    const int XMM_float_lanes_count = 4;
    const int tanh_n_polynomials = 32;

    // register mapping
    // TODO: put sign on stack and alias zmm_table2 with vmm_sign to save a reg ?
    Vmm vmm_dst = vmm_aux1, vmm_src_shift = vmm_aux1, vmm_coeff = vmm_aux1,
        vmm_pol = vmm_aux2, vmm_indices = vmm_aux3, vmm_src_original = vmm_aux4,
        vmm_sign = vmm_aux4;
    Reg64 gpr_idx[XMM_float_lanes_count];
    // tmp_reg to save the state of regs64 (for avx only).
    Reg64 tmp_reg;

    if (isa == sse41 || isa == avx) {
        assert(aux_gprs_count() >= XMM_float_lanes_count);
        for (int i = 0; i < XMM_float_lanes_count; i++)
            gpr_idx[i] = Reg64(preserved_gpr_idxs[i]);
    }

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.

    // The polynomials are of degree 6, so we need to gather 7 coefficients.
    // - sse4.1: we do it the naive way using vextract/vinsert.
    //           Here we will extract the indices in gpr only once and
    //           reuse them as there are only 4 of them.
    // - avx: we do the same as for sse4.1 but use half of the 64-bits
    //           registers to store the idx of second half of YMM and half for
    //           responding XMM. Halfway through the copy we exchange Xmm and
    //           higher half of Ymm and we get the expected result.
    // - avx2: we use vpermps and blend for each coefficient.
    //         This needs an extra vmm to store the mask
    // - avx512: because the table fits in 2 registers, we can use vpermi2d.
    auto coeffs_off = [&](int coeff_off, int off = 0) {
        return table_off(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto coeffs_address = [&](int coeff_off, int off = 0) {
        return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient_init = [&](Vmm vmm_pol_idx, int nelems) {
        switch (isa) {
            case sse41:
                for (int i = 0; i < XMM_float_lanes_count; ++i)
                    h->pextrd(gpr_idx[i].cvt32(), vmm_pol_idx, i);
                break;
            case avx: {
                Xmm xmm_pol_idx = Xmm(vmm_pol_idx.getIdx());
                for (int i = 0; i < XMM_float_lanes_count; ++i)
                    h->vpextrd(gpr_idx[i].cvt32(), xmm_pol_idx, i);
            } break;
            case avx2:
                // needed for gather instruction
                h->uni_vxorps(vmm_mask, vmm_mask, vmm_mask);
                break;
            case avx512_common:
            case avx512_core: break;
            default: assert(!"unimplemented");
        }
    };
    auto gather_coefficient = [&](Vmm vmm_coeff, int coeff_idx,
                                      Vmm vmm_pol_idx) {
        switch (isa) {
            case sse41:
                for (int idx = 0; idx < 4; ++idx) {
                    Xbyak::Address coeff_addr
                            = ptr[p_table + coeffs_off(coeff_idx)
                                    + gpr_idx[idx] * sizeof(float)];
                    h->pinsrd(vmm_coeff, coeff_addr, idx);
                }
                break;
            case avx: {
                Xmm xmm_coeff = Xmm(vmm_coeff.getIdx());
                for (int idx = 0; idx < 4; ++idx) {
                    Xbyak::Address coeff_addr
                            = ptr[p_table + coeffs_off(coeff_idx)
                                    + gpr_idx[idx] * sizeof(float)];
                    h->vpinsrd(xmm_coeff, xmm_coeff, coeff_addr, idx);
                }
            } break;
            case avx2: {
                Xbyak::Address idx_addr = ptr[p_table + coeffs_off(coeff_idx)
                        + vmm_pol_idx * sizeof(float)];
                // we set the mask to all ones to gather full
                // register.  needs to be done after each gather since
                // since the gather instructions zeros the mask if
                // successful
                h->uni_vcmpps(vmm_mask, vmm_mask, vmm_mask, _cmp_eq_oq);
                h->vgatherdps(vmm_coeff, idx_addr, vmm_mask);
                break;
            }
                // use gather instruction
            case avx512_common:
            case avx512_core:
                // we use vpermt2ps to not override the indices
                // this also enables to save a register for table loading
                {
                    Zmm zmm_coeff(vmm_coeff.getIdx());
                    Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
                    h->uni_vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
                    h->vpermt2ps(zmm_coeff, zmm_pol_idx,
                            coeffs_address(coeff_idx, 16));
                    break;
                }
            default: assert(!"unimplemented");
        }
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    h->uni_vmovups(vmm_src_original, vmm_src);
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // We compute the indices for the table lookup
    h->uni_vmovups(vmm_indices, vmm_src);
    if (isa != avx)
        h->uni_vpsubd(vmm_indices, vmm_indices, table_val(tanh_idx_bias));
    else {
        Ymm ymm_indices = Ymm(vmm_indices.getIdx());
        Xmm xmm_indices = Xmm(vmm_indices.getIdx());
        h->vextractf128(xmm_tmp, ymm_indices, 1);
        h->vpsubd(xmm_tmp, xmm_tmp, table_val(tanh_idx_bias));
        h->vpsubd(xmm_indices, xmm_indices, table_val(tanh_idx_bias));
        h->vinsertf128(ymm_indices, ymm_indices, xmm_tmp, 1);
    }
    h->uni_vandps(vmm_indices, vmm_indices, table_val(tanh_idx_mask));
    vec_shift(vmm_indices, vmm_indices, false, 22);

    // we do the argument reduction
    h->uni_vmovups(vmm_src_shift, vmm_src);
    h->uni_vandps(vmm_src_shift, vmm_src_shift, table_val(tanh_idx_mask));
    h->uni_vsubps(vmm_src, vmm_src, vmm_src_shift);

    // we gather and evaluate the polynonials
    gather_coefficient_init(vmm_indices, vlen / sizeof(float));
    gather_coefficient(vmm_pol, 6, vmm_indices);
    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(vmm_coeff, deg, vmm_indices);
        h->uni_vfmadd213ps(vmm_pol, vmm_src, vmm_coeff);
    }

    if (isa == avx) {
        Ymm ymm_indices = Ymm(vmm_indices.getIdx());
        Ymm ymm_pol = Ymm(vmm_pol.getIdx());
        Ymm ymm_src = Ymm(vmm_src.getIdx());
        Xmm xmm_src = Xmm(vmm_src.getIdx());
        Xmm xmm_coeff = Xmm(vmm_coeff.getIdx());

        h->vperm2f128(ymm_src, ymm_src, ymm_src, 1);
        h->vperm2f128(ymm_indices, ymm_indices, ymm_indices, 1);
        gather_coefficient_init(vmm_indices, vlen / sizeof(float));
        gather_coefficient(vmm_tmp, 6, vmm_indices);
        for (int deg = 5; deg >= 0; --deg) {
            gather_coefficient(vmm_coeff, deg, vmm_indices);
            h->vmulps(xmm_tmp, xmm_tmp, xmm_src);
            h->vaddps(xmm_tmp, xmm_tmp, xmm_coeff);
        }
        h->vinsertf128(ymm_pol, ymm_pol, xmm_tmp, 1);
    }

    // we restore src with cleared sign, and keep sign
    assert(vmm_sign.getIdx() == vmm_src_original.getIdx());
    h->uni_vmovups(vmm_src, vmm_src_original);
    h->uni_vandps(vmm_sign, vmm_sign, table_val(sign_mask));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    h->uni_vmovups(vmm_dst, table_val(one));
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    h->uni_vmovups(vmm_mask, table_val(tanh_saturation_lbound));
    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_pol);
    // [0; linear_ubound]  : we return x
    h->uni_vmovups(vmm_mask, table_val(tanh_linear_ubound));
    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_src);

    // We reapply the sign and return
    h->uni_vxorps(vmm_dst, vmm_dst, vmm_sign);
    h->uni_vmovups(vmm_src, vmm_dst);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // save x on stack as tanh uses vmm_aux0
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux0);

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vsqrtps(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(alpha));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(beta));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    h->uni_vminps(vmm_src, vmm_src, table_val(alpha));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vminps(vmm_src, vmm_src, table_val(beta));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // keep src for further computations
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0);

    // x = x - fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(ln2f));
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    h->uni_vmovups(vmm_aux3, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(one));

    // compute 2^(-n)
    if (has_avx512()) {
        h->vmulps(vmm_aux1, vmm_src, table_val(minus_one));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else if (isa == avx) {
        h->uni_vxorps(vmm_aux1, vmm_src, table_val(sign_mask));
        h->uni_vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(minus_one));
    }

    if (isa != avx)
        h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(exponent_bias));
    else {
        Ymm ymm_aux1 = Ymm(vmm_aux1.getIdx());
        Xmm xmm_aux1 = Xmm(vmm_aux1.getIdx());
        h->vextractf128(xmm_tmp, ymm_aux1, 1);
        h->vpaddd(xmm_tmp, xmm_tmp, table_val(exponent_bias));
        h->vpaddd(xmm_aux1, xmm_aux1, table_val(exponent_bias));
        h->vinsertf128(ymm_aux1, ymm_aux1, xmm_tmp, 1);
    }
    vec_shift(vmm_aux1, vmm_aux1, true, n_mantissa_bits); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // frexp()
    vec_shift(vmm_src, vmm_aux3, false, n_mantissa_bits);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six));

    // and with mask (to get 0.5 * mantissa)
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(soft_relu_mantissa_sign_mask));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(half));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(one));

    // compute log1p polynomial
    h->uni_vmovups(vmm_aux1, table_val(soft_relu_pol, 8));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 7));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 6));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 5));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 0));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(ln2f));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // we store the original sign and make x negative
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(sign_mask));
    h->uni_vorps(vmm_src, vmm_src, table_val(sign_mask));

    exp_compute_vector_fwd(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, table_val(one));
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    if (has_avx512()) {
        h->vptestmd(k_mask, vmm_aux3, vmm_aux3);
    } else {
        h->uni_vmovups(vmm_mask, vmm_aux3);
    }
    blend_with_mask(vmm_aux2, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
        const Vmm &vmm_src) {
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
    // log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates
    //   1 / y, i is index of one of precomputed values;
    // log(1 + z) ~~ polynomial(z), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(z) - log(r_i),
    // where log(r_i) is table value.
    //
    // If (x == 0) result = -inf;
    // If (x < 0) result = qnan;

    // set unused register as tmp for avx
    if (isa == avx) {
        ymm_tmp = Ymm(vmm_aux0.getIdx());
        xmm_tmp = Xmm(vmm_aux0.getIdx());
    }

    // save source on stack to check neg and zero values at the end
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // compute i
    const int approx_order = 5;
    vec_shift(vmm_aux1, vmm_src, false, n_mantissa_bits - approx_order);
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val(log_five_bit_offset));
    vec_shift(vmm_aux1, vmm_aux1, true, 1); // multiply i by 2

    // compute anticancellation i
    vec_shift(vmm_aux2, vmm_aux1, false, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    vec_shift(vmm_aux3, vmm_src, false, n_mantissa_bits);
    if (isa != avx)
        h->uni_vpaddd(vmm_aux3, vmm_aux3, vmm_aux2);
    else {
        Ymm ymm_aux2 = Ymm(vmm_aux2.getIdx());
        Ymm ymm_aux3 = Ymm(vmm_aux3.getIdx());
        Xmm xmm_aux2 = Xmm(vmm_aux2.getIdx());
        Xmm xmm_aux3 = Xmm(vmm_aux3.getIdx());
        h->vextractf128(xmm_tmp, ymm_aux3, 1);
        h->vpaddd(xmm_aux3, xmm_aux3, xmm_aux2);
        h->vperm2f128(ymm_aux2, ymm_aux2, ymm_aux2, 1);
        h->vpaddd(xmm_tmp, xmm_tmp, xmm_aux2);
        h->vperm2f128(ymm_aux2, ymm_aux2, ymm_aux2, 1);
        h->vinsertf128(ymm_aux3, ymm_aux3, xmm_tmp, 1);
    }
    h->uni_vcvtdq2ps(vmm_aux3, vmm_aux3);

    // get m (mantissa)
    h->uni_vxorps(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    vec_shift(vmm_aux2, vmm_aux2, true, n_mantissa_bits);
    h->uni_vandps(vmm_src, vmm_src, table_val(log_mantissa_mask));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux2);

    // At first, adjust indices for table structure which broadcasts elements
    // by multiplying by simd_w
    const int simd_w = math::ilog2q(
            vlen / sizeof(float)); // equal to 2/3/4 for xmm/ymm/zmm
    vec_shift(vmm_aux1, vmm_aux1, true, simd_w);

    const auto it = entry_map_.find(log_predefined_vals);
    assert(it != entry_map_.end());
    const auto table_start_idx = (*it).second.off;

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        Xbyak::Address table_idx = h->ptr[p_table + table_start_idx + offt
                + vmm_idxs * sizeof(float)];
        if (has_avx512()) {
            h->kmovw(k_mask, table_val(log_full_k_reg_mask));
            h->vgatherdps(vmm_dst | k_mask, table_idx);
        } else if (isa == avx2) {
            h->uni_vmovups(vmm_mask, table_val(sign_mask));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask);
        } else if (isa == avx || isa == sse41) {
            Xbyak::Reg64 reg_tmp
                    = p_table.getIdx() != h->r9.getIdx() ? h->r9 : h->r10;

            const int gpr_size = 8;
            // save reg_tmp state as we are not allowed to spoil it.
            h->sub(h->rsp, gpr_size);
            h->mov(h->ptr[h->rsp], reg_tmp);

            // rest of code puts indices on stack, fetching a table number based
            // on an index, replaces index with the value, and, finally, moves
            // fetched values into vector register.
            h->sub(h->rsp, vlen);
            h->uni_vmovups(h->ptr[h->rsp], vmm_idxs);

            for (size_t i = 0; i < vlen / sizeof(float); ++i) {
                h->mov(reg_tmp.cvt32(), h->ptr[h->rsp + i * sizeof(float)]);
                h->shl(reg_tmp.cvt32(), 2); // multiply by simd_w
                table_idx = h->ptr[p_table + table_start_idx + offt + reg_tmp];
                h->mov(reg_tmp.cvt32(), table_idx);
                h->mov(h->ptr[h->rsp + i * sizeof(float)], reg_tmp.cvt32());
            }

            h->uni_vmovups(vmm_dst, h->ptr[h->rsp]);
            h->add(h->rsp, vlen);
            // restore GPR state
            h->mov(reg_tmp, h->ptr[h->rsp]);
            h->add(h->rsp, gpr_size);
        }
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux2, vmm_aux1, 0);

    // compute relative error (rel_err = m * r_i - 1)
    h->uni_vfmsub213ps(vmm_aux2, vmm_src, table_val(one));

    // compute polynomial(rel_err)
    h->uni_vmovups(vmm_src, table_val(log_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * ln(2) - log(r_i))
    h->uni_vfmadd231ps(vmm_aux2, vmm_aux3, table_val(ln2f));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux1, vmm_src, vmm_aux2); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux3, vmm_aux1, vmm_aux2); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux1, vmm_aux3); // res_hi = pol + pres

    h->uni_vmovups(vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux2); // res_hi = pol + pres
    h->uni_vmovups(vmm_aux3, vmm_aux1);
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_aux2); // res_lo = res_hi - pres
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    h->uni_vmovups(vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux3); // res_hi = pol + pres

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    Xbyak::Label end_log_label;
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_le_os);
    test_mask();
    h->jz(end_log_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));

    h->L(end_log_label);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_fwd(
        const Vmm &vmm_src) {
    // dispatch between special cases.
    if (beta_ == -1) { // alpha / x
        h->uni_vmovups(vmm_aux0, table_val(alpha));
        h->uni_vdivps(vmm_src, vmm_aux0, vmm_src, vmm_aux0);
    } else if (beta_ == 0) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else if (beta_ == 0.5) { // alpha * sqrt(x)
        sqrt_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha * x
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 2) { // alpha * x^2
        square_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else { // general path
        // caller obligation to save gprs as callee may use them
        size_t gpr_size = 8;
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // caller obligation to save k-regs as callee may use them
        size_t n_k_regs_to_save = 8;
        if (has_avx512()) {
            h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
            for (size_t i = 0; i < n_k_regs_to_save; ++i) {
                if (mayiuse(avx512_core))
                    h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
                else
                    h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
            }
        }

        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. Additionally save space for vmm_src, to put the answer in-place on
        // this space and space for beta.
        // 3. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        h->sub(h->rsp, (vecs_count + 2) * vlen);
        for (size_t i = 2; i < vecs_count + 2; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen], Vmm(i - 2));
        h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_src); // src
        h->uni_vmovups(vmm_src, table_val(beta));
        h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_src); // beta

        // save function address in gpr to pass in in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

        // align stack on 16-byte as ABI requires
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        // Take src, apply powf on it and replace value on a stack with dst.
        Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);
        for (size_t i = 0; i < vlen / sizeof(float); ++i) {
            const Address &source = h->ptr[h->rsp + h->rbx + i * sizeof(float)];
            h->uni_vmovss(xmm0, source);
            h->uni_vmovss(xmm1, h->ptr[h->rsp + h->rbx + vlen]); // beta
            h->uni_vzeroupper(); // eliminate performance penalties on avx
            h->call(h->rbp);
            // eliminate performance penalties on sse isa
            if (isa == sse41) h->uni_vzeroupper();
            h->uni_vmovss(source, xmm0);
        }

        h->add(h->rsp, h->rbx);

        // restore vector registers
        for (size_t i = vecs_count + 1; i >= 2; --i)
            h->uni_vmovups(Vmm(i - 2), h->ptr[h->rsp + i * vlen]);
        h->uni_vmovups(vmm_src, h->ptr[h->rsp + 0 * vlen]);
        h->add(h->rsp, (vecs_count + 2) * vlen);

        // restore k registers
        if (has_avx512()) {
            for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
                if (mayiuse(avx512_core))
                    h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
                else
                    h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            }
            h->add(h->rsp, n_k_regs_to_save * k_mask_size);
        }

        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);

        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);

    // -exp(-x*x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, vmm_aux3);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, vmm_aux3);
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    h->uni_vmovups(vmm_aux2, table_val(gelu_erf_approx_const));
    h->uni_vfmadd213ps(vmm_aux2, vmm_aux1, table_val(one));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux2);

    // -exp(-x*x)*t
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // S = 0.5 * s = x / sqrt^2(2)
    h->uni_vmulps(vmm_aux3, vmm_aux3, table_val(gelu_erf_one_over_sqrt_two));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->uni_vfmadd213ps(vmm_src, vmm_aux3, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // invariant to whether `s` or `d` is passed.
    // get mask of `s` > 0
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    // fill with alpha, then blend with 1.f
    h->uni_vmovups(vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->uni_vaddps(vmm_src, vmm_src, table_val(alpha));
    }
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vfnmadd231ps(vmm_aux0, vmm_src, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->uni_vmovups(vmm_aux2, table_val(gelu_tanh_fitting_const_times_three));
    h->uni_vfmadd213ps(vmm_aux2, vmm_src, table_val(one));

    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(gelu_tanh_sqrt_two_over_pi));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux2);

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    if (isa == sse41 || isa == avx) {
        h->uni_vmovups(vmm_aux3, table_val(one));
        h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src);
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux3);
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);
        h->uni_vaddps(vmm_src, vmm_src, vmm_aux2);
    } else {
        // 1) R = G2 * (1 - T) = G2 - G2 * T
        h->uni_vfnmadd231ps(vmm_aux2, vmm_aux2, vmm_src);
        // 2) Q = 1 + T
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        // 3) res = Q * (1 + R) = Q + Q * R
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux2);
    }
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 2 * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(two));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(half));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_src, table_val(alpha));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero));
    // make all negative values zeros
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vsubps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = alpha * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    // compute Q * (1 + R * (1 - Q))
    if (utils::one_of(isa, sse41, avx)) {
        h->uni_vmovups(vmm_aux1, table_val(one));
        h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_src);
        h->uni_vmulps(vmm_aux1, vmm_aux1, vmm_aux0);
        h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
        h->uni_vmulps(vmm_src, vmm_src, vmm_aux1);
    } else {
        // T = R * (1 - Q) = R - R * Q
        h->uni_vfnmadd231ps(vmm_aux0, vmm_aux0, vmm_src);
        // Q * (1 + T) = Q + Q * T
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux0);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 / s
    h->uni_vmovups(vmm_aux0, table_val(one));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // set result with 1.f
    h->uni_vmovups(vmm_aux1, table_val(one));
    // get mask of values > beta and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(beta), _cmp_gt_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    // get mask of values <= alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_le_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    h->uni_vmovups(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_bwd(
        const Vmm &vmm_src) {
    // dispatch some special cases.
    if (beta_ == 0) { // zero
        h->uni_vmovups(vmm_src, table_val(zero));
    } else if (beta_ == 0.5) { // 0.5 * alpha / sqrt(s)
        sqrt_compute_vector_bwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else {
        // Save `s` on stack for later usage
        h->sub(h->rsp, vlen);
        h->uni_vmovups(h->ptr[h->rsp], vmm_src);
        // R = alpha * pow(s, beta)
        pow_compute_vector_fwd(vmm_src);
        // Restore `s` from stack
        h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
        h->add(h->rsp, vlen);
        // Save mask of zero elements to convert them into zeros at the end
        if (beta_ >= 1) compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
        // res = alpha * beta * pow(s, beta - 1) = beta * R / s;
        h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);
        h->uni_vmulps(vmm_src, vmm_src, table_val(beta));

        // beta < 1 leads to NaN as `s` appears in denominator, but beta >= 1
        // should lead to zero, when `s` is zero.
        if (beta_ >= 1) blend_with_mask(vmm_src, table_val(zero));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // Q = exp(-R*R)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->uni_vmulps(vmm_aux2, vmm_aux2, table_val(gelu_erf_one_over_sqrt_pi));
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    h->uni_vmovups(vmm_aux3, table_val(gelu_erf_approx_const));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, vmm_aux4);
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux3);

    // Q * W
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // P = T + 0.5
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(half));
    // res = P + 0.5 * erf
    h->uni_vfmadd231ps(vmm_aux2, vmm_src, table_val(half));
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_gprs_count() {
    using namespace alg_kind;
    switch (alg_) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
        case eltwise_gelu_tanh: return isa == sse41 || isa == avx ? 4 : 0;
        default: return 0;
    }
    return 0;
};

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vroundps(vmm_src, vmm_src, _op_mxcsr);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 0;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 5;
            case eltwise_clip: return 0;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            case eltwise_round: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd: return 1;
            case eltwise_elu: return 3;
            case eltwise_tanh_use_dst_for_bwd: return 1;
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 1;
            case eltwise_linear: return 0;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd: return 1;
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 1;
            case eltwise_clip: return 2;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(Vmm(idx));
                    else
                        relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(Vmm(idx)); break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_log: log_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->uni_vmulps(Vmm(idx), Vmm(idx), table_val(scale));
        }
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {{tanh_idx_bias, {0x39800000, true}},
            {tanh_idx_mask, {0xffc00000, true}},
            {tanh_linear_ubound, {0x39ddb3d7, true}},
            {tanh_saturation_lbound, {0x41102cb3, true}}};

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    static const table_t tanh_polynomial_table {
            // coefficients of degree 0
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x39bfffff, false}},
            {tanh_pol_table, {0x39ffffff, false}},
            {tanh_pol_table, {0x3a3ffffe, false}},
            {tanh_pol_table, {0x3a7ffffb, false}},
            {tanh_pol_table, {0x3abffff7, false}},
            {tanh_pol_table, {0x3affffeb, false}},
            {tanh_pol_table, {0x3b3fffdc, false}},
            {tanh_pol_table, {0x3b7fffab, false}},
            {tanh_pol_table, {0x3bbfff70, false}},
            {tanh_pol_table, {0x3bfffeab, false}},
            {tanh_pol_table, {0x3c3ffdc0, false}},
            {tanh_pol_table, {0x3c7ffaab, false}},
            {tanh_pol_table, {0x3cbff701, false}},
            {tanh_pol_table, {0x3cffeaad, false}},
            {tanh_pol_table, {0x3d3fdc08, false}},
            {tanh_pol_table, {0x3d7faacd, false}},
            {tanh_pol_table, {0x3dbf7081, false}},
            {tanh_pol_table, {0x3dfeacc9, false}},
            {tanh_pol_table, {0x3e3dc7fd, false}},
            {tanh_pol_table, {0x3e7acbf5, false}},
            {tanh_pol_table, {0x3eb77a9f, false}},
            {tanh_pol_table, {0x3eec9a9f, false}},
            {tanh_pol_table, {0x3f22991f, false}},
            {tanh_pol_table, {0x3f42f7d6, false}},
            {tanh_pol_table, {0x3f67b7cc, false}},
            {tanh_pol_table, {0x3f76ca83, false}},
            {tanh_pol_table, {0x3f7ebbe9, false}},
            {tanh_pol_table, {0x3f7fd40c, false}},
            {tanh_pol_table, {0x3f7fff32, false}},
            {tanh_pol_table, {0x3f7ffffc, false}},
            {tanh_pol_table, {0x3f800000, false}},
            // coefficients of degree 1
            {tanh_pol_table, {0x3f800000, false}},
            {tanh_pol_table, {0x3f800018, false}},
            {tanh_pol_table, {0x3f7fffe8, false}},
            {tanh_pol_table, {0x3f7fffda, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffac, false}},
            {tanh_pol_table, {0x3f7fff70, false}},
            {tanh_pol_table, {0x3f7ffeec, false}},
            {tanh_pol_table, {0x3f7ffdc0, false}},
            {tanh_pol_table, {0x3f7ffbed, false}},
            {tanh_pol_table, {0x3f7ff704, false}},
            {tanh_pol_table, {0x3f7feff5, false}},
            {tanh_pol_table, {0x3f7fdbca, false}},
            {tanh_pol_table, {0x3f7fbfff, false}},
            {tanh_pol_table, {0x3f7f7041, false}},
            {tanh_pol_table, {0x3f7f009b, false}},
            {tanh_pol_table, {0x3f7dc36c, false}},
            {tanh_pol_table, {0x3f7c0aa8, false}},
            {tanh_pol_table, {0x3f7734b8, false}},
            {tanh_pol_table, {0x3f70a4de, false}},
            {tanh_pol_table, {0x3f5f1fd8, false}},
            {tanh_pol_table, {0x3f495493, false}},
            {tanh_pol_table, {0x3f18b9ec, false}},
            {tanh_pol_table, {0x3ed706cb, false}},
            {tanh_pol_table, {0x3e390b06, false}},
            {tanh_pol_table, {0x3d90b11f, false}},
            {tanh_pol_table, {0x3c21a053, false}},
            {tanh_pol_table, {0x3aaf7fdb, false}},
            {tanh_pol_table, {0x37ccc1a3, false}},
            {tanh_pol_table, {0x355c6733, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 2
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xbe4e0ff1, false}},
            {tanh_pol_table, {0x3d25b1b1, false}},
            {tanh_pol_table, {0x3d6b6dab, false}},
            {tanh_pol_table, {0x3c9fb1d5, false}},
            {tanh_pol_table, {0xbabff06f, false}},
            {tanh_pol_table, {0x3c07b3f6, false}},
            {tanh_pol_table, {0xbb3fc1bc, false}},
            {tanh_pol_table, {0x3a9f5921, false}},
            {tanh_pol_table, {0xbbbf06f2, false}},
            {tanh_pol_table, {0xbbb0f402, false}},
            {tanh_pol_table, {0xbc47db9e, false}},
            {tanh_pol_table, {0xbc73d5e7, false}},
            {tanh_pol_table, {0xbca25bda, false}},
            {tanh_pol_table, {0xbcfca780, false}},
            {tanh_pol_table, {0xbd40e07c, false}},
            {tanh_pol_table, {0xbd7dab03, false}},
            {tanh_pol_table, {0xbdbe4a0f, false}},
            {tanh_pol_table, {0xbdfb14a5, false}},
            {tanh_pol_table, {0xbe36cc8d, false}},
            {tanh_pol_table, {0xbe6bd102, false}},
            {tanh_pol_table, {0xbe9fe7c5, false}},
            {tanh_pol_table, {0xbeba0f10, false}},
            {tanh_pol_table, {0xbec206a8, false}},
            {tanh_pol_table, {0xbea3c388, false}},
            {tanh_pol_table, {0xbe277d62, false}},
            {tanh_pol_table, {0xbd8b7960, false}},
            {tanh_pol_table, {0xbc209f49, false}},
            {tanh_pol_table, {0xbaad44ca, false}},
            {tanh_pol_table, {0xb7c6eeac, false}},
            {tanh_pol_table, {0xb663aa41, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 3
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x45b3ae96, false}},
            {tanh_pol_table, {0xc414eb20, false}},
            {tanh_pol_table, {0xc450e02e, false}},
            {tanh_pol_table, {0xc3152b4e, false}},
            {tanh_pol_table, {0xbead2f56, false}},
            {tanh_pol_table, {0xc2162e02, false}},
            {tanh_pol_table, {0xbeb4bd5a, false}},
            {tanh_pol_table, {0xc11a59a4, false}},
            {tanh_pol_table, {0xbed2f507, false}},
            {tanh_pol_table, {0xc020d32c, false}},
            {tanh_pol_table, {0x3dd0f506, false}},
            {tanh_pol_table, {0xbf2a75e2, false}},
            {tanh_pol_table, {0xbff950e3, false}},
            {tanh_pol_table, {0xbed47334, false}},
            {tanh_pol_table, {0xbe809b8c, false}},
            {tanh_pol_table, {0xbeb64532, false}},
            {tanh_pol_table, {0xbe961a5b, false}},
            {tanh_pol_table, {0xbe9b63ac, false}},
            {tanh_pol_table, {0xbea0d4b2, false}},
            {tanh_pol_table, {0xbe828a77, false}},
            {tanh_pol_table, {0xbe378612, false}},
            {tanh_pol_table, {0xbdc20908, false}},
            {tanh_pol_table, {0x3d2d3957, false}},
            {tanh_pol_table, {0x3dd46e89, false}},
            {tanh_pol_table, {0x3db3f629, false}},
            {tanh_pol_table, {0x3d2c5e7b, false}},
            {tanh_pol_table, {0x3bd20403, false}},
            {tanh_pol_table, {0x3a59dfae, false}},
            {tanh_pol_table, {0x3770af45, false}},
            {tanh_pol_table, {0x372cc014, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 4
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xcc981a1b, false}},
            {tanh_pol_table, {0x4a7edd3d, false}},
            {tanh_pol_table, {0x4ab1007c, false}},
            {tanh_pol_table, {0x48fedd9c, false}},
            {tanh_pol_table, {0x41a557b5, false}},
            {tanh_pol_table, {0x477ee32a, false}},
            {tanh_pol_table, {0x422557f5, false}},
            {tanh_pol_table, {0x45ff3ce4, false}},
            {tanh_pol_table, {0x42a55641, false}},
            {tanh_pol_table, {0x446e0867, false}},
            {tanh_pol_table, {0xc33dc19a, false}},
            {tanh_pol_table, {0x42915214, false}},
            {tanh_pol_table, {0x43af4fad, false}},
            {tanh_pol_table, {0x4110fe88, false}},
            {tanh_pol_table, {0xc1099b75, false}},
            {tanh_pol_table, {0x3fc8a8dc, false}},
            {tanh_pol_table, {0xbfbeaef5, false}},
            {tanh_pol_table, {0xbe365aad, false}},
            {tanh_pol_table, {0x3f4d9652, false}},
            {tanh_pol_table, {0x3ddfa08f, false}},
            {tanh_pol_table, {0x3e34e9b8, false}},
            {tanh_pol_table, {0x3e2d07a6, false}},
            {tanh_pol_table, {0x3dc63567, false}},
            {tanh_pol_table, {0x3cdaeb78, false}},
            {tanh_pol_table, {0xbcd17537, false}},
            {tanh_pol_table, {0xbc92829c, false}},
            {tanh_pol_table, {0xbb43ab99, false}},
            {tanh_pol_table, {0xb9b471dd, false}},
            {tanh_pol_table, {0xb6baad5a, false}},
            {tanh_pol_table, {0xb78bafc7, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 5
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x52f688d5, false}},
            {tanh_pol_table, {0xd0505c72, false}},
            {tanh_pol_table, {0xd08f98e3, false}},
            {tanh_pol_table, {0xce505cc9, false}},
            {tanh_pol_table, {0xc7162b8a, false}},
            {tanh_pol_table, {0xcc5061d6, false}},
            {tanh_pol_table, {0xc7162bdf, false}},
            {tanh_pol_table, {0xca50b37f, false}},
            {tanh_pol_table, {0xc7162a3a, false}},
            {tanh_pol_table, {0xc8422086, false}},
            {tanh_pol_table, {0x471a714e, false}},
            {tanh_pol_table, {0xc5ece1f1, false}},
            {tanh_pol_table, {0xc70e3d90, false}},
            {tanh_pol_table, {0xc3eba94a, false}},
            {tanh_pol_table, {0x43e0c424, false}},
            {tanh_pol_table, {0xc21f4552, false}},
            {tanh_pol_table, {0x42217cc8, false}},
            {tanh_pol_table, {0x405e7dc4, false}},
            {tanh_pol_table, {0xc10dd401, false}},
            {tanh_pol_table, {0x3e96b602, false}},
            {tanh_pol_table, {0xbd1a6d2f, false}},
            {tanh_pol_table, {0xbd393883, false}},
            {tanh_pol_table, {0xbd674682, false}},
            {tanh_pol_table, {0xbd310016, false}},
            {tanh_pol_table, {0xb961e269, false}},
            {tanh_pol_table, {0x3ba32495, false}},
            {tanh_pol_table, {0x3a7680d5, false}},
            {tanh_pol_table, {0x38b3173c, false}},
            {tanh_pol_table, {0x35a9deea, false}},
            {tanh_pol_table, {0x375c3f2a, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 6
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xd8995ed1, false}},
            {tanh_pol_table, {0x558285ea, false}},
            {tanh_pol_table, {0x55b2cd69, false}},
            {tanh_pol_table, {0x53028625, false}},
            {tanh_pol_table, {0x4bc9991f, false}},
            {tanh_pol_table, {0x5082898a, false}},
            {tanh_pol_table, {0x4b4999b3, false}},
            {tanh_pol_table, {0x4e02c07c, false}},
            {tanh_pol_table, {0x4ac99764, false}},
            {tanh_pol_table, {0x4b72c822, false}},
            {tanh_pol_table, {0xca40c0e1, false}},
            {tanh_pol_table, {0x489413e4, false}},
            {tanh_pol_table, {0x49b12224, false}},
            {tanh_pol_table, {0x46134c4e, false}},
            {tanh_pol_table, {0xc60c2d57, false}},
            {tanh_pol_table, {0x43c83910, false}},
            {tanh_pol_table, {0xc3c872d1, false}},
            {tanh_pol_table, {0xc186bc9e, false}},
            {tanh_pol_table, {0x42325bc3, false}},
            {tanh_pol_table, {0xbf2ffa4a, false}},
            {tanh_pol_table, {0x3d9a203c, false}},
            {tanh_pol_table, {0xbc545a43, false}},
            {tanh_pol_table, {0xbae08fee, false}},
            {tanh_pol_table, {0x3c80225d, false}},
            {tanh_pol_table, {0x3b1fd1df, false}},
            {tanh_pol_table, {0xba36b9d1, false}},
            {tanh_pol_table, {0xb91de544, false}},
            {tanh_pol_table, {0xb71f100f, false}},
            {tanh_pol_table, {0xb408e2ed, false}},
            {tanh_pol_table, {0xb685fec8, false}},
            {tanh_pol_table, {0x00000000, false}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };

    // log(x) constants
    static const table_t log_consts {
            {log_minus_inf, {0xff800000, true}},
            {log_qnan, {0x7fc00000, true}},
            {log_mantissa_mask, {0x007fffff, true}},
            {log_full_k_reg_mask, {0x0000ffff, true}},
            {log_five_bit_offset, {0x0000001f, true}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0xbf000000, true}}, // p1 = -0.5f
            {log_pol, {0x3eaaaaab, true}}, // p2 =  0.333333343f
            {log_pol, {0xbe8004ab, true}}, // p3 = -0.250035613f
            {log_pol, {0x3e4cc8a3, true}}, // p4 =  0.199984118f
    };

    // log(x) pre-defined values. First goes index}, then val[index].
    static const table_t log_predefined_values {
            {log_predefined_vals, {0x3f800000, true}}, //  0: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, //  1: -88.029693603515625
            {log_predefined_vals, {0x3f780000, true}}, //  2: 0.96875
            {log_predefined_vals,
                    {0xc2affef2, true}}, //  3: -87.9979400634765625
            {log_predefined_vals, {0x3f700000, true}}, //  4: 0.9375
            {log_predefined_vals,
                    {0xc2afee29, true}}, //  5: -87.9651565551757812
            {log_predefined_vals, {0x3f680000, true}}, //  6: 0.90625
            {log_predefined_vals,
                    {0xc2afdccd, true}}, //  7: -87.9312515258789062
            {log_predefined_vals, {0x3f600000, true}}, //  8: 0.875
            {log_predefined_vals,
                    {0xc2afcad6, true}}, //  9: -87.8961639404296875
            {log_predefined_vals, {0x3f580000, true}}, // 10: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 11: -87.859794616699218
            {log_predefined_vals, {0x3f580000, true}}, // 12: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 13: -87.859794616699218
            {log_predefined_vals, {0x3f500000, true}}, // 14: 0.8125
            {log_predefined_vals,
                    {0xc2afa4e4, true}}, // 15: -87.822052001953125
            {log_predefined_vals, {0x3f480000, true}}, // 16: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 17: -87.782829284667968
            {log_predefined_vals, {0x3f480000, true}}, // 18: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 19: -87.782829284667968
            {log_predefined_vals, {0x3f400000, true}}, // 20: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 21: -87.742012023925781
            {log_predefined_vals, {0x3f400000, true}}, // 22: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 23: -87.742012023925781
            {log_predefined_vals, {0x3f380000, true}}, // 24: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 25: -87.699447631835937
            {log_predefined_vals, {0x3f380000, true}}, // 26: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 27: -87.699447631835937
            {log_predefined_vals, {0x3f300000, true}}, // 28: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 29: -87.654998779296875
            {log_predefined_vals, {0x3f300000, true}}, // 30: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 31: -87.654998779296875
            {log_predefined_vals, {0x3fa80000, true}}, // 32: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 33: -88.301628112792968
            {log_predefined_vals, {0x3fa80000, true}}, // 34: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 35: -88.301628112792968
            {log_predefined_vals, {0x3fa00000, true}}, // 36: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 37: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 38: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 39: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 40: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 41: -88.252838134765625
            {log_predefined_vals, {0x3f980000, true}}, // 42: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 43: -88.201545715332031
            {log_predefined_vals, {0x3f980000, true}}, // 44: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 45: -88.201545715332031
            {log_predefined_vals, {0x3f900000, true}}, // 46: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 47: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 48: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 49: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 50: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 51: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 52: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 53: -88.147476196289062
            {log_predefined_vals, {0x3f880000, true}}, // 54: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 55: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 56: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 57: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 58: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 59: -88.090316772460937
            {log_predefined_vals, {0x3f800000, true}}, // 60: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 61: -88.029693603515625
            {log_predefined_vals, {0x3f800000, true}}, // 62: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 63: -88.029693603515625
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh: gelu_tanh_ = true; break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_ = true; break;
                default: break;
            }
        }

        bool exp_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial_table);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);
    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.log()) push_entries_of(log_predefined_values);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_injector_f32<avx512_core_bf16>;
template struct jit_uni_eltwise_injector_f32<avx512_core>;
template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<avx>;
template struct jit_uni_eltwise_injector_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
