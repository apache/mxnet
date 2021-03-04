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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/cpu_barrier.hpp"

#include "cpu/x64/jit_avx512_core_bf16_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    if (utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc)
            && jcp.ngroups > 1 && jcp.oc < 16) {
        jcp.loop_order = loop_nhwcg;
    } else if (jcp.prop_kind == backward_data) {
        // ow-threading is currently implemented for forward only
        // TODO: single code for fwd and bwd after ow-thr for bwd
        // meaningless switch was removed
        if (jcp.ndims < 5)
            jcp.loop_order = (w <= small_spatial && h <= small_spatial)
                    ? loop_cwgn
                    : loop_gncw;
        else
            jcp.loop_order = (w <= small_spatial && h <= small_spatial)
                    ? loop_cgn
                    : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? loop_cwgn
                                                                    : loop_gncw;
    }
}
inline bool is_ow_threading_available(const jit_conv_conf_t &jcp) {
    /*is 1D conv */
    return (jcp.id == 1 && jcp.ih == 1 && jcp.kd == 1 && jcp.kh == 1);
}
inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}
inline bool is_iw_threading_available(const jit_conv_conf_t &jcp) {
    return one_of(jcp.ndims, 3, 4);
}
inline bool is_iw_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_iw > 1);
}
inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    const bool no_big_offt = nstl::max<size_t>(jcp.ic, jcp.oc)
                    * nstl::max(jcp.typesize_in, jcp.typesize_out) * jcp.id
                    * jcp.ih * jcp.iw
            < INT_MAX;
    return jcp.ic < 16 && jcp.ngroups == 1 && no_big_offt;
}
} // namespace

template <typename Vmm>
void _jit_avx512_core_bf16_fwd_kernel<Vmm>::prepare_dst(int ur_w) {
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_dst(j, k);
            vpxord(vmm, vmm, vmm);
        }
}

template <typename Vmm>
void _jit_avx512_core_bf16_fwd_kernel<Vmm>::store_dst(int ur_w) {
    Label store_label;
    const int oc_tail = jcp.oc_tail;
    if (!isa_has_bf16(jcp.isa)) bf16_emu_->init_vcvtneps2bf16();

    if (jcp.with_sum) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            for (int j = 0; j < ur_w; j++) {
                // mask only needed for last oc_block
                bool mask_flag = oc_tail && k + 1 == jcp.nb_oc_blocking;
                Vmm vmm = vmm_dst(j, k);
                size_t aux_dst_offset = get_dst_offset(j, k);
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(may_be_mask_vmm(vmm_prev_dst, mask_flag, true),
                            make_safe_addr(reg_dst, aux_dst_offset,
                                    reg_dst_long_offt));
                    vpslld(vmm_prev_dst, vmm_prev_dst, 16);
                    vaddps(vmm, vmm_prev_dst);
                } else {
                    vaddps(may_be_mask_vmm(vmm, mask_flag, true),
                            make_safe_addr(reg_dst, aux_dst_offset,
                                    reg_dst_long_offt));
                }
            }
        }
    }

    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_bia * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                // mask only needed for last oc_block
                bool mask_flag = oc_tail && k + 1 == jcp.nb_oc_blocking;
                Vmm vmm = vmm_dst(j, k);
                if (jcp.bia_dt == data_type::bf16) {
                    vpmovzxwd(may_be_mask_vmm(vmm_bias, mask_flag, true),
                            EVEX_compress_addr(reg_bias, bias_offset));
                    vpslld(vmm_bias, vmm_bias, 16);
                    vaddps(vmm, vmm_bias);
                } else
                    vaddps(may_be_mask_vmm(vmm, mask_flag, true),
                            EVEX_compress_addr(reg_bias, bias_offset));
            }
        }
    }

    if (jcp.with_eltwise) {
        eltwise_injector_->compute_vector_range(0, jcp.nb_oc_blocking * ur_w);
    }

    L(store_label);
    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Vmm vmm = vmm_dst(j, k);
                size_t aux_dst_offset = get_dst_offset(j, k);
                auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);
                // mask only needed for last oc_block
                bool mask_flag = oc_tail && k + 1 == jcp.nb_oc_blocking;
                vmovups(addr, may_be_mask_vmm(vmm, mask_flag, false));
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa) && is_dst_layout_nxc()) {
            // Optimization: use single store instruction for pair of the
            // nearest vectors along OC dimension
            for (int j = 0; j < ur_w; j++) {
                int k = 0;
                for (; k < rnd_dn(jcp.nb_oc_blocking, 2); k += 2) {
                    Vmm vmm = vmm_dst(j, k);
                    Vmm vmm_next = vmm_dst(j, k + 1);
                    size_t aux_dst_offset = get_dst_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);
                    vcvtne2ps2bf16(vmm, vmm_next, vmm);
                    // mask only needed for last oc_block
                    bool mask_flag = oc_tail && k + 2 == jcp.nb_oc_blocking;
                    vmovdqu16(
                            addr, may_be_mask_vmm(vmm, mask_flag, false, true));
                }
                if (jcp.nb_oc_blocking % 2 != 0) {
                    Vmm vmm = vmm_dst(j, k);
                    auto vmm_down = Vmm_down_t(vmm.getIdx());
                    size_t aux_dst_offset = get_dst_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);
                    vcvtneps2bf16(vmm_down, vmm);
                    // for xmm, upper half is zero after conversion to
                    // bf16, so mask always & mask for tails
                    bool mask_flag = jcp.simd_w == 4 || oc_tail;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down, mask_flag));
                }
            }
        } else if (isa_has_bf16(jcp.isa) /* !is_dst_layout_nxc() */) {
            // Optimization: use single store instruction for pair of the
            // nearest vectors along WIDTH dimension
            for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    size_t aux_dst_offset = get_dst_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);

                    auto vmm_str = vmm_src(j, jcp.nb_oc_blocking);
                    vcvtne2ps2bf16(vmm_str, vmm_dst(j + 1, k), vmm_dst(j, k));
                    vmovups(addr, vmm_str);
                }
                if (j < ur_w) {
                    size_t aux_dst_offset = get_dst_offset(j, k);

                    auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);
                    auto vmm_down_str = vmm_src_down(j, jcp.nb_oc_blocking);
                    vcvtneps2bf16(vmm_down_str, vmm_dst(j, k));
                    // for xmm, upper half is zero after conversion to
                    // bf16, so mask always & mask for tails
                    const bool mask_flag = jcp.simd_w == 4;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down_str, mask_flag));
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Vmm vmm = vmm_dst(j, k);
                    size_t aux_dst_offset = get_dst_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_dst, aux_dst_offset);
                    auto vmm_down = vmm_src_down(0, jcp.nb_oc_blocking);
                    bf16_emu_->vcvtneps2bf16(
                            Ymm(vmm_down.getIdx()), Zmm(vmm.getIdx()));
                    bool mask_flag = (oc_tail && k + 1 == jcp.nb_oc_blocking)
                            // for xmm, upper half is zero after conversion to
                            // bf16, so mask always & mask for tails
                            || jcp.simd_w == 4;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down, mask_flag));
                }
        }
    } else
        assert(!"unsupported destination type");
}

template <typename Vmm>
void _jit_avx512_core_bf16_fwd_kernel<Vmm>::compute_loop(
        int ur_w, int pad_l, int pad_r) {
    Label kh_label, kd_label;
    const int ic_tail = jcp.ic_tail;
    const int ic_step = 2;

    /* max_src_offset is explicitly used in the 1st convolution.
     * Set its value so that accessing the double-word memory
     * referenced by ptr[src_base + offset] is safe whenever
     *     0 <= offset < max_src_offset
     *
     * Note: Since the arguments pad_l, pad_r might not exactly match
     * with jcp.l_pad and jcp.r_pad respectively so this value needs to be
     * computed separately for each invocation of the compute_loop.
     */
    dim_t max_src_offset = 0;
    if (jcp.is_1stconv || ic_tail) {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_fst = get_ow_start(ki, pad_l);
            int ow_last = get_ow_end(ur_w, ki, pad_r) - 1;
            if (ow_fst > ow_last) continue;
            int ic_last = rnd_up(nstl::min(jcp.ic_block,
                                         nstl::max(jcp.ic, ic_tail)),
                                  ic_step)
                    - ic_step;

            dim_t src_offset = get_src_offset(
                    ic_last, filter_w_to_src(ki, ow_last, pad_l));
            if (src_offset > max_src_offset) max_src_offset = src_offset;
        }
    }

    prepare_dst(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        mov(reg_kj, ptr[param1 + GET_OFF(kd_padding)]);
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_kj, 0);
            je(skip_compute_loop, T_NEAR);
        }
    }
    mov(reg_kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_compute_loop, T_NEAR);
    }

    // IC loop
    Label icb_label;
    mov(reg_ic, jcp.ic);
    L(icb_label);

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_src_d, reg_src);

        L(kd_label);
        mov(aux_reg_src, aux_reg_src_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        mov(aux_reg_src, reg_src);
        mov(aux_reg_ker, reg_ker);
    }

    mov(reg_kj, reg_kh);

    std::vector<Label> ic_tail_jmp(jcp.kw);
    L(kh_label);
    {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_start = get_ow_start(ki, pad_l);
            int ow_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0;
                    ic < rnd_up(nstl::min(jcp.ic_block, jcp.ic), ic_step);
                    ic += ic_step) {
                if (ic_tail && ic == rnd_up(ic_tail, ic_step)) {
                    // insert this check at most once per icb, no more.
                    cmp(reg_ic, ic_tail);
                    je(ic_tail_jmp[ki], T_NEAR);
                }
                for (int oi = ow_start; oi < ow_end; oi++) {
                    dim_t src_offset = get_src_offset(
                            ic, filter_w_to_src(ki, oi, pad_l));
                    auto vmm_in = vmm_src(oi, jcp.nb_oc_blocking);
                    const auto addr_base
                            = EVEX_compress_addr(aux_reg_src, src_offset);
                    const bool tail_load
                            = ic_tail && ic == rnd_dn(ic_tail, ic_step);
                    if (jcp.is_1stconv || tail_load) {
                        const bool need_single_load
                                = (ic + 1 == jcp.ic || ic + 1 == ic_tail);
                        const bool safe_overstep = (src_offset < max_src_offset)
                                && !is_src_layout_nxc();

                        /* For the comment below, let us define three words
                         * x_b = ptr[addr_base] and x_s = ptr[addr_strided]
                         * x_g = ptr[addr_base + 2]
                         *
                         * For single load case:
                         * Without overstep zmm_in register is loaded as
                         *     [0, x_b, ..., 0, x_b, 0, x_b]
                         * On the other hand, "with overstep" zmm_in register
                         * is loaded as
                         *     [x_g, x_b, ..., x_g, x_b, x_g, x_b]
                         * where x_g is a garbage word.
                         *
                         * Note:
                         * 1. In single load case with safe_overstep enabled,
                         * it is implicitly assumed that the element in zmm_wei
                         * register corresponding to the "garbage value x_g" in
                         * zmm_in register is zero.
                         * 2. One can have potential problem when x_g is
                         * either Inf or NaN since it is multiplied by zero
                         * in accumulation. But as x_g is a "valid input"
                         * for different offset so one might assume that x_g is
                         * neither Inf nor Nan.
                         *
                         * For non single load case:
                         * zmm_in register is loaded as
                         *     [x_s, x_b, ...., x_s, x_b, x_s, x_b]
                         */
                        if (tail_load) {
                            if (need_single_load) {
                                Label mask_load, load_done;
                                cmp(reg_ic, ic + ic_step);
                                jl(mask_load, T_NEAR);
                                vpbroadcastd(vmm_in, addr_base);
                                jmp(load_done, T_NEAR);
                                L(mask_load);
                                vpbroadcastw(vmm_in | odd_load_mask | T_z,
                                        addr_base);
                                L(load_done);
                            } else {
                                vpbroadcastd(vmm_in, addr_base);
                            }
                        } else if (need_single_load && !safe_overstep)
                            vpbroadcastw(
                                    vmm_in | odd_load_mask | T_z, addr_base);
                        else if (IMPLICATION(!is_src_layout_nxc(),
                                         need_single_load && safe_overstep))
                            vpbroadcastd(vmm_in, addr_base);
                        else {
                            const auto addr_strided
                                    = EVEX_compress_addr(aux_reg_src,
                                            src_offset + get_src_offset(1, 0));
                            vpbroadcastd(vmm_in, addr_base);
                            vpbroadcastw(vmm_in | even_load_mask, addr_strided);
                        }
                    } else {
                        vpbroadcastd(vmm_in, addr_base);
                    }
                }
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    auto wei_off = get_kernel_offset(kk, ic, ki);
                    vmovups(vmm_wei, EVEX_compress_addr(aux_reg_ker, wei_off));
                    for (int oi = ow_start; oi < ow_end; oi++) {
                        auto acc = vmm_dst(oi, kk);
                        auto src = vmm_src(oi, jcp.nb_oc_blocking);
                        if (isa_has_bf16(jcp.isa)) {
                            vdpbf16ps(acc, vmm_wei, src);
                        } else
                            bf16_emu_->vdpbf16ps(Zmm(acc.getIdx()),
                                    Zmm(vmm_wei.getIdx()), Zmm(src.getIdx()));
                    }
                }
            }
            L(ic_tail_jmp[ki]);
        }
        add(aux_reg_ker, get_kernel_offset(0, 0, 0, 1));
        add(aux_reg_src, get_src_offset(0, filter_h_to_src(1)));

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_src_d, get_src_offset(0, filter_d_to_src(1)));
        add(aux_reg_ker_d, get_kernel_offset(0, 0, 0, 0, 1));
        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    // End of IC Loop
    dim_t src_step = get_src_offset(jcp.ic_block, 0);
    size_t ker_step = get_kernel_offset(0, jcp.ic_block, 0);
    add(reg_src, src_step);
    add(reg_ker, ker_step);

    sub(reg_ic, jcp.ic_block);
    cmp(reg_ic, 0);
    jg(icb_label, T_NEAR);

    sub(reg_src, src_step * jcp.nb_ic);
    sub(reg_ker, ker_step * jcp.nb_ic);

    L(skip_compute_loop);
    store_dst(ur_w);
}

template <typename Vmm>
void _jit_avx512_core_bf16_fwd_kernel<Vmm>::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    auto src_shift = get_src_offset(0, filter_w_to_src(0, ur_w));
    auto dst_shift = get_dst_offset(ur_w, 0);

    auto src_shift_pad = get_src_offset(0, filter_w_to_src(0, ur_w, l_pad));
    auto src_shift_pad_second_block
            = get_src_offset(0, filter_w_to_src(0, 0, l_pad));

    preamble();

    if (jcp.is_1stconv || jcp.ic_tail) {
        Xbyak::Reg64 reg_alt_mask = r8;
        const auto odd_mask = size_t {0x5555555555555555};
        const auto even_mask = size_t {0xaaaaaaaaaaaaaaaa};
        mov(reg_alt_mask, odd_mask);
        kmovq(odd_load_mask, reg_alt_mask);
        mov(reg_alt_mask, even_mask);
        kmovq(even_load_mask, reg_alt_mask);
    }

    if (jcp.simd_w == 4) {
        auto reg_tail_32 = reg_oc.cvt32();
        mov(reg_tail_32, (1 << jcp.simd_w) - 1);
        kmovb(k_oc_tail_mask, reg_tail_32);
    }

    if (jcp.oc_tail) {
        Label done;
        // dummy mask all 1's
        if (jcp.simd_w != 4) { // simd_w == 4, has its dummy mask set already
            kxnord(k_oc_tail_mask, k_oc_tail_mask, k_oc_tail_mask);
        }
        // To account for special store optimization, where two oc_blocks are
        // combined with one single write, extend the mask for 32bits (32 bf16s)
        const bool need_extended_mask = jcp.dst_dt == data_type::bf16
                && isa_has_bf16(jcp.isa) && jcp.nb_oc_blocking > 1;
        if (need_extended_mask)
            kxnord(k_oc_tail_mask_extended, k_oc_tail_mask_extended,
                    k_oc_tail_mask_extended);

        test(byte[param1 + GET_OFF(load_work)], jcp.oc_block - 1);
        jz(done, T_NEAR);
        auto reg_tail_32 = reg_oc.cvt32();
        mov(reg_tail_32, (1 << jcp.oc_tail) - 1);
        kmovd(k_oc_tail_mask, reg_tail_32);
        if (need_extended_mask) {
            mov(reg_tail_32, (1 << (jcp.oc_tail + jcp.simd_w)) - 1);
            kmovd(k_oc_tail_mask_extended, reg_tail_32);
        }
        L(done);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0) n_oi--;

        xor_(reg_oi, reg_oi);
        if (ow == ur_w) {
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            if (n_oi == 0) {
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_src, src_shift_pad);
                add(reg_dst, dst_shift);
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            } else {
                if (l_pad > 0) {
                    compute_loop(ur_w, l_pad, 0);
                    add(reg_src, src_shift_pad);
                    add(reg_dst, dst_shift);
                    inc(reg_oi);
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        compute_loop(ur_w, 0, 0);
                        add(reg_src, src_shift);
                        add(reg_dst, dst_shift);

                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0) {
                    compute_loop(ur_w, 0, r_pad1);
                    add(reg_src, src_shift);
                    add(reg_dst, dst_shift);
                }
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow - 1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        if (l_pad > 0) {
            compute_loop(ur_w, l_pad, 0);
            add(reg_src, src_shift_pad);
            add(reg_dst, dst_shift);
            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_src, src_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        je(oi_loop_label, T_NEAR);
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        je(oi_loop_label, T_NEAR);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        jle(oi_loop_end_label, T_NEAR);

        compute_loop(ur_w, 0, 0);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        dec(reg_oi);
        jmp(oi_loop_start_label, T_NEAR);
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        // that is last block
        if (!last_ow_block_padded) { jmp(tail_label, T_NEAR); }

        // last oi block with right padding
        L(last_oi_label);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx512_core_bf16_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len()) {
        case 0: return true; // no post_ops
        case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
        case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
        default: return false;
    }

    return false;
}

void jit_avx512_core_bf16_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace memory_tracking::names;
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_bia);
    }
}

status_t jit_avx512_core_bf16_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.dst_dt = dst_d.data_type();

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    jcp.bia_dt = jcp.with_bias ? bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c,
            dat_tag_nCx8c, dat_tag_nCx4c, dat_tag_ncx);
    auto curr_dst_tag = dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    jcp.is_1stconv = is_1stconv(jcp);

    const int regs = isa_has_bf16(jcp.isa) ? 31 /* expl_bcast case */ : 26;
    const bool ok_to_pad_channels = jcp.ngroups == 1 && !is_data_layout_nxc;

    jcp.simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    const bool ok_to_try_lower_zmm = true
            && IMPLICATION(is_data_layout_nxc,
                    jcp.oc < jcp.simd_w && jcp.ic < jcp.simd_w
                            && jcp.ngroups > 1)
            && !jcp.is_1stconv && !ok_to_pad_channels
            && (jcp.ic % jcp.simd_w != 0 || jcp.oc % jcp.simd_w != 0);

    if (ok_to_try_lower_zmm) {
        for (auto simd : {8, 4}) {
            if (jcp.ic % simd == 0 && jcp.oc % simd == 0) {
                jcp.simd_w = simd;
                break;
            }
        }
    }

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    if (!IMPLICATION(!is_data_layout_nxc,
                jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0))
        return status::unimplemented;
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    format_tag_t src_tag, dst_tag, wei_tag;

    if (jcp.simd_w == 8) {
        assert(with_groups);
        dst_tag = src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
        wei_tag = pick(ndims - 3, gOIw4i8o2i, gOIhw4i8o2i, gOIdhw4i8o2i);
    } else if (jcp.simd_w == 4) {
        assert(with_groups);
        dst_tag = src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
        wei_tag = pick(ndims - 3, gOIw2i4o2i, gOIhw2i4o2i, gOIdhw2i4o2i);
    } else if (jcp.is_1stconv) {
        dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_ncx;
        wei_tag = pick(2 * ndims - 6 + with_groups, OwI16o2i, gOwI16o2i,
                OhwI16o2i, gOhwI16o2i, OdhwI16o2i, gOdhwI16o2i);
    } else {
        dst_tag = src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw8i16o2i, gOIw8i16o2i,
                OIhw8i16o2i, gOIhw8i16o2i, OIdhw8i16o2i, gOIdhw8i16o2i);
    }

    if (src_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
    else if (curr_src_tag != src_tag)
        return status::unimplemented;
    jcp.src_tag = src_tag;

    if (dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));
    else if (curr_dst_tag != dst_tag)
        return status::unimplemented;
    jcp.dst_tag = dst_tag;

    if (weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;
    }

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    jcp.aligned_threads = 0;

    bool args_ok = true && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    jcp.nb_ic = utils::div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = utils::div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.kernel_kind = expl_bcast;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--) {
        int ur_w = regs / (jcp.nb_oc_blocking + 1);
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && (jcp.l_pad <= ur_w
                        && IMPLICATION(jcp.ow != 1, jcp.ow % ur_w != 1)))
            break;
    }

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    if (is_ow_threading_available(jcp)) {
        const int L1_part = platform::get_per_core_cache_size(1) * 5 / 8;
        int size_src_chunk = jcp.typesize_in * jcp.ic_block * jcp.ur_w;
        int size_dst_chunk = jcp.typesize_out * jcp.oc_block
                * jcp.nb_oc_blocking * jcp.ur_w;
        int size_wei_chunk = jcp.typesize_in * jcp.oc_block * jcp.ic_block
                * jcp.nb_oc_blocking * jcp.kw;
        int nurw = (L1_part - size_wei_chunk)
                / (size_dst_chunk + size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        jcp.ow_block = jcp.ur_w * nstl::max(2, nurw);
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (jcp.l_pad > jcp.ur_w || r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

template <typename Vmm>
void _jit_avx512_core_bf16_bwd_data_kernel<Vmm>::prepare_output(int ur_w) {
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_dsrc(j, k);
            vpxord(vmm, vmm, vmm);
        }
    }
}

template <typename Vmm>
void _jit_avx512_core_bf16_bwd_data_kernel<Vmm>::store_output(int ur_w) {
    if (!isa_has_bf16(jcp.isa)) bf16_emu_->init_vcvtneps2bf16();
    const int ic_tail = jcp.ic_tail;

    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_ic_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Vmm vmm = vmm_dsrc(j, k);
                size_t aux_diff_src_offset = get_diff_src_offset(j, k);
                auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                // mask only needed for last ic_block
                bool mask_flag = ic_tail && k + 1 == jcp.nb_ic_blocking;
                vmovups(addr, may_be_mask_vmm(vmm, mask_flag, false));
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa) && is_ddst_layout_nxc()) {
            // Optimization: use single store instruction for pair of the
            // nearest vectors along IC dimension
            for (int j = 0; j < ur_w; j++) {
                int k = 0;
                for (; k < rnd_dn(jcp.nb_ic_blocking, 2); k += 2) {
                    Vmm vmm = vmm_dsrc(j, k);
                    Vmm vmm_next = vmm_dsrc(j, k + 1);
                    size_t aux_dsrc_offset = get_diff_src_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_src, aux_dsrc_offset);
                    vcvtne2ps2bf16(vmm, vmm_next, vmm);
                    bool mask_flag = ic_tail && k + 2 == jcp.nb_ic_blocking;
                    vmovdqu16(
                            addr, may_be_mask_vmm(vmm, mask_flag, false, true));
                }
                if (jcp.nb_ic_blocking % 2 != 0) {
                    Vmm vmm = vmm_dsrc(j, k);
                    auto vmm_down = Vmm_down_t(vmm.getIdx());
                    size_t aux_dsrc_offset = get_diff_src_offset(j, k);
                    auto addr = EVEX_compress_addr(reg_src, aux_dsrc_offset);
                    vcvtneps2bf16(vmm_down, vmm);
                    // for xmm, upper half is zero after conversion to
                    // bf16, so mask always & mask for tails
                    bool mask_flag = jcp.simd_w == 4 || ic_tail;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down, mask_flag));
                }
            }
        } else if (isa_has_bf16(jcp.isa) /* && !is_ddst_layout_nxc() */) {
            // Optimization: use single store instruction for pair of the
            // nearest vectors along WIDTH dimension
            int store_idx = 0;
            const int max_regs = 32;
            const int free_regs_start_idx = jcp.ur_w * jcp.nb_ic_blocking;
            const int num_regs_available = max_regs - free_regs_start_idx;
            int reg_idx = 0;
            for (int k = 0; k < jcp.nb_ic_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    reg_idx = free_regs_start_idx
                            + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);
                    size_t aux_diff_src_offset = get_diff_src_offset(j, k);
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);

                    auto vmm_str = Vmm(reg_idx);
                    vcvtne2ps2bf16(vmm_str, vmm_dsrc(j + 1, k), vmm_dsrc(j, k));
                    vmovups(addr, vmm_str);
                    store_idx++;
                }
                if (j < ur_w) {
                    reg_idx = free_regs_start_idx
                            + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);

                    size_t aux_diff_src_offset = get_diff_src_offset(j, k);
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    auto vmm_down_str = Vmm_down_t(reg_idx);
                    vcvtneps2bf16(vmm_down_str, vmm_dsrc(j, k));
                    // for xmm, upper half is zero after conversion to
                    // bf16, so mask always & mask for tails
                    bool mask_flag = jcp.simd_w == 4;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down_str, mask_flag));
                    store_idx++;
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_ic_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Vmm vmm = vmm_dsrc(j, k);
                    size_t aux_diff_src_offset = get_diff_src_offset(j, k);
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    auto vmm_down = vmm_ddst_down(0);
                    bf16_emu_->vcvtneps2bf16(
                            Ymm(vmm_down.getIdx()), Zmm(vmm.getIdx()));
                    bool mask_flag = (ic_tail && k + 1 == jcp.nb_ic_blocking)
                            // for xmm, upper half is zero after conversion to
                            // bf16, so mask always & mask for tails
                            || jcp.simd_w == 4;
                    vmovdqu16(addr, may_be_mask_vmm(vmm_down, mask_flag));
                }
        }
    } else
        assert(!"unsupported diff_src type");
}

template <typename Vmm>
void _jit_avx512_core_bf16_bwd_data_kernel<Vmm>::compute_loop(
        int ur_w, int l_overflow, int r_overflow) {
    int kw = jcp.kw;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    const int oc_tail = jcp.oc_tail;
    Label kh_label, skip_compute_label;

    prepare_output(ur_w);

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_ki, 0);
        jle(skip_compute_label, T_NEAR);
    }

    cmp(reg_kh, 0);
    jle(skip_compute_label, T_NEAR);

    // OC loop
    Label ocb_label;
    mov(reg_oc, jcp.oc);
    L(ocb_label);

    if (jcp.ndims < 5) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }
    Label kd_label;
    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, reg_ker);

        L(kd_label);
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    std::vector<Label> oc_tail_jmp(jcp.kw);
    mov(reg_kj, reg_kh);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            const int ref_jj_start
                    = nstl::max(0, l_overflow - (kw - 1 - ki) * dilate_w);
            const int ref_jj_end
                    = ur_w - nstl::max(0, r_overflow - ki * dilate_w);
            assert(IMPLICATION(stride_w == 1,
                    jj_start == ref_jj_start && jj_end == ref_jj_end));
            UNUSED(ref_jj_start);
            UNUSED(ref_jj_end);
            const int oc_step = 2;
            for (int oc = 0;
                    oc < rnd_up(nstl::min(jcp.oc_block, jcp.oc), oc_step);
                    oc += oc_step) {
                if (oc_tail && oc == rnd_up(oc_tail, oc_step)) {
                    cmp(reg_oc, oc_tail);
                    je(oc_tail_jmp[ki], T_NEAR);
                }
                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + jcp.l_pad - ki * dilate_w) % stride_w == 0);
                    int ow_idx = (jj + jcp.l_pad - ki * dilate_w) / stride_w;
                    auto aux_ddst_offset = get_diff_dst_offset(ow_idx, oc);
                    auto ddst = vmm_ddst(jj / stride_w);
                    const bool tail_load = oc_tail && oc == rnd_dn(oc_tail, 2);
                    const bool need_single_load = oc + 1 == oc_tail;

                    if (tail_load && need_single_load) {
                        Label mask_load, load_done;
                        cmp(reg_oc, oc + 2);
                        jl(mask_load, T_NEAR);
                        vpbroadcastd(ddst, ptr[aux_reg_dst + aux_ddst_offset]);
                        jmp(load_done, T_NEAR);
                        L(mask_load);
                        // We broadcast w here. As the weights are zero-padded
                        // at oc + 1, vdpbf16ps({0, w}, {dst, dst}) is okay.
                        vpbroadcastw(ddst, ptr[aux_reg_dst + aux_ddst_offset]);
                        L(load_done);
                    } else {
                        vpbroadcastd(ddst, ptr[aux_reg_dst + aux_ddst_offset]);
                    }
                }
                for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
                    size_t aux_kernel_offset = get_kernel_offset(kk, oc, ki);
                    vmovups(vmm_wei,
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));

                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        auto ddst = vmm_ddst(jj / stride_w);
                        auto acc = vmm_dsrc(jj, kk);

                        if (isa_has_bf16(jcp.isa)) {
                            vdpbf16ps(acc, vmm_wei, ddst);
                        } else
                            bf16_emu_->vdpbf16ps(Zmm(acc.getIdx()),
                                    Zmm(vmm_wei.getIdx()), Zmm(ddst.getIdx()));
                    }
                }
            }
            L(oc_tail_jmp[ki]);
        }

        add(aux_reg_ker, get_kernel_offset(0, 0, 0, stride_h));
        sub(aux_reg_dst, get_diff_dst_offset(filter_h_to_dst(1), 0));

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d, get_diff_dst_offset(filter_d_to_dst(1), 0));
        add(aux_reg_ker_d, get_kernel_offset(0, 0, 0, 0, jcp.stride_d));

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    // End of OC Loop
    auto diff_dst_step = get_diff_dst_offset(0, 0, 1);
    auto ker_step = get_kernel_offset(0, jcp.oc_block, 0);
    add(reg_dst, diff_dst_step);
    add(reg_ker, ker_step);

    sub(reg_oc, jcp.oc_block);
    cmp(reg_oc, 0);
    jg(ocb_label, T_NEAR);

    sub(reg_dst, diff_dst_step * jcp.nb_oc);
    sub(reg_ker, ker_step * jcp.nb_oc);

    L(skip_compute_label);
    store_output(ur_w);
}

template <typename Vmm>
void _jit_avx512_core_bf16_bwd_data_kernel<Vmm>::generate() {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int nb_iw = jcp.nb_iw;
    int iw_block = jcp.iw_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto dst_shift = get_diff_dst_offset(ur_w / stride_w, 0);
    const auto src_shift = get_diff_src_offset(ur_w, 0);

    preamble();

    if (jcp.simd_w == 4) {
        Reg32 reg_tail_32 = reg_oc.cvt32();
        mov(reg_tail_32, (1 << jcp.simd_w) - 1);
        kmovb(k_ic_tail_mask, reg_tail_32);
    }

    if (jcp.ic_tail) {
        Label done;
        // dummy mask all 1's
        if (jcp.simd_w != 4)
            kxnord(k_ic_tail_mask, k_ic_tail_mask, k_ic_tail_mask);
        // To account for special store optimization, where two ic_blocks are
        // combined with one single write, extend the mask for 32bits (32 bf16s)
        const bool need_extended_mask
                = isa_has_bf16(jcp.isa) && jcp.nb_ic_blocking > 1;
        if (need_extended_mask)
            kxnord(k_ic_tail_mask_extended, k_ic_tail_mask_extended,
                    k_ic_tail_mask_extended);

        test(byte[param1 + GET_OFF(load_work)], jcp.ic_block - 1);
        jz(done, T_NEAR);
        Reg32 reg_tail_32 = reg_ic.cvt32();
        mov(reg_tail_32, (1 << jcp.ic_tail) - 1);
        kmovd(k_ic_tail_mask, reg_tail_32);
        if (need_extended_mask) {
            mov(reg_tail_32, (1 << (jcp.ic_tail + jcp.simd_w)) - 1);
            kmovd(k_ic_tail_mask_extended, reg_tail_32);
        }
        L(done);
    }

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(
            0, ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0,
            ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad + ur_w_tail))
                    / stride_w);

    int body_l_overflow = 0, body_r_overflow = 0;
    int n_oi = iw / ur_w;
    int head_n_oi = 0, body_n_oi = 0, pretail_n_oi = 0, tail_n_oi = 0;
    int head_thread = 0, pretail_thread = 0, tail_thread = 0;
    bool threaded = is_iw_threading_on(jcp);
    Label head_label, body_label, pretail_label, tail_label, end_label;
    assert(n_oi > 0);

    if (r_overflow1 > 0) n_oi--;
    if (l_overflow > 0) n_oi--;
    if (n_oi < 0) {
        // l_overflow and r_overflow1 are handled in the same compute_loop.
        // Perform one iteration of body handling l_overflow and r_overflow1.
        body_l_overflow = l_overflow;
        body_r_overflow = r_overflow1;
        n_oi = 1;
        l_overflow = 0;
        r_overflow1 = 0;
    }

    if (!threaded) {
        if (n_oi > 1) { mov(reg_oi, n_oi); }
    } else {
        // Setup for threaded code generation, and jump into the correct
        // portion of code for execution.
        head_thread = 0;
        tail_thread = nb_iw - 1;
        pretail_thread = tail_thread;

        int base_n_oi = iw_block / ur_w;
        head_n_oi = l_overflow > 0 ? base_n_oi - 1 : base_n_oi;
        tail_n_oi = (iw - iw_block * (nb_iw - 1)) / ur_w;
        pretail_n_oi = tail_n_oi;
        if (r_overflow1 > 0) {
            if (tail_n_oi > 0) {
                pretail_n_oi--;
                tail_n_oi = pretail_n_oi;
            } else {
                // pretail_thread and tail_thread are different
                pretail_n_oi = base_n_oi - 1;
                pretail_thread = tail_thread - 1;
            }
            if (head_thread == pretail_thread) {
                head_n_oi--;
                pretail_n_oi = 0;
                tail_n_oi = 0;
            }
        }
        body_n_oi = (head_thread < pretail_thread - 1) ? base_n_oi : 0;

        // n_oi is used to determine how much control flow in the body portion
        // of the code needs generated. As such, n_oi needs to be set to the
        // maximum number of iterations it will be used the body code section.
        n_oi = nstl::max(body_n_oi, head_n_oi);
        n_oi = nstl::max(n_oi, pretail_n_oi);

        assert(iw_block % ur_w == 0);
        mov(reg_iwb, ptr[param1 + GET_OFF(iwb)]);

        if (head_n_oi != 0) mov(reg_oi, head_n_oi);
        cmp(reg_iwb, head_thread);
        je(head_label, T_NEAR);

        cmp(reg_iwb, pretail_thread);
        if (pretail_n_oi == 0) {
            je(pretail_label, T_NEAR);
        } else {
            mov(reg_oi, pretail_n_oi);
            je(body_label, T_NEAR);
        }
        if (pretail_thread != tail_thread) {
            cmp(reg_iwb, tail_thread);
            je(tail_label, T_NEAR);
        }
        if (body_n_oi != 0) {
            mov(reg_oi, body_n_oi);
            jmp(body_label, T_NEAR);
        } else {
            jmp(end_label, T_NEAR);
        }
    }
    L(head_label);
    if (l_overflow > 0) {
        compute_loop(ur_w, l_overflow, 0);
        if (threaded && head_n_oi == 0 && head_thread != pretail_thread)
            jmp(end_label, T_NEAR);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
    }
    L(body_label);
    if (n_oi > 0) {
        Label ow_loop_label;
        L(ow_loop_label);
        {
            compute_loop(ur_w, body_l_overflow, body_r_overflow);
            if (n_oi > 1 || r_overflow1 > 0 || ur_w_tail != 0) {
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
            }
            if (n_oi > 1) {
                sub(reg_oi, 1);
                jg(ow_loop_label, T_NEAR);
            }
        }
    }
    if (threaded) {
        cmp(reg_iwb, pretail_thread);
        jne(end_label, T_NEAR);
    }
    L(pretail_label);
    if (r_overflow1 > 0) {
        compute_loop(ur_w, 0, r_overflow1);
        if (ur_w_tail != 0) {
            if (threaded && tail_thread != pretail_thread)
                jmp(end_label, T_NEAR);
            else {
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
            }
        }
    }
    L(tail_label);
    if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_overflow); }
    L(end_label);

    postamble();
}

status_t jit_avx512_core_bf16_bwd_data_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &diff_src_md,
        memory_desc_t &weights_md, memory_desc_t &diff_dst_md, int nthreads) {

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.nthr = nthreads;
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];
    jcp.dst_dt = cd.diff_src_desc.data_type;
    jcp.nb_iw = 1;
    jcp.iw_block = jcp.iw;

    /* Dilated convolutions supported with unit strides only */
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    jcp.aligned_threads = 0;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = diff_src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    auto curr_dst_tag = diff_dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    bool ok_to_pad_channels = jcp.ngroups == 1 && !is_data_layout_nxc;

    jcp.simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    const bool ok_to_try_lower_zmm = true
            && IMPLICATION(is_data_layout_nxc,
                    jcp.oc < jcp.simd_w && jcp.ic < jcp.simd_w
                            && jcp.ngroups > 1)
            && !ok_to_pad_channels
            && (jcp.ic % jcp.simd_w != 0 || jcp.oc % jcp.simd_w != 0);

    if (ok_to_try_lower_zmm) {
        for (auto simd : {8, 4}) {
            if (jcp.ic % simd == 0 && jcp.oc % simd == 0) {
                jcp.simd_w = simd;
                break;
            }
        }
    }

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    if (!IMPLICATION(!is_data_layout_nxc,
                jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0))
        return status::unimplemented;
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    format_tag_t wei_tag, dat_tag;

    if (jcp.simd_w == 8) {
        dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
        wei_tag = utils::pick(ndims - 3, gOIw4o8i2o, gOIhw4o8i2o, gOIdhw4o8i2o);
    } else if (jcp.simd_w == 4) {
        dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
        wei_tag = utils::pick(ndims - 3, gOIw2o4i2o, gOIhw2o4i2o, gOIdhw2o4i2o);
    } else {
        dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw8o16i2o, gOIw8o16i2o,
                OIhw8o16i2o, gOIhw8o16i2o, OIdhw8o16i2o, gOIdhw8o16i2o);
    }

    if (diff_src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag));
    } else if (curr_src_tag != dat_tag)
        return status::unimplemented;
    jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag));
    } else if (curr_dst_tag != dat_tag)
        return status::unimplemented;
    jcp.dst_tag = dat_tag;

    if (weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;
    }

    bool args_ok = true && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    jcp.nb_ic = utils::div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = utils::div_up(jcp.oc, jcp.oc_block);

    jcp.ur_w = jcp.stride_w;

    /* Maximum number of registers available for result accumulation and delta
       dst data. One additional register is reserved for weights data. */
    const int max_regs
            = isa_has_bf16(jcp.isa) ? 31 : 26; /* In case of cpx emulation
                                                  additional 5 registers are
                                                  reserved */
    int l_overflow = nstl::max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.iw % jcp.ur_w))
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());

    /* Find the best blocking with maximum number of compute instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs) /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    {
        jcp.kernel_kind = expl_bcast;
        int best_compute_pipeline_length = 0;
        const int max_ic_blocks = 4;
        for (int b = 1; b <= max_ic_blocks; b++) {
            if (jcp.nb_ic % b != 0) continue;

            for (int u = jcp.stride_w; u * b + u / jcp.stride_w <= max_regs
                    && u < jcp.iw + jcp.stride_w;
                    u += jcp.stride_w) {
                int ur_w = nstl::min(u, jcp.iw);
                /* maximum 1 step with l_overflow so far */
                if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw)
                    continue;
                int pipeline_length = utils::div_up(ur_w, jcp.stride_w) * b;
                if (pipeline_length > best_compute_pipeline_length
                        || (pipeline_length == best_compute_pipeline_length
                                && jcp.ur_w < ur_w)) {
                    jcp.ur_w = ur_w;
                    jcp.nb_ic_blocking = b;
                    best_compute_pipeline_length = pipeline_length;
                }
            }
        }
        if (best_compute_pipeline_length == 0) /* can't find
                                                  appropriate blocking */
            return status::unimplemented;
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (is_iw_threading_available(jcp)) {
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_units = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        float no_iw_block_eff
                = (float)work_units / rnd_up(work_units, jcp.nthr);

        // current design of generate() requires iw_block >= 2 * ur_w
        const int min_iw_block = jcp.ur_w * 2;
        int iw_threads = jcp.nthr / math::gcd(work_units, jcp.nthr);
        int iw_block = nstl::max(min_iw_block,
                rnd_up(jcp.iw, jcp.ur_w * iw_threads) / iw_threads);
        int nb_iw = div_up(jcp.iw, iw_block);

        float block_eff = (float)jcp.iw / rnd_up(jcp.iw, iw_block);
        work_units = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih * nb_iw;
        float work_eff = (float)work_units / rnd_up(work_units, jcp.nthr);
        float iw_block_eff = block_eff * work_eff;

        const int iw_thread_min_size = 16 * 128;
        const float iw_block_cost = 20.0;
        float block_overhead = nstl::max(0.0f, 1.0f - iw_block_cost / iw_block);

        bool iw_thread_useful = no_iw_block_eff < block_overhead * iw_block_eff
                && jcp.ic_block * jcp.iw > iw_thread_min_size;

        if (iw_thread_useful) {
            jcp.iw_block = iw_block;
            jcp.nb_iw = nb_iw;
        }
    }

    if (l_overflow * jcp.stride_w > jcp.ur_w) return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.ur_w_tail))
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

const int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        od_step_comeback_pointers() {
    Label kd_comeback_label;
    mov(kj, reg_kd_count);
    L(kd_comeback_label);
    {
        sub(reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        sub(reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        oh_step_comeback_pointers() {
    Label kh_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label);
    {
        sub(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
        sub(reg_kernel, get_kernel_offset(0, jcp.kw));
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_extern(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int src_offset, int kernel_offset,
                int ddst_offset, bool is_tail) {
    assert(!is_src_layout_nxc() && !is_ddst_layout_nxc());
    int kw = jcp.kw;
    bool no_src_pad = jcp.is_1stconv && !jcp.transpose_src;
    const int ddst_zmm_base_idx = 24;
    const int num_ddst_zmm_regs = !isa_has_bf16(jcp.isa) ? 2 : 4;
    const int zmm_src_reg = ddst_zmm_base_idx + num_ddst_zmm_regs;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };
    auto zmm_ddst = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        return Zmm(ddst_zmm_base_idx + i_iw % num_ddst_zmm_regs);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        auto local_offset = get_kernel_offset(i_ic, i_kw);
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };
    auto src_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0,
                            bool vnni_bcast = false) {
        auto local_offset = get_src_offset(i_ic, i_iw);
        return EVEX_compress_addr(
                reg_src, local_offset + src_offset + extra_offset, vnni_bcast);
    };
    auto ddst_addr = [=](int i_ur) {
        auto ow_scale = 2;
        return EVEX_compress_addr(
                reg_ddst, get_ddst_offset(ow_scale * i_ur) + ddst_offset);
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }
    assert(ur_w % 2 == 0);
    auto steps = ur_w / 2;

    const int str_w = jcp.stride_w;
    const int underflow_boundary = -1;
    int i_iw_shift = jcp.tr_ow - ur_w - ((jcp.l_pad != pad_l) ? jcp.l_pad : 0);
    const int overflow_boundary = jcp.iw - 1 - i_iw_shift;

    for (int s = 0; s < str_w; s++) {
        const int kw_start = s;
        assert(jcp.tr_iw % str_w == 0);
        const int src_stride_w_shift = jcp.tr_iw / str_w;
        for (int i_ur = 0; i_ur < steps; i_ur++) {
            auto zmm = zmm_ddst(i_ur);
            vmovdqu16(zmm, ddst_addr(i_ur));

            for (int i_kw = kw_start; i_kw < kw; i_kw += str_w) {
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    int i_iw = 2 * i_ur + (i_kw * (jcp.dilate_w + 1)) / str_w
                            + s * src_stride_w_shift;
                    bool underflow = false;
                    bool overflow = false;
                    if (no_src_pad) {
                        i_iw = i_iw - pad_l;
                        underflow = i_iw <= underflow_boundary;
                        overflow = is_tail && i_iw >= overflow_boundary;
                    }

                    auto src = Zmm(zmm_src_reg);
                    auto acc = zmm_ker(i_kw, i_ic);
                    auto ddst = zmm_ddst(i_ur);
                    if (underflow || overflow || !isa_has_bf16(jcp.isa)) {
                        assert(ddst != src);
                        assert(acc != src);
                    }
                    assert(ddst != acc);
                    if (underflow || overflow) {
                        if (underflow && i_iw == underflow_boundary)
                            vpbroadcastw(src | everyother_shift_mask | T_z,
                                    src_addr(i_iw + 1, i_ic, 0));
                        else if (overflow && i_iw == overflow_boundary)
                            vpbroadcastw(src | everyother_mask | T_z,
                                    src_addr(i_iw, i_ic, 0));
                        else
                            continue;

                        if (!isa_has_bf16(jcp.isa))
                            bf16_emu_->vdpbf16ps(acc, ddst, src);
                        else
                            vdpbf16ps(acc, ddst, src);
                    } else if (!isa_has_bf16(jcp.isa)) {
                        vpbroadcastd(src, src_addr(i_iw, i_ic, 0));
                        bf16_emu_->vdpbf16ps(acc, ddst, src);
                    } else
                        vdpbf16ps(acc, ddst, src_addr(i_iw, i_ic, 0, true));
                }
            }
        }
        for (int i_kw = kw_start; i_kw < kw; i_kw += str_w) {
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                auto addr = ker_addr(i_kw, i_ic);
                auto zmm = zmm_ker(i_kw, i_ic);
                vaddps(zmm, zmm, addr);
                vmovups(addr, zmm);
            }
        }
    }
}

int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::interleave_w_reorder_size(
        int ur_w) {
    const int reorder_block = 16;
    return rnd_up(jcp.stride_w * (ur_w - 1) + jcp.kw, reorder_block);
}
int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        interleave_w_reorder_bytes(int ur_w) {
    return 2 * jcp.typesize_in * interleave_w_reorder_size(ur_w);
}
int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::interleave_stack_size(
        int ur_w, int ic_block_step) {
    return ic_block_step * interleave_w_reorder_bytes(ur_w);
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_interleave(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int src_offset, int kernel_offset,
                int ddst_offset, bool is_tail) {
    // Only supports nchw format src
    assert(jcp.is_1stconv && !jcp.transpose_src);
    int kw = jcp.kw;
    const int ddst_zmm_base_idx = 24;
    const int in_zmm_base_idx = 24;
    const int num_ddst_zmm_regs = !isa_has_bf16(jcp.isa) ? 2 : 4;
    //const int num_in_zmm_regs = 8;
    const int zmm_src_reg = ddst_zmm_base_idx + num_ddst_zmm_regs;
    const int reorder_block = 16;
    const int reorder_size = interleave_w_reorder_size(ur_w);
    const int reorder_bytes = interleave_w_reorder_bytes(ur_w);
    const int stack_size = interleave_stack_size(ur_w, ic_block_step);
    if (stack_size > ic_block_step_stack_size) {
        // This is a guard. Ideally it is never used, but is included to defend
        // against overlooked edge cases.
        assert(stack_size <= ic_block_step_stack_size);
        sub(rsp, stack_size - ic_block_step_stack_size);
    }

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };
    auto zmm_ddst = [=](int i_iw) {
        return Zmm(ddst_zmm_base_idx + i_iw % num_ddst_zmm_regs);
    };
    auto zmm_in = [=](int i_iw, int i_ic, bool stride_reg) {
        int stride = stride_reg ? 1 : 0;
        return Zmm(in_zmm_base_idx + 4 * (i_ic % 2) + 2 * (i_iw % 2) + stride);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        auto local_offset = get_kernel_offset(i_ic, i_kw);
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };
    auto src_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0,
                            bool vnni_bcast = false) {
        int local_offset = i_ic * reorder_bytes + 2 * jcp.typesize_in * i_iw;
        return EVEX_compress_addr(rsp, local_offset, vnni_bcast);
    };
    auto ddst_addr = [=](int i_ur) {
        auto ow_scale = 2;
        return EVEX_compress_addr(
                reg_ddst, get_ddst_offset(ow_scale * i_ur) + ddst_offset);
    };
    auto load_src_to_stack = [=](int i_iw, int i_ic, Opmask mask,
                                     bool mask_empty, Opmask stride_mask,
                                     bool stride_mask_empty) {
        auto local_offset = get_src_offset(i_ic, i_iw);
        int stack_offset
                = i_ic * reorder_bytes + 2 * jcp.typesize_in * (i_iw + pad_l);

        auto zmm = zmm_in(i_iw, i_ic, false);
        auto zmm_stride = zmm_in(i_iw, i_ic, true);
        auto base_addr
                = EVEX_compress_addr(reg_src, local_offset + src_offset, false);
        auto stride_addr = EVEX_compress_addr(reg_src,
                local_offset + src_offset + get_src_offset(0, jcp.stride_w));
        auto stack_addr = EVEX_compress_addr(rsp, stack_offset);
        assert(IMPLICATION(mask_empty, stride_mask_empty));
        if (mask_empty) {
            vpxord(zmm, zmm, zmm);
        } else {
            vpmovzxwd(zmm | mask | T_z, base_addr);
        }
        if (!stride_mask_empty) {
            vpmovzxwd(zmm_stride | stride_mask | T_z, stride_addr);
            vpslld(zmm_stride, zmm_stride, 16);
            vpord(zmm, zmm, zmm_stride);
        }
        vmovdqu16(stack_addr, zmm);
    };

    assert(ur_w % 2 == 0);
    auto steps = ur_w / 2;

    const int str_w = jcp.stride_w;
    int i_iw_shift = str_w * (jcp.tr_ow - ur_w)
            - ((jcp.l_pad != pad_l) ? jcp.l_pad : 0);
    const int overflow_boundary
            = is_tail ? jcp.iw - i_iw_shift : str_w * (ur_w - 1) + kw - pad_l;

    // Calculate padding required by the data reorder using 32 byte loads
    int reorder_overflow = reorder_size - pad_l - overflow_boundary;
    int reorder_stride_overflow = reorder_overflow + str_w;
    reorder_overflow = nstl::max(0, reorder_overflow);
    reorder_stride_overflow = nstl::max(0, reorder_stride_overflow);
    int reorder_pad_r = reorder_overflow % reorder_block;
    int reorder_stride_pad_r = reorder_stride_overflow % reorder_block;
    if (reorder_stride_overflow >= reorder_size && reorder_stride_pad_r == 0) {
        assert(reorder_stride_overflow == reorder_size);
        reorder_stride_pad_r = reorder_block;
    }
    reorder_overflow -= reorder_pad_r;
    reorder_stride_overflow -= reorder_stride_pad_r;

    int pad_l_mask = (0xffff << pad_l) & 0xffff;
    int pad_l_mask_strided
            = (0xffff << (pad_l >= str_w ? (pad_l - str_w) : 0)) & 0xffff;
    int pad_r_mask = 0xffff >> reorder_pad_r;
    int pad_r_mask_strided = 0xffff >> (reorder_stride_pad_r);
    pad_r_mask = pad_r_mask & 0xffff;

    // Setup masks to load and reorder data
    if (reorder_size - reorder_stride_overflow > reorder_block) {
        // Overflow and underflow happen in different data reorder rounds
        kxnorw(overflow_stride_mask, overflow_stride_mask,
                overflow_stride_mask);
        kshiftlw(underflow_mask, overflow_stride_mask, pad_l);
        kshiftlw(underflow_stride_mask, overflow_stride_mask,
                pad_l >= str_w ? pad_l - str_w : 0);
        kshiftrw(overflow_mask, overflow_stride_mask, reorder_pad_r);
        kshiftrw(overflow_stride_mask, overflow_stride_mask,
                reorder_stride_pad_r);
    } else if (reorder_size - reorder_overflow > reorder_block) {
        // Overflow and underflow happen in the same round for loading the data
        // at the stride offset.
        kxnorw(overflow_mask, overflow_mask, overflow_mask);
        kshiftlw(underflow_mask, overflow_mask, pad_l);
        kshiftrw(overflow_mask, overflow_mask, reorder_pad_r);
        mov(reg_tmp.cvt32(), pad_l_mask_strided & pad_r_mask_strided);
        kmovw(underflow_stride_mask, reg_tmp.cvt32());
    } else {
        // Overflow and underflow happen in the same round for all data loads
        mov(reg_tmp.cvt32(), pad_l_mask & pad_r_mask);
        kmovw(underflow_mask, reg_tmp.cvt32());
        mov(reg_tmp.cvt32(), pad_l_mask_strided & pad_r_mask_strided);
        kmovw(underflow_stride_mask, reg_tmp.cvt32());
    }

    // Load and reorder data to the stack
    int reorder_start = -pad_l;
    int reorder_end = reorder_size - pad_l;
    for (int i_iw = reorder_start; i_iw < reorder_end; i_iw += reorder_block) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            Opmask mask, stride_mask;
            bool mask_empty, stride_mask_empty;
            // Performing this reorder on the stack may not be (always) optimal.
            // There are a couple of methods involving externally reordering the
            // data that were not considered due to time constraints. The first
            // is to transpose similar to the extern method. The other is to
            // perform the same interleave transform used here. The tradeoff
            // between these methods is the transpose method does not lend
            // itself to SIMD instructions (except possibly for some specific
            // strides) since the data is not blocked. The transform performed
            // here does, but uses twice as much data since
            // most data elements are duplicated.

            if (i_iw == reorder_start) {
                mask = underflow_mask;
                mask_empty = false;
                if (pad_l_mask == 0) mask_empty = true;
            } else if (i_iw + reorder_overflow >= reorder_end) {
                mask_empty = true;
            } else if (i_iw + reorder_block + reorder_overflow >= reorder_end) {
                mask = overflow_mask;
                mask_empty = false;
                if (pad_r_mask == 0) mask_empty = true;
            } else {
                mask = m_ffffffff;
                mask_empty = false;
            }
            if (i_iw == reorder_start) {
                stride_mask = underflow_stride_mask;
                stride_mask_empty = false;
                if (pad_l_mask_strided == 0) mask_empty = true;
            } else if (i_iw + reorder_stride_overflow >= reorder_end) {
                stride_mask_empty = true;
            } else if (i_iw + reorder_block + reorder_stride_overflow
                    >= reorder_end) {
                stride_mask = overflow_stride_mask;
                stride_mask_empty = false;
                if (pad_r_mask_strided == 0) mask_empty = true;
            } else {
                stride_mask = m_ffffffff;
                stride_mask_empty = false;
            }
            load_src_to_stack(i_iw, i_ic, mask, mask_empty, stride_mask,
                    stride_mask_empty);
        }
    }

    // Initialize kernel accumulators. It should sometimes be possible to skip
    // initializing and storing this data between calls to this function.
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }

    // Calculate this blocks contribution
    for (int i_ur = 0; i_ur < steps; i_ur++) {
        auto zmm = zmm_ddst(i_ur);
        vmovdqu16(zmm, ddst_addr(i_ur));

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                int i_iw = 2 * i_ur * str_w + i_kw;
                auto acc = zmm_ker(i_kw, i_ic);
                auto ddst = zmm_ddst(i_ur);

                const bool isa_supports_bf16 = isa_has_bf16(jcp.isa);
                auto src_stack_addr
                        = src_addr(i_iw, i_ic, 0, isa_supports_bf16);

                if (isa_supports_bf16)
                    vdpbf16ps(acc, ddst, src_stack_addr);
                else {
                    auto src = Zmm(zmm_src_reg);
                    vpbroadcastd(src, src_stack_addr);
                    bf16_emu_->vdpbf16ps(acc, ddst, src);
                }
            }
        }
    }

    // Store kernel accumulators
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr = ker_addr(i_kw, i_ic);
            auto zmm = zmm_ker(i_kw, i_ic);
            vaddps(zmm, zmm, addr);
            vmovups(addr, zmm);
        }
    }

    if (stack_size > ic_block_step_stack_size) {
        // This is a guard. Ideally it is never used, but is included to defend
        // against overlooked edge cases.
        add(rsp, stack_size - ic_block_step_stack_size);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        convert_src_to_vnni_format(
                int ur_w, int pad_l, int pad_r, int src_offset) {
    Reg64 reg_trans_tmp = r11;
    const int ic_tail = jcp.ic_tail;
    mov(EVEX_compress_addr(rsp, trans_tmp_offset), reg_trans_tmp);

    mov(reg_trans_tmp, dst_prm_table);
    vmovups(get_perm_reg(), ptr[reg_trans_tmp]);

    mov(reg_trans_tmp, EVEX_compress_addr(rsp, trans_tmp_offset));
    const int max_regs = 16;
    if (ic_tail) {
        Label skip_tail_mask;
        cmp(reg_icb, jcp.simd_w);
        jge(skip_tail_mask);
        kandd(m_0000ffff, m_0000ffff, m_0000_ic_tail);
        kandd(m_ffff0000, m_ffff0000, m_ic_tail_0000);
        L(skip_tail_mask);
    }
    for (int src_count = 0;
            sizeof_cacheline * src_count < permw_stack_size(ur_w);
            src_count++) {
        int i_ur = nstl::min(src_count, ur_w - 2);
        int i_kw = src_count - i_ur;
        int buffer_offset = permw_buffer_start + src_count * 64;
        auto bcast_values = Zmm(src_count % max_regs);
        bool check = check_borders(ur_w, pad_l, pad_r, i_ur, i_kw);
        if (check) {
            if (is_src_layout_nxc()) {
                int iw_1, iw_2;
                get_w_positions(ur_w, pad_l, pad_r, i_ur, i_kw, iw_1, iw_2);
                if (iw_1 == -1)
                    vxorpd(bcast_values, bcast_values, bcast_values);
                else {
                    dim_t local_src_offset = src_offset
                            + get_src_offset(
                                    0, filter_w_to_src(i_kw, i_ur, pad_l));
                    vmovdqu16(bcast_values | m_0000ffff | T_z,
                            ptr[reg_src + local_src_offset]);
                }
                if (iw_2 != -1) {
                    dim_t local_src_offset = src_offset - 32
                            + get_src_offset(
                                    0, filter_w_to_src(i_kw, i_ur + 1, pad_l));
                    vmovdqu16(bcast_values | m_ffff0000,
                            ptr[reg_src + local_src_offset]);
                }
            } else {
                Opmask load_mask;
                get_load_mask(ur_w, pad_l, pad_r, i_ur, i_kw, load_mask);

                dim_t local_src_offset = src_offset
                        + get_src_offset(0, filter_w_to_src(i_kw, i_ur, pad_l));
                vmovdqu16(bcast_values | load_mask | T_z,
                        ptr[reg_src + local_src_offset]);
            }
            vpermw(bcast_values, get_perm_reg(), bcast_values);
        } else {
            vpxord(bcast_values, bcast_values, bcast_values);
        }
        vmovups(ptr[rsp + buffer_offset], bcast_values);
    }
    if (ic_tail) {
        // Reset-back the masks
        kxnorw(m_0000ffff, m_0000ffff, m_0000ffff);
        kshiftld(m_ffff0000, m_0000ffff, 16);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        may_be_set_oc_tail_mask() {
    if (jcp.oc_tail) {
        Label skip_tail_mask;
        cmp(dword[param + GET_OFF(load_work)], jcp.simd_w);
        jge(skip_tail_mask);
        kandd(m_0000ffff, m_0000ffff, m_0000_oc_tail);
        kandd(m_ffff0000, m_ffff0000, m_oc_tail_0000);
        L(skip_tail_mask);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        may_be_reset_oc_tail_mask() {
    if (jcp.oc_tail) {
        // Reset-back the masks
        kxnorw(m_0000ffff, m_0000ffff, m_0000ffff);
        kshiftld(m_ffff0000, m_0000ffff, 16);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_vpermw_expl(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int src_offset, int kernel_offset,
                int ddst_offset, bool is_tail) {
    assert(!jcp.is_1stconv); // This method does not support nchw data
    int kw = jcp.kw;
    int src_count = 0;
    int ic_block_step_idx = src_offset / (jcp.typesize_in * ic_block_step);
    const int max_regs = (!isa_has_bf16(jcp.isa)) ? 26 : 31;
    int src_pl_len = kw;
    const int diff_dst_pl_start_reg_idx = ic_block_step * (kw + src_pl_len);
    const int diff_dst_pl_len = max_regs - diff_dst_pl_start_reg_idx;

    auto get_diff_wei_reg_idx
            = [=](int i_kw, int i_ic) { return i_kw * ic_block_step + i_ic; };
    auto get_src_reg_idx = [=](int i_iw, int i_ic) {
        return kw * ic_block_step + (i_iw % src_pl_len) * ic_block_step + i_ic;
    };
    auto get_diff_dst_reg_idx = [=](int i_ur) {
        return diff_dst_pl_start_reg_idx + (i_ur / 2) % diff_dst_pl_len;
    };

    may_be_set_oc_tail_mask();
    auto load_dst = [=](int c) {
        bool is_tail = ur_w % 2 && c * 2 + 2 >= ur_w;
        bool is_ddst_nxc = is_ddst_layout_nxc();
        auto offset = get_ddst_offset(c * 2) + ddst_offset;

        Opmask load_mask = is_ddst_nxc || is_tail ? m_0000ffff : m_ffffffff;
        vmovdqu16(Zmm(get_diff_dst_reg_idx(2 * c)) | load_mask | T_z,
                EVEX_compress_addr(reg_ddst, offset));

        if (is_ddst_nxc && !is_tail) {
            offset += get_ddst_offset(1) - 32;
            vmovdqu16(Zmm(get_diff_dst_reg_idx(2 * c)) | m_ffff0000,
                    EVEX_compress_addr(reg_ddst, offset));
        }
        vpermw(Zmm(get_diff_dst_reg_idx(2 * c)), get_perm_reg(),
                Zmm(get_diff_dst_reg_idx(2 * c)));
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vpxord(Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                    Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                    Zmm(get_diff_wei_reg_idx(i_kw, i_ic)));

    auto get_bcast_ptr = [=](int i_ur, int i_kw, int ic) {
        int scale = 2 * jcp.typesize_in;
        return rsp + b_ic * scale + permw_buffer_start + (i_ur + i_kw) * 64
                + jcp.typesize_in * 2
                * (ic_block_step_idx * ic_block_step + ic);
    };
    int src_count_last = 0;
    for (int i_ur = 0; i_ur < ur_w; i_ur += 2) {
        if (i_ur == 0) {
            for (int dst_count = 0;
                    dst_count < nstl::min(diff_dst_pl_len, div_up(ur_w, 2));
                    dst_count++) {
                load_dst(dst_count);
            }
            for (src_count = 0; src_count < src_pl_len; src_count++) {
                int _i_ur = src_count / kw;
                int _i_kw = src_count % kw;
                if (check_borders(ur_w, pad_l, pad_r, _i_ur, _i_kw))
                    for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                        vbroadcastss(Zmm(get_src_reg_idx(src_count, i_ic)),
                                ptr[get_bcast_ptr(_i_ur, _i_kw, i_ic)]);
                    }
            }
            src_count_last = src_count;
        } else {
            int diff_dst_load_idx = i_ur + 2 * (diff_dst_pl_len - 1);
            if (diff_dst_load_idx < ur_w) load_dst(diff_dst_load_idx / 2);
            for (src_count = i_ur; src_count < i_ur + src_pl_len; src_count++) {
                if (src_count < src_count_last) continue;
                int _i_ur = (src_count - i_ur) / kw + i_ur;
                int _i_kw = (src_count - i_ur) % kw;
                if (check_borders(ur_w, pad_l, pad_r, _i_ur, _i_kw))
                    for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                        vbroadcastss(Zmm(get_src_reg_idx(src_count, i_ic)),
                                ptr[get_bcast_ptr(_i_ur, _i_kw, i_ic)]);
                    }
            }
            src_count_last = src_count;
        }
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur + i_kw;
            if (check_borders(ur_w, pad_l, pad_r, i_ur, i_kw)) {
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(
                                Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                                Zmm(get_diff_dst_reg_idx(i_ur)),
                                Zmm(get_src_reg_idx(i_iw, i_ic)));
                    } else {
                        vdpbf16ps(Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                                Zmm(get_diff_dst_reg_idx(i_ur)),
                                Zmm(get_src_reg_idx(i_iw, i_ic)));
                    }
                }
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto l_offset = get_kernel_offset(i_ic, i_kw);
            vaddps(Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                    EVEX_compress_addr(reg_kernel, l_offset + kernel_offset));
        }

    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto l_offset = get_kernel_offset(i_ic, i_kw);
            vmovups(EVEX_compress_addr(reg_kernel, l_offset + kernel_offset),
                    Zmm(get_diff_wei_reg_idx(i_kw, i_ic)));
        }
    }

    may_be_reset_oc_tail_mask();
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_vpermw(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int src_offset, int kernel_offset,
                int ddst_offset, bool is_tail) {
    assert(!jcp.is_1stconv); // This method does not support nchw data
    int kw = jcp.kw;

    int dst_count = 0;

    int ic_block_step_idx = src_offset / (jcp.typesize_in * ic_block_step);

    int pipeline_length = (isa_has_bf16(jcp.isa))
            ? nstl::max(1, nstl::min(4, ur_w / 2))
            : 1;
    may_be_set_oc_tail_mask();

    const int dst_off_reg = (!isa_has_bf16(jcp.isa)) ? 26 : 31;
    auto load_dst = [=](int c) {
        bool is_tail = ur_w % 2 && c * 2 + 2 >= ur_w;
        bool is_ddst_nxc = is_ddst_layout_nxc();
        auto offset = get_ddst_offset(2 * c) + ddst_offset;

        Opmask load_mask = is_ddst_nxc || is_tail ? m_0000ffff : m_ffffffff;
        vmovdqu16(Zmm(dst_off_reg - c % pipeline_length) | load_mask | T_z,
                EVEX_compress_addr(reg_ddst, offset));

        if (is_ddst_nxc && !is_tail) {
            offset += get_ddst_offset(1) - 32;
            vmovdqu16(Zmm(dst_off_reg - c % pipeline_length) | m_ffff0000,
                    EVEX_compress_addr(reg_ddst, offset));
        }
        vpermw(Zmm(dst_off_reg - c % pipeline_length), get_perm_reg(),
                Zmm(dst_off_reg - c % pipeline_length));
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                    EVEX_compress_addr(reg_kernel,
                            get_kernel_offset(i_ic, i_kw) + kernel_offset));

    for (dst_count = 0; dst_count < pipeline_length; dst_count++) {
        load_dst(dst_count);
    }
    auto get_bcast_ptr = [=](int i_ur, int i_kw, int ic) {
        int scale = 2 * jcp.typesize_in;
        return rsp + b_ic * scale + permw_buffer_start + (i_ur + i_kw) * 64
                + jcp.typesize_in * 2
                * (ic_block_step_idx * ic_block_step + ic);
    };

    for (int i_ur = 0; i_ur < ur_w; i_ur += 2) {
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            if (check_borders(ur_w, pad_l, pad_r, i_ur, i_kw)) {
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    if (!isa_has_bf16(jcp.isa)) {
                        auto zmm_src = Zmm(28);
                        vpbroadcastd(
                                zmm_src, ptr[get_bcast_ptr(i_ur, i_kw, i_ic)]);
                        bf16_emu_->vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic),
                                Zmm(dst_off_reg - dst_count % pipeline_length),
                                zmm_src);
                    } else {
                        vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic),
                                Zmm(dst_off_reg - dst_count % pipeline_length),
                                zword_b[get_bcast_ptr(i_ur, i_kw, i_ic)]);
                    }
                }
            }
        }
        if (dst_count * 2 < ur_w) load_dst(dst_count);
        dst_count++;
    }
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto l_offset = get_kernel_offset(i_ic, i_kw);
            vmovups(EVEX_compress_addr(reg_kernel, l_offset + kernel_offset),
                    Zmm(i_kw * ic_block_step + i_ic));
        }
    }

    may_be_reset_oc_tail_mask();
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_diff_bias_init() {
    auto reg_unit_val = reg_tmp.cvt16();
    mov(reg_unit_val, 0x3f80); // bf16 value of 1.
    vpbroadcastw(vreg_bias_unit, reg_unit_val);

    mov(reg_tmp, ptr[param + GET_OFF(bias)]);
    vmovups(vreg_bias_acc, ptr[reg_tmp]);

    if (jcp.uses_permw_transposition) {
        mov(reg_tmp, dst_prm_table);
        vmovups(get_perm_reg(), ptr[reg_tmp]);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_diff_bias_row(
        bool is_partial) {
    if (!jcp.with_bias) return;
    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    Label skip_label;
    test(reg_tmp, FLAG_IC_FIRST);
    jz(skip_label, T_NEAR);

    may_be_set_oc_tail_mask();

    if (is_partial) compute_diff_bias_init();

    auto compute_step = [&](bool is_tail) {
        if (jcp.transpose_dst) {
            UNUSED(is_tail);
            vmovups(vreg_bias_ddst, ptr[reg_ddst]);
        } else {
            auto vreg_ddst_load = is_ddst_layout_nxc() || is_tail
                    ? vreg_bias_ddst | m_0000ffff | T_z
                    : vreg_bias_ddst;
            vmovdqu16(vreg_ddst_load, ptr[reg_ddst]);
            if (is_ddst_layout_nxc() && !is_tail) {
                const int shift_16_elems = 16 * jcp.typesize_in;
                vmovdqu16(vreg_bias_ddst | m_ffff0000,
                        ptr[reg_ddst + get_ddst_offset(1) - shift_16_elems]);
            }
            vpermw(vreg_bias_ddst, get_perm_reg(), vreg_bias_ddst);
        }
        if (!isa_has_bf16(jcp.isa))
            bf16_emu_->vdpbf16ps(vreg_bias_acc, vreg_bias_ddst, vreg_bias_unit);
        else
            vdpbf16ps(vreg_bias_acc, vreg_bias_ddst, vreg_bias_unit);
    };

    Label ow_loop, ow_tail;
    int niters = jcp.tr_ow / 2;
    if (niters > 0) {
        mov(reg_tmp, jcp.tr_ow / 2);
        L(ow_loop);
        compute_step(false);
        add(reg_ddst, get_ddst_offset(2));
        sub(reg_tmp, 1);
        jnz(ow_loop, T_NEAR);
    }
    if (jcp.tr_ow % 2) compute_step(true);

    if (niters > 0) sub(reg_ddst, get_ddst_offset(2 * niters));

    if (is_partial) {
        mov(reg_tmp, ptr[param + GET_OFF(bias)]);
        vmovups(ptr[reg_tmp], vreg_bias_acc);
    }

    may_be_reset_oc_tail_mask();

    L(skip_label);
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        maybe_compute_diff_bias() {
    // In harness_3d_reduction case calculation of diff_bias is called
    // for every ow row separately to be aligned with od loop in
    // compute_od_loop_common()
    if (!jcp.with_bias || jcp.harness == harness_3d_reduction) return;
    mov(reg_tmp, ptr[param + GET_OFF(flags)]);

    Label skip_label;
    test(reg_tmp, FLAG_IC_FIRST);
    jz(skip_label, T_NEAR);

    switch (jcp.harness) {
        case harness_2d_reduction:
            mov(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            sub(reg_oj, ptr[param + GET_OFF(os_index_begin)]);
            break;
        case harness_mb_reduction:
        case harness_compute_full_spatial: mov(reg_oj, jcp.oh); break;
        case harness_3d_reduction:
        default: assert(!"Invalid harness type");
    }

    compute_diff_bias_init();

    cmp(reg_oj, 0);
    jle(skip_label, T_NEAR); // nothing to do
    Label bias_loop;
    L(bias_loop);
    {
        compute_diff_bias_row(false);
        add(reg_ddst, get_ddst_offset(0, 1));

        sub(reg_oj, 1);
        jnz(bias_loop, T_NEAR);
    }

    mov(reg_tmp, ptr[param + GET_OFF(bias)]);
    vmovups(ptr[reg_tmp], vreg_bias_acc);

    // restore reg_ddst value
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);

    L(skip_label);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int src_offset,
        int kernel_offset, int ddst_offset, bool is_tail) {

    if (jcp.uses_permw_transposition)
        if (jcp.kernel_kind == expl_bcast)
            compute_ic_block_step_vpermw_expl(ur_w, pad_l, pad_r, ic_block_step,
                    src_offset, kernel_offset, ddst_offset, is_tail);
        else
            compute_ic_block_step_vpermw(ur_w, pad_l, pad_r, ic_block_step,
                    src_offset, kernel_offset, ddst_offset, is_tail);
    else if (jcp.is_1stconv && !jcp.transpose_src && jcp.stride_w > 1)
        compute_ic_block_step_interleave(ur_w, pad_l, pad_r, ic_block_step,
                src_offset, kernel_offset, ddst_offset, is_tail);
    else
        compute_ic_block_step_extern(ur_w, pad_l, pad_r, ic_block_step,
                src_offset, kernel_offset, ddst_offset, is_tail);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::get_ur_w(
        int &ur_w, int &ur_w_tail, int &ur_w_trips) {
    if (jcp.tr_ow <= max_ur_w) {
        ur_w = jcp.tr_ow;
        ur_w_tail = 0;
        ur_w_trips = 1;
        return;
    }

    int r_pad = 0;
    if (!jcp.transpose_src) {
        // If jcp.transpose_src, the buffer has physical padding
        int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
        r_pad = nstl::max(0,
                calculate_end_padding(
                        jcp.l_pad, jcp.tr_ow, jcp.tr_iw, jcp.stride_w, ext_kw));
    }
    int l_pad = (jcp.transpose_src) ? 0 : jcp.l_pad;
    ur_w = max_ur_w;
    ur_w_trips = jcp.tr_ow / ur_w;
    ur_w_tail = jcp.tr_ow % ur_w;
    if ((ur_w_tail == 0 && jcp.r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            int ur_w_tail_total = ur_w + ur_w_tail;
            ur_w = (ur_w_tail_total % 4 == 0) ? ur_w_tail / 2
                                              : ur_w_tail / 2 + 1;
            ur_w_tail = ur_w_tail_total - ur_w;
            if (l_pad > ur_w / 2) {
                ur_w = (l_pad % 2 == 0) ? l_pad : l_pad + 1;
                ur_w_tail = ur_w_tail_total - ur_w;
            } else if (r_pad > ur_w_tail) {
                ur_w_tail = (r_pad % 2 == 0) ? r_pad : r_pad + 1;
                ur_w = ur_w_tail_total - ur_w_tail;
            }
        }
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_oh_step_unroll_ow_icblock(int ic_block_step) {
    Label kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int ic_tail = jcp.ic_tail;
    int ow = jcp.tr_ow;
    int r_pad = 0;
    int ur_w, ur_w_tail, ur_w_trips;
    get_ur_w(ur_w, ur_w_tail, ur_w_trips);
    assert(ur_w_tail == 0 && ur_w_trips == 1);

    if (!jcp.transpose_src) {
        // If jcp.transpose_src, the buffer has physical padding
        int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
        int iw = jcp.tr_iw;
        r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, ow, iw, jcp.stride_w, ext_kw));
    }
    int l_pad = (jcp.transpose_src) ? 0 : jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
        // icb loop is supported for nxc layout only
        assert(IMPLICATION(generate_icb_loop,
                is_src_layout_nxc() && is_ddst_layout_nxc()));
        Label icb_block_label, icb_block_label_end;
        if (generate_icb_loop || ic_tail) {
            mov(ptr[rsp + icb_loop_ker_ptr], reg_kernel);
            mov(ptr[rsp + icb_loop_src_ptr], reg_src);
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
            L(icb_block_label);
        }

        if (jcp.uses_permw_transposition) {
            convert_src_to_vnni_format(ur_w, l_pad, r_pad, 0);
            xor_(b_ic, b_ic);
        }

        const int ic_tail_loop_work = rnd_up(ic_tail, ic_block_step);
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int src_offset = get_src_offset(i_b_ic, 0);
            compute_ic_block_step(ur_w, l_pad, r_pad, ic_block_step, src_offset,
                    get_kernel_offset(i_b_ic, 0), 0, true);
            if (generate_icb_loop || ic_tail) sub(reg_icb, ic_block_step);
            // We relax the boundary for reg_icb, as the src is already
            // converted to vnni_format with appropriate padding either through
            // transpose_src or convert_to_src_to_vnni_format. We can safely
            // allow compute_ic_block_step overstep as it operates on buffer
            // instead of src.
            if (ic_tail && i_b_ic + ic_block_step == ic_tail_loop_work) {
                assert(jcp.transpose_src || jcp.uses_permw_transposition);
                cmp(reg_icb, 0);
                jle(icb_block_label_end, T_NEAR);
            }
        }
        L(icb_block_label_end);

        const auto src_icb_loop_shift_bytes = get_src_offset(ic_block, 0);
        const auto kernel_icb_loop_shift_bytes
                = get_kernel_offset(0, jcp.kd * jcp.kh * jcp.kw);
        if (generate_icb_loop) {
            add(reg_src, src_icb_loop_shift_bytes);
            safe_add(reg_kernel, kernel_icb_loop_shift_bytes, reg_long_offt);

            assert(jcp.uses_permw_transposition);
            cmp(reg_icb, 0);
            jg(icb_block_label, T_NEAR);
        }

        if (generate_icb_loop || ic_tail) {
            // restore pointers
            mov(reg_kernel, ptr[rsp + icb_loop_ker_ptr]);
            mov(reg_src, ptr[rsp + icb_loop_src_ptr]);
        }

        add(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
        add(reg_kernel, get_kernel_offset(0, jcp.kw));
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        add(aux_reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_oh_step_unroll_ow(int ic_block_step) {
    Label kh_label, ic_block_label, kd_label;

    int ic_block = jcp.ic_block;
    const int ic_tail = jcp.ic_tail;
    int ow = jcp.tr_ow;

    int r_pad = 0;
    int ur_w, ur_w_tail, ur_w_trips;
    get_ur_w(ur_w, ur_w_tail, ur_w_trips);
    assert(ur_w_tail == 0 && ur_w_trips == 1);

    if (!jcp.transpose_src) {
        // If jcp.transpose_src, the buffer has physical padding
        int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
        int iw = jcp.tr_iw;
        r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, ow, iw, jcp.stride_w, ext_kw));
    }
    int l_pad = (jcp.transpose_src) ? 0 : jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        size_t src_offset = get_src_offset(ic_block_step, 0);

        const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
        // icb loop is supported for nxc layout only
        assert(IMPLICATION(generate_icb_loop,
                is_src_layout_nxc() && is_ddst_layout_nxc()));
        Label icb_block_label, icb_block_label_end;
        if (generate_icb_loop || ic_tail) {
            mov(ptr[rsp + icb_loop_ker_ptr], reg_kernel);
            mov(ptr[rsp + icb_loop_src_ptr], reg_src);
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
            L(icb_block_label);
        }

        xor_(b_ic, b_ic);
        if (jcp.uses_permw_transposition) {
            convert_src_to_vnni_format(ow, l_pad, r_pad, 0);
            xor_(b_ic, b_ic);
        }

        L(ic_block_label);
        {
            compute_ic_block_step(
                    ur_w, l_pad, r_pad, ic_block_step, 0, 0, 0, true);
            assert(jcp.ic_block % jcp.ic_block_step == 0);
            safe_add(reg_src, src_offset, reg_long_offt);
            add(reg_kernel, get_kernel_offset(ic_block_step, 0));
            add(b_ic, ic_block_step);
            if (generate_icb_loop || ic_tail) sub(reg_icb, ic_block_step);
            // We relax the boundary for reg_icb, as the src is already
            // converted to vnni_format with appropriate padding either through
            // transpose_src or convert_to_src_to_vnni_format. We can safely
            // allow compute_ic_block_step overstep as it operates on buffer
            // instead of src.
            if (ic_tail) {
                assert(jcp.transpose_src || jcp.uses_permw_transposition);
                cmp(reg_icb, 0);
                jle(icb_block_label_end, T_NEAR);
            }
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        L(icb_block_label_end);

        if (jcp.uses_permw_transposition) {
            if (generate_icb_loop || ic_tail) {
                // substract pointer shift made within ic block loop
                // and move to next ic block
                safe_add(reg_kernel,
                        get_kernel_offset(-ic_block, jcp.kd * jcp.kh * jcp.kw),
                        reg_long_offt);

                cmp(reg_icb, 0);
                jg(icb_block_label, T_NEAR);
                // restore pointers
                mov(reg_kernel, ptr[rsp + icb_loop_ker_ptr]);
                mov(reg_src, ptr[rsp + icb_loop_src_ptr]);

                add(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
                add(reg_kernel, get_kernel_offset(0, jcp.kw));
            } else {
                add(reg_src,
                        get_src_offset(0, 0, filter_h_to_src(1))
                                - jcp.typesize_in * ic_block);
            }
        } else if (ic_tail) {
            // restore pointers
            mov(reg_kernel, ptr[rsp + icb_loop_ker_ptr]);
            mov(reg_src, ptr[rsp + icb_loop_src_ptr]);

            add(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
            add(reg_kernel, get_kernel_offset(0, jcp.kw));
        } else if (jcp.is_1stconv && !jcp.transpose_src) {
            // Fixup reg_src to point to the correct location
            safe_add(reg_src,
                    get_src_offset(0, 0, filter_h_to_src(1))
                            - src_offset * (jcp.ic_block / ic_block_step),
                    reg_long_offt);
        } else {
            if (jcp.dilate_h > 0)
                add(reg_src, get_src_offset(0, 0, jcp.dilate_h));
        }
        if (!generate_icb_loop && !ic_tail)
            // substract pointer shift made within ic block loop
            // and move to next kh index
            add(reg_kernel, get_kernel_offset(-ic_block, jcp.kw));
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        add(aux_reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        int ic_block_step) {
    Label kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int ic_tail = jcp.ic_tail;
    int ow = jcp.tr_ow;
    int r_pad = 0;
    if (!jcp.transpose_src) {
        // If jcp.transpose_src, the buffer has physical padding
        int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
        int iw = jcp.tr_iw;
        r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, ow, iw, jcp.stride_w, ext_kw));
    }
    int l_pad = (jcp.transpose_src) ? 0 : jcp.l_pad;

    int ur_w, ur_w_trips, ur_w_tail;
    get_ur_w(ur_w, ur_w_tail, ur_w_trips);
    assert(l_pad <= ur_w);
    assert(r_pad <= ur_w_tail);

    auto src_comeback
            = get_src_offset(0, filter_w_to_src(0, ur_w_trips * ur_w, l_pad));
    auto ddst_comeback = get_ddst_offset(ur_w_trips * ur_w);

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
    }

    bool use_kh_ic_ow_loop_order = !jcp.uses_permw_transposition;
    if (use_kh_ic_ow_loop_order) {
        assert(!jcp.uses_permw_transposition);

        auto ic_loop = [=](int ic_block_step) {
            Label ow_block_label;
            // create a local copy
            int ur_w_blocks = ur_w_trips;
            auto src_offset = get_src_offset(ic_block_step, 0);
            if (l_pad != 0) {
                ur_w_blocks--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_src,
                        get_src_offset(0, filter_w_to_src(0, ur_w, l_pad)));
                add(reg_ddst, get_ddst_offset(ur_w));
            }

            if (ur_w_blocks > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label);
                {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_src,
                            get_src_offset(0, filter_w_to_src(0, ur_w, 0)));
                    add(reg_ddst, get_ddst_offset(ur_w));

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_blocks);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) {
                compute_ic_block_step(
                        ur_w_tail, 0, r_pad, ic_block_step, 0, 0, 0, true);
            }

            sub(reg_src, src_comeback);
            sub(reg_ddst, ddst_comeback);

            safe_add(reg_src, src_offset, reg_long_offt);
            add(reg_kernel, get_kernel_offset(ic_block_step, 0));
        };

        mov(kj, reg_kh);
        L(kh_label);
        {
            Label ic_tail_label, skip_ic_tail_offset_compensation;
            if (ic_tail) {
                // It appears currently, generate_icb_loop is not enabled here,
                // implying at most one icb is processed.
                assert(jcp.nb_ic_blocking_max == 1);
                mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
            } else {
                mov(reg_icb, ic_block);
            }

            L(ic_block_label);
            {
                ic_loop(ic_block_step);
                sub(reg_icb, ic_block_step);
                // We relax the boundary for reg_icb, as the src is already
                // converted to vnni_format with appropriate padding either
                // through transpose_src or convert_to_src_to_vnni_format. We
                // can safely allow compute_ic_block_step overstep as it
                // operates on buffer instead of src.
                if (ic_tail) {
                    assert(jcp.transpose_src || jcp.uses_permw_transposition);
                }
                cmp(reg_icb, 0);
                jg(ic_block_label, T_NEAR);
            }

            if (ic_tail) {
                mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
                cmp(reg_icb, jcp.simd_w);
                je(skip_ic_tail_offset_compensation);
                add(reg_kernel,
                        get_kernel_offset(
                                jcp.ic_block - rnd_up(ic_tail, ic_block_step),
                                0));
                safe_add(reg_src,
                        get_src_offset(0, 0, filter_h_to_src(1))
                                - get_src_offset(
                                        rnd_up(ic_tail, ic_block_step), 0),
                        reg_long_offt);
                L(skip_ic_tail_offset_compensation);
            }
            if (jcp.is_1stconv && !jcp.transpose_src) {
                // Fixup reg_src to point to the correct location
                auto src_offset = get_src_offset(ic_block_step, 0);
                safe_add(reg_src,
                        get_src_offset(0, 0, filter_h_to_src(1))
                                - src_offset * (jcp.ic_block / ic_block_step),
                        reg_long_offt);
            } else if (jcp.dilate_h > 0) {
                add(reg_src, get_src_offset(0, 0, jcp.dilate_h));
            }
            // substract pointer shift made within ic block loop
            // and move to next kh index
            add(reg_kernel, get_kernel_offset(-ic_block, jcp.kw));
            dec(kj);
            cmp(kj, 0);
            jg(kh_label, T_NEAR);
        }
    } else {
        assert(!jcp.is_1stconv);
        auto src_icbstep_shift = get_src_offset(1, 0);

        auto ic_loop = [=](int ic_block_step) {
            int ic_work = ic_block;
            Label ow_block_label, ic_block_label_padl, ic_block_label_general,
                    ic_block_label_tail;
            int ur_w_blocks = ur_w_trips;
            if (l_pad != 0) {
                ur_w_blocks--;
                xor_(b_ic, b_ic);
                if (jcp.uses_permw_transposition) {
                    convert_src_to_vnni_format(ur_w, l_pad, 0, 0);
                }
                L(ic_block_label_padl);
                {
                    compute_ic_block_step(
                            ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                    safe_add(reg_src, src_icbstep_shift * ic_block_step,
                            reg_long_offt);
                    add(reg_kernel, get_kernel_offset(ic_block_step, 0));

                    add(b_ic, ic_block_step);
                    cmp(b_ic, ic_work);
                    jl(ic_block_label_padl, T_NEAR);
                }
                safe_sub(reg_src, src_icbstep_shift * ic_work, reg_long_offt);
                sub(reg_kernel, get_kernel_offset(ic_work, 0));
                add(reg_src,
                        get_src_offset(0, filter_w_to_src(0, ur_w, l_pad)));
                add(reg_ddst, get_ddst_offset(ur_w));
            }

            if (ur_w_blocks > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label);
                {
                    if (jcp.uses_permw_transposition) {
                        convert_src_to_vnni_format(ur_w, 0, 0, 0);
                    }
                    xor_(b_ic, b_ic);
                    L(ic_block_label_general);
                    {
                        compute_ic_block_step(
                                ur_w, 0, 0, ic_block_step, 0, 0, 0);
                        safe_add(reg_src, src_icbstep_shift * ic_block_step,
                                reg_long_offt);
                        add(reg_kernel, get_kernel_offset(ic_block_step, 0));

                        add(b_ic, ic_block_step);
                        cmp(b_ic, ic_work);
                        jl(ic_block_label_general, T_NEAR);
                    }
                    safe_sub(reg_src, src_icbstep_shift * ic_work,
                            reg_long_offt);
                    sub(reg_kernel, get_kernel_offset(ic_work, 0));
                    add(reg_src, get_src_offset(0, filter_w_to_src(0, ur_w)));
                    add(reg_ddst, get_ddst_offset(ur_w));

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_blocks);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) {
                if (jcp.uses_permw_transposition) {
                    convert_src_to_vnni_format(ur_w_tail, 0, r_pad, 0);
                }
                xor_(b_ic, b_ic);
                L(ic_block_label_tail);
                {
                    compute_ic_block_step(
                            ur_w_tail, 0, r_pad, ic_block_step, 0, 0, 0, true);
                    safe_add(reg_src, src_icbstep_shift * ic_block_step,
                            reg_long_offt);
                    add(reg_kernel, get_kernel_offset(ic_block_step, 0));

                    add(b_ic, ic_block_step);
                    cmp(b_ic, ic_work);
                    jl(ic_block_label_tail, T_NEAR);
                }
                safe_sub(reg_src, src_icbstep_shift * ic_work, reg_long_offt);
                sub(reg_kernel, get_kernel_offset(ic_work, 0));
            }

            sub(reg_src, src_comeback);
            sub(reg_ddst, ddst_comeback);
        };

        mov(kj, reg_kh);
        L(kh_label);
        {
            const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
            // icb loop is supported for nxc layout only
            assert(IMPLICATION(generate_icb_loop,
                    is_src_layout_nxc() && is_ddst_layout_nxc()));
            Label icb_block_label, icb_block_label_cb, ic_tail_loop_label;

            if (generate_icb_loop) {
                mov(ptr[rsp + icb_loop_ker_ptr], reg_kernel);
                mov(ptr[rsp + icb_loop_src_ptr], reg_src);
            }
            if (ic_tail || generate_icb_loop)
                mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
            L(icb_block_label);

            ic_loop(ic_block_step);

            if (generate_icb_loop) {
                add(reg_src, get_src_offset(ic_block, 0));
                safe_add(reg_kernel,
                        get_kernel_offset(0, jcp.kd * jcp.kh * jcp.kw),
                        reg_long_offt);
                sub(reg_icb, ic_block);
                cmp(reg_icb, 0);
                jg(icb_block_label, T_NEAR);
            }

            if (generate_icb_loop) {
                // restore pointers
                mov(reg_kernel, ptr[rsp + icb_loop_ker_ptr]);
                mov(reg_src, ptr[rsp + icb_loop_src_ptr]);
            }

            add(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
            add(reg_kernel, get_kernel_offset(0, jcp.kw));
            dec(kj);
            cmp(kj, 0);
            jg(kh_label, T_NEAR);
        }
    }
    if (jcp.ndims == 5) {
        add(aux_reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        add(aux_reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_oh_step_disp() {
    int ic_block_step = jcp.ic_block_step;

    bool too_large_to_unroll = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
            && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    int ow = jcp.tr_ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_src = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        mov(EVEX_compress_addr(rsp, kd_count_offset), reg_kd_count);
        mov(aux_reg_src, reg_src);
        mov(aux_reg_kernel, reg_kernel);
    }
    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll) {
        compute_oh_step_unroll_ow_icblock(ic_block_step);
    } else if (ow <= max_ur_w) {
        compute_oh_step_unroll_ow(ic_block_step);
    } else {
        compute_oh_step_common(ic_block_step);
    }

    // In harness_3d_reduction case calculation of diff_bias is called
    // for every ow row separately to be aligned with od loop in
    // compute_od_loop_common()
    if (jcp.harness == harness_3d_reduction) compute_diff_bias_row();
    if (jcp.ndims == 5) {
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
        mov(reg_kd_count, EVEX_compress_addr(rsp, kd_count_offset));
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::maybe_zero_kernel() {
    if (jcp.harness == harness_compute_full_spatial && !jcp.with_bias) return;
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    if (jcp.with_bias) {
        Label skip_bias_zeroing;
        mov(reg_tmp, ptr[param + GET_OFF(flags)]);
        test(reg_tmp, FLAG_IC_FIRST);
        jz(skip_bias_zeroing, T_NEAR);

        mov(reg_tmp, ptr[param + GET_OFF(bias)]);
        vmovups(ptr[reg_tmp], zero);

        L(skip_bias_zeroing);
        if (jcp.harness == harness_compute_full_spatial)
            jmp(skip_zeroing, T_NEAR);
    }

    const size_t kernel_block_bytes
            = get_kernel_offset(0, jcp.kw * jcp.kh * jcp.kd);
    Label icb_block_label, icb_block_label_cb;

    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    // icb loop is supported for nxc layout only
    assert(IMPLICATION(
            generate_icb_loop, is_src_layout_nxc() && is_ddst_layout_nxc()));
    if (generate_icb_loop) {
        mov(ptr[rsp + icb_loop_ker_ptr], reg_kernel);
        mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
        L(icb_block_label);
    }

    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop);
    {
        assert(get_kernel_offset(1, 0) == cpu_isa_traits<avx512_core>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp + get_kernel_offset(ic1, 0)],
                    zero);
        add(reg_tmp, get_kernel_offset(jcp.ic_block, 0));
        cmp(reg_tmp, kernel_block_bytes);
        jnz(zeroing_loop);
    }

    if (generate_icb_loop) {
        add(reg_kernel, kernel_block_bytes);
        sub(reg_icb, jcp.ic_block);
        cmp(reg_icb, 0);
        jg(icb_block_label, T_NEAR);
        // restore pointer
        mov(reg_kernel, ptr[rsp + icb_loop_ker_ptr]);
    }

    L(skip_zeroing);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::compute_oh_loop_common(
        bool is_partial) {
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    auto filter_step_size = get_kernel_offset(0, jcp.kw);
    auto src_step_size = get_src_offset(0, 0, 1);
    auto ddst_step_size = get_ddst_offset(0, 1);
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_label_end,
            oh_tpad_tail_label, oh_tpad_tail_label_end, oh_bpad_label,
            oh_bpad_label_end, oh_dilate_label_shift, oh_dilate_label_noshift,
            oh_dilate_label_end, oh_dilate_setup_label_shift,
            oh_dilate_setup_label_noshift, oh_dilate_setup_label_end;

    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int oh_body_end = div_up(t_pad + jcp.ih - ext_kh + 1, stride_h);
    int oh_head_end = nstl::min(div_up(t_pad, stride_h), oh_body_end);
    int oh_head_overflow_end = div_up(t_pad, stride_h);
    int oh_tail_end = jcp.oh;

    int body_src_start_offset = (stride_h - (t_pad % stride_h)) % stride_h;
    int ih_body_end
            = nstl::max(-t_pad + oh_body_end * stride_h, body_src_start_offset);

    if (is_partial)
        mov(reg_oj, ptr[param + GET_OFF(os_index_begin)]);
    else
        xor_(reg_oj, reg_oj);

    /* Compute 'top' edge */
    if (t_pad > 0) {
        if (is_partial) {
            cmp(reg_oj, oh_head_overflow_end);
            jge(oh_tpad_tail_label_end, T_NEAR);
        }
        const int overflow
                = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_kh = jcp.kh - overflow - underflow;

        // Setup reg_kh, reg_kernel, and reg_src
        mov(reg_kh, initial_kh);
        add(reg_kernel, filter_step_size * underflow);
        if (is_dilated) {
            const int tail = t_pad % dilate_h;
            const int shift = tail == 0 ? 0 : dilate_h - tail;
            mov(reg_ih_shift, shift);
            if (!is_partial) mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
            add(reg_src, src_step_size * shift);
        }

        if (is_partial) {
            Label head_setup, head_setup_finish;
            cmp(reg_oj, 0);
            je(head_setup_finish, T_NEAR);
            mov(reg_oj_setup, reg_oj);

            L(head_setup);
            if (is_dilated) {
                inc(reg_ih_shift);
                cmp(reg_ih_shift, dilate_h);
                jl(oh_dilate_setup_label_shift, T_NEAR);
                // unshift src as new kernel element enters
                sub(reg_src, src_step_size * (dilate_h - 1));
                xor_(reg_ih_shift, reg_ih_shift);
            }
            // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
            add(reg_kh, stride_h);
            sub(reg_kernel, filter_step_size * stride_h);
            if (is_dilated) {
                jmp(oh_dilate_setup_label_noshift, T_NEAR);
                L(oh_dilate_setup_label_shift);
                // shift src as old kernel element progresses
                add(reg_src, src_step_size * stride_h);
                L(oh_dilate_setup_label_noshift);
            }
            sub(reg_oj_setup, 1);
            jg(head_setup, T_NEAR);
            L(head_setup_finish);

            if (is_dilated) mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
            if (oh_head_end < oh_head_overflow_end) {
                cmp(reg_oj, oh_head_end);
                jge(oh_tpad_label_end, T_NEAR);
            }
        }

        //Setup reg_kernel
        // If dilated, shift src ptr
        // Loop
        L(oh_tpad_label);
        compute_oh_step_disp();
        add(reg_ddst, ddst_step_size);
        if (is_dilated) {
            mov(reg_ih_shift, ptr[rsp + ih_dilate_shift]);
            inc(reg_ih_shift);
            mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
            cmp(reg_ih_shift, dilate_h);
            jl(oh_dilate_label_shift, T_NEAR);
            // unshift src as new kernel element enters
            sub(reg_src, src_step_size * (dilate_h - 1));
            xor_(reg_ih_shift, reg_ih_shift);
            mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
        }
        // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
        add(reg_kh, stride_h);
        sub(reg_kernel, filter_step_size * stride_h);
        if (is_dilated) {
            jmp(oh_dilate_label_noshift, T_NEAR);
            L(oh_dilate_label_shift);
            // shift src as old kernel element progresses
            add(reg_src, src_step_size * stride_h);
            L(oh_dilate_label_noshift);
        }
        inc(reg_oj);

        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }
        cmp(reg_oj, oh_head_end);
        jl(oh_tpad_label, T_NEAR);

        L(oh_tpad_label_end);
        // need second loop to process kernel if it is larger than the src
        // (does not apply to dilations as they must have unit stride)
        if (oh_head_end < oh_head_overflow_end) {
            assert(!is_dilated);

            cmp(reg_oj, oh_head_overflow_end);
            jge(oh_tpad_tail_label_end, T_NEAR);

            mov(reg_kh, jcp.ih);
            L(oh_tpad_tail_label);
            {
                compute_oh_step_disp();
                add(reg_ddst, ddst_step_size);
                sub(reg_kernel, filter_step_size * stride_h);

                inc(reg_oj);

                if (is_partial) {
                    cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
                    jge(oh_bpad_label_end, T_NEAR);
                }
                cmp(reg_oj, oh_head_overflow_end);
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        if (body_src_start_offset != 0) {
            add(reg_kernel, filter_step_size * body_src_start_offset);
            add(reg_src, src_step_size * body_src_start_offset);
        }
        L(oh_tpad_tail_label_end);
    }

    if (is_partial) {
        cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
        jge(oh_bpad_label_end, T_NEAR);
    }
    cmp(reg_oj, oh_body_end);
    jge(oh_label_end, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, jcp.kh);
    L(oh_label);
    {
        compute_oh_step_disp();
        add(reg_src, src_step_size * stride_h);
        add(reg_ddst, ddst_step_size);

        inc(reg_oj);

        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }

        cmp(reg_oj, oh_body_end);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            // Assumes unit stride for dilations
            mov(reg_kh, jcp.kh - 1);
            xor_(reg_ih_shift, reg_ih_shift);
        } else {
            assert(jcp.dilate_h == 0);
            mov(reg_kh, jcp.ih - ih_body_end);
        }
        if (is_partial) {
            lea(reg_oj_setup,
                    ptr[reg_oj - nstl::max(oh_body_end, oh_head_overflow_end)]);
            if (stride_h == 1 && !is_dilated) {
                sub(reg_kh, reg_oj_setup);
            } else {
                Label body_setup, body_setup_finish, dilate_skip;
                cmp(reg_oj_setup, 0);
                je(body_setup_finish, T_NEAR);

                L(body_setup);
                if (is_dilated) {
                    inc(reg_ih_shift);
                    cmp(reg_ih_shift, dilate_h);
                    jl(dilate_skip, T_NEAR);
                    xor_(reg_ih_shift, reg_ih_shift);
                }
                sub(reg_kh, stride_h);
                L(dilate_skip);
                sub(reg_oj_setup, 1);
                jg(body_setup, T_NEAR);
                L(body_setup_finish);
            }
        }

        if (is_dilated) mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_src, src_step_size * stride_h);
            add(reg_ddst, ddst_step_size);

            if (is_dilated) {
                mov(reg_ih_shift, ptr[rsp + ih_dilate_shift]);
                inc(reg_ih_shift);
                mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
                cmp(reg_ih_shift, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_ih_shift, reg_ih_shift);
                mov(ptr[rsp + ih_dilate_shift], reg_ih_shift);
            }
            sub(reg_kh, stride_h);
            L(oh_dilate_label_end);
            inc(reg_oj);
            if (is_partial) {
                cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
                jge(oh_bpad_label_end, T_NEAR);
            }
            cmp(reg_oj, oh_tail_end);
            jl(oh_bpad_label, T_NEAR);
        }
    }
    L(oh_bpad_label_end);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::compute_od_loop_common(
        bool is_partial) {
    assert(jcp.harness == harness_3d_reduction);

    const int src_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const auto filter_shift = get_kernel_offset(0, jcp.kh * jcp.kw);
    const auto src_shift = get_src_offset(0, 0, jcp.ih);
    const auto ddst_shift = get_ddst_offset(0, jcp.oh);

    const int kd_front_pad = nstl::max(0, jcp.f_pad);
    const int kd_back_pad = nstl::max(0, jcp.kd - jcp.f_pad - jcp.id);

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    /* initially offset 'kd' by f_pad */
    mov(reg_src_d, ptr[param + GET_OFF(src)]);
    mov(reg_ddst_d, ptr[param + GET_OFF(dst)]);

    if (is_partial) {
        add(reg_kernel, ptr[param + GET_OFF(kd_offset)]);
        mov(reg_d_index, ptr[param + GET_OFF(os_index_begin)]);
        mov(reg_kd_count, ptr[param + GET_OFF(kd_padding)]);
    } else {
        const int kd_padding = jcp.kd - kd_front_pad - kd_back_pad;
        const int kd_offset = get_kernel_offset(
                0, nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw);
        add(reg_kernel, kd_offset);
        xor_(reg_d_index, reg_d_index);
        mov(reg_kd_count, kd_padding);
    }

    cmp(reg_kd_count, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kd
    if (is_partial)
        cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    else
        cmp(reg_d_index, jcp.od);
    jge(loop_end_label, T_NEAR); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_src, reg_src_d);
    mov(reg_ddst, reg_ddst_d);

    mov(EVEX_compress_addr(rsp, src_d_offset), reg_src_d);
    mov(EVEX_compress_addr(rsp, ddst_d_offset), reg_ddst_d);
    mov(EVEX_compress_addr(rsp, d_index_offset), reg_d_index);

    compute_oh_loop_common();

    mov(reg_src_d, EVEX_compress_addr(rsp, src_d_offset));
    mov(reg_ddst_d, EVEX_compress_addr(rsp, ddst_d_offset));
    mov(reg_d_index, EVEX_compress_addr(rsp, d_index_offset));

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {
        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with src */
        const int src_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp(reg_kd_count, src_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and src */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int src_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * src_corr);
                add(reg_src_d, src_shift * src_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov(reg_kd_count, src_ker_overlap);
        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, src_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count,
                jcp.id + jcp.f_pad - src_backpad_overlap * jcp.stride_d);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_src_d, src_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_ddst_d, ddst_shift);
    inc(reg_d_index);
    if (is_partial)
        cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    else
        cmp(reg_d_index, jcp.od);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_full_spat_loop() {
    // General code layout:
    //
    // Blocking over OH -- top level
    // (Reduces L2 pressure; not very useful right now)
    //  Loop over all KHxKW kernel -- emit_kh_kw_loop()
    //    Loop over OH block -- emit_h_loop()
    //      Loop over OW blocks -- emit_fma_block()
    //      (Supports both fully unrolled and partially unrolled
    //      versions to reduce code size)
    //          Loop over OW block -- emit_fma_step()

    auto src_row_size = get_src_offset(0, 0, 1);
    auto ddst_row_size = get_ddst_offset(0, 1);
    auto row_size = src_row_size + ddst_row_size;

    int h_block_size = jcp.oh;
    int h_last_block_size = h_block_size;
    int min_h_block_size = nstl::max(1, nstl::max(jcp.b_pad, jcp.t_pad));
    auto working_set_size = row_size * h_block_size;

    if (working_set_size > full_spat_max_working_set_size) {
        assert(full_spat_opt_working_set_size < full_spat_max_working_set_size);

        while (working_set_size > full_spat_opt_working_set_size
                && h_block_size >= min_h_block_size) {
            for (int i = 2; i <= h_block_size; i++)
                if (i == h_block_size)
                    h_block_size = h_block_size / 2;
                else if (h_block_size % i == 0) {
                    h_block_size = h_block_size / i;
                    break;
                }
            working_set_size = row_size * h_block_size;
        }
        h_block_size = nstl::max(min_h_block_size, h_block_size);
        h_last_block_size = jcp.oh % h_block_size;
        if (h_last_block_size < jcp.b_pad) h_last_block_size += h_block_size;
    }

    Opmask reg_h_block = k1;
    Reg64 reg_kh = rax;
    Reg64 reg_kw = rbx;
    Reg64 reg_tmp = abi_not_param1;
    Reg32 reg_tmp_w = reg_tmp.cvt32();
    Reg64 reg_ohs = rdx;
    Reg64 reg_ihs = rsi;
    Reg64 reg_h = r8;
    Reg64 reg_i = r9;
    Reg64 reg_j = r10;

    Reg64 reg_src = r13;
    Reg64 reg_ddst = r14;
    Reg64 reg_ker = r15;

    Reg64 reg_src_save = abi_param1;
    Reg64 reg_ddst_save = reg_tmp;

    auto zmm_ddst = [&](int oi) { return Zmm(24 + oi % 8); };
    auto zmm_ker = [&](int ic1) { return Zmm(ic1); };
    auto src_addr = [&](int oi, int ic1) {
        return zword_b[reg_src + get_src_offset(ic1, oi)];
    };
    auto ddst_addr = [&](int oi) {
        auto ow_per_oc = 2;
        return ptr[reg_ddst + get_ddst_offset(ow_per_oc * oi)];
    };
    auto ker_addr
            = [&](int ic1) { return ptr[reg_ker + get_kernel_offset(ic1, 0)]; };

    auto emit_block = [&]() {
        auto pad_ow = jcp.tr_ow;
        int ow_per_oc = 2;
        int def_step_size = 16;
        bool has_w_tail = pad_ow % def_step_size != 0;
        bool full_w_unroll = pad_ow / def_step_size < 2 + has_w_tail;

        auto emit_step = [&](int ur_ow, bool is_w_tail) {
            int tail_size = pad_ow % ur_ow;
            int this_ur_ow = (is_w_tail && tail_size) ? tail_size : ur_ow;
            auto numloads = 1;

            assert(this_ur_ow % ow_per_oc == 0);
            int steps = this_ur_ow / ow_per_oc;
            for (int oi_base = 0; oi_base < steps; oi_base += numloads) {
                for (int oi_offset = 0; oi_offset < numloads; oi_offset++) {
                    int oi = oi_base + oi_offset;
                    if (oi < steps) {
                        vmovups(zmm_ddst(oi), ddst_addr(oi));
                    } else {
                        auto zmm = zmm_ddst(oi);
                        vpxord(zmm, zmm, zmm);
                    }
                }

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    vdpbf16ps(zmm_ker(ic1), zmm_ddst(oi_base),
                            src_addr(ow_per_oc * oi_base, ic1));
                }
            }
        };

        if (full_w_unroll) {
            emit_step(pad_ow, true);
        } else {
            Label w_loop;
            int num_w_iters = pad_ow / def_step_size;
            mov(reg_i, num_w_iters);
            L(w_loop);
            {
                emit_step(def_step_size, false);
                add(reg_src, get_src_offset(0, def_step_size));
                add(reg_ddst, get_ddst_offset(def_step_size));
                sub(reg_i, 1);
                jnz(w_loop);
            }
            if (has_w_tail) { emit_step(def_step_size, true); }
            // reset reg_src and reg_ddst because emit_h_loop expects
            // unmodified pointers
            int w_offset = num_w_iters * def_step_size;
            sub(reg_src, get_src_offset(0, w_offset));
            sub(reg_ddst, get_ddst_offset(w_offset));
        }
    };

    auto emit_h_loop = [&]() {
        Label h_loop, skip_h_loop;
        mov(reg_j, 1);
        cmp(reg_j, reg_h);
        je(skip_h_loop, T_NEAR);
        L(h_loop);
        {
            emit_block();

            add(reg_src, get_src_offset(0, 0, 1));
            add(reg_ddst, get_ddst_offset(0, 1));
            add(reg_j, 1);
            cmp(reg_j, reg_h);
            jb(h_loop);
        }
        L(skip_h_loop);

        emit_block();
    };

    auto emit_kh_kw_loop = [&](bool is_first_block, bool is_last_block) {
        xor_(reg_kh, reg_kh);
        Label kh_loop, kh_loop_end;

        int oh_block_size = (is_last_block) ? h_last_block_size : h_block_size;
        // NB: this is correct because we only support t_pad = kh / 2 and thus
        // ih == oh
        int ih_block_size = oh_block_size
                + (!is_first_block + !is_last_block) * jcp.t_pad;

        L(kh_loop);
        {
            if (is_first_block) {
                xor_(reg_tmp, reg_tmp);
                mov(reg_ohs, jcp.t_pad);
                sub(reg_ohs, reg_kh);
                cmovb(reg_ohs, reg_tmp);

                mov(reg_ihs, reg_ohs);
                sub(reg_ihs, jcp.t_pad);
                add(reg_ihs, reg_kh);
            } else {
                xor_(reg_ohs, reg_ohs);
                mov(reg_ihs, reg_kh);
            }

            mov(reg_tmp, oh_block_size);
            sub(reg_tmp, reg_ohs);
            mov(reg_h, ih_block_size);
            sub(reg_h, reg_ihs);
            cmp(reg_tmp, reg_h);
            cmovb(reg_h, reg_tmp);

            Label kh_loop_work;
            cmp(reg_h, 0);
            jg(kh_loop_work, T_NEAR);

            // empty h loop for this jcp.kh:
            // - set the ddst to 0 if necessary
            // - move ker pt
            // - jump to the end
            sub(reg_h, 1);
            Label skip_ker_zeroing;

            // The reg_ker ptr has highest bit set if the ddst needs to be
            // zeroed. Those who have byte-aligned their data will suffer the
            // consequences :(
            // TODO: move the flag to a mask register? (Roma)
            test(reg_ker, 1);
            jz(skip_ker_zeroing, T_NEAR);

            Label zeroing_loop;
            vpxord(zmm0, zmm0, zmm0);
            and_(reg_ker, ~1); // temporarily clear the zeroing flag
            mov(reg_tmp, jcp.kw);
            L(zeroing_loop);
            {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
                    vmovups(ker_addr(ic1), zmm0);
                add(reg_ker, get_kernel_offset(jcp.ic_block, 0));
                sub(reg_tmp, 1);
                jnz(zeroing_loop, T_NEAR);
            }
            // restore the zeroing flag (it will be cleared after the end of
            // emit_kh_kw_loop, but we may need it until then)
            or_(reg_ker, 1);
            jmp(kh_loop_end, T_NEAR);

            L(skip_ker_zeroing);
            add(reg_ker, get_kernel_offset(0, jcp.kw));
            jmp(kh_loop_end, T_NEAR);

            L(kh_loop_work);

            mul_by_const(reg_ihs, reg_tmp, get_src_offset(0, 0, 1));
            mul_by_const(reg_ohs, reg_tmp, get_ddst_offset(0, 1));

            add(reg_src, reg_ihs);
            add(reg_ddst, reg_ohs);

            Label kw_loop;
            xor_(reg_kw, reg_kw);
            L(kw_loop);
            {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vpxord(zmm, zmm, zmm);
                }

                mov(reg_ddst_save, reg_ddst);
                mov(reg_src_save, reg_src);
                lea(reg_src, ptr[reg_src + reg_kw * jcp.typesize_in]);

                emit_h_loop();

                mov(reg_ddst, reg_ddst_save);
                mov(reg_src, reg_src_save);

                Label do_store;
                // The reg_ker ptr has highest bit set if the ddst needs to
                // be zeroed. Those who have byte-aligned their data will
                // suffer the consiquences :(
                mov(reg_tmp, reg_ker);
                and_(reg_ker, ~1);
                test(reg_tmp, 1);
                jnz(do_store, T_NEAR);

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vaddps(zmm, ker_addr(ic1));
                }

                L(do_store);
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vmovups(ker_addr(ic1), zmm);
                }

                mov(reg_ker, reg_tmp);
                add(reg_ker, get_kernel_offset(jcp.ic_block, 0));
                add(reg_kw, 1);
                cmp(reg_kw, jcp.kw);
                jl(kw_loop);
            }

            sub(reg_src, reg_ihs);
            sub(reg_ddst, reg_ohs);

            L(kh_loop_end);
            add(reg_kh, 1);
            cmp(reg_kh, jcp.kh);
            jl(kh_loop);
        }
    };

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);
    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    or_(reg_ker, reg_tmp);

    bool single_kh_kw_loop = (h_last_block_size == jcp.oh);

    auto src_row_step = get_src_offset(0, 0, 1);
    auto first_src_block_step = src_row_step * (h_block_size - jcp.t_pad);
    auto ddst_block_step = get_ddst_offset(0, h_block_size);

    emit_kh_kw_loop(true, single_kh_kw_loop);

    if (!single_kh_kw_loop) {
        auto ker_reset_offset = get_kernel_offset(0, jcp.kw * jcp.kh);
        sub(reg_ker, ker_reset_offset);
        and_(reg_ker, ~1); // Clear the zeroing flag for subsequent updates

        add(reg_src, first_src_block_step);
        add(reg_ddst, ddst_block_step);

        int num_innermost_iters
                = (jcp.oh - h_last_block_size) / h_block_size - 1;
        if (num_innermost_iters > 0) {
            Label h_block_loop;

            mov(reg_tmp_w, num_innermost_iters);
            kmovw(reg_h_block, reg_tmp_w);
            L(h_block_loop);
            {
                emit_kh_kw_loop(false, false);
                sub(reg_ker, ker_reset_offset);
                add(reg_src, src_row_step * h_block_size);
                add(reg_ddst, ddst_block_step);

                kmovw(reg_tmp_w, reg_h_block);
                sub(reg_tmp_w, 1);
                kmovw(reg_h_block, reg_tmp_w);
                jnz(h_block_loop);
            }
        }

        emit_kh_kw_loop(false, true);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::compute_loop() {
    Reg64 reg_mask_load = r11;
    if (jcp.uses_permw_transposition) {

        mov(reg_mask_load.cvt32(), 0xffffffff);
        kmovd(m_ffffffff, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), 0x0000ffff);
        kmovd(m_0000ffff, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), 0xffff0000);
        kmovd(m_ffff0000, reg_mask_load.cvt32());
        const int oc_tail = jcp.oc_tail;
        if (oc_tail) {
            mov(reg_mask_load.cvt32(), (1 << oc_tail) - 1);
            kmovd(m_0000_oc_tail, reg_mask_load.cvt32());
            kshiftld(m_oc_tail_0000, m_0000_oc_tail, 16);
        }
        const int ic_tail = jcp.ic_tail;
        if (ic_tail) {
            mov(reg_mask_load.cvt32(), (1 << ic_tail) - 1);
            kmovd(m_0000_ic_tail, reg_mask_load.cvt32());
            kshiftld(m_ic_tail_0000, m_0000_ic_tail, 16);
        }
    } else if (jcp.is_1stconv && !jcp.transpose_src) {
        if (jcp.stride_w == 1) {
            int ieveryother_mask = 0x55555555;
            mov(reg_mask_load.cvt32(), ieveryother_mask);
            kmovd(everyother_mask, reg_mask_load.cvt32());
            kshiftld(everyother_shift_mask, everyother_mask, 1);
        } else {
            mov(reg_mask_load.cvt32(), 0xffffffff);
            kmovd(m_ffffffff, reg_mask_load.cvt32());
        }
    }

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    maybe_zero_kernel();
    maybe_compute_diff_bias();

    switch (jcp.harness) {
        case harness_3d_reduction: compute_od_loop_common(true); break;
        case harness_2d_reduction: compute_oh_loop_common(true); break;
        case harness_mb_reduction: compute_oh_loop_common(); break;
        case harness_compute_full_spatial: compute_full_spat_loop(); break;
        default: assert(!"Invalid harness type");
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::setup_stack_space() {

    if ((jcp.is_1stconv && !jcp.transpose_src && jcp.stride_w > 1)
            || jcp.uses_permw_transposition) {
        int ur_w, ur_w_tail, ur_w_trips;
        get_ur_w(ur_w, ur_w_tail, ur_w_trips);
        ur_w = nstl::max(ur_w, ur_w_tail);
        ic_block_step_stack_size = jcp.uses_permw_transposition
                ? permw_stack_size(ur_w)
                : interleave_stack_size(ur_w, jcp.ic_block_step);
    } else
        ic_block_step_stack_size = extern_ic_block_step_stack_size;

    permw_buffer_start = 0;
    kd_count_offset = ic_block_step_stack_size;
    src_d_offset = ic_block_step_stack_size + 8;
    ddst_d_offset = ic_block_step_stack_size + 16;
    d_index_offset = ic_block_step_stack_size + 24;
    trans_tmp_offset = ic_block_step_stack_size + 32;
    ih_dilate_shift = ic_block_step_stack_size + 40;
    icb_loop_ker_ptr = ic_block_step_stack_size + 48;
    icb_loop_src_ptr = ic_block_step_stack_size + 56;
    stack_space_needed = ic_block_step_stack_size + 64;
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::generate() {
    preamble();

    setup_stack_space();

    sub(rsp, stack_space_needed);

    compute_loop();

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.uses_permw_transposition) {
        align(64);
        L(dst_prm_table);
        const uint16_t dst_prm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                29, 14, 30, 15, 31};

        for (size_t i = 0; i < 32; ++i)
            dw(dst_prm_array[i]);
    }
}

status_t jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md, int nthreads) {
    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    bool ok = true
            // general condition to simplify dilations
            && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
            && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
            && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
            // special condition to simplify dilations in compute_oh_loop_common
            && IMPLICATION(jcp.dilate_h != 0, ext_kh <= jcp.ih);
    if (!ok) return status::unimplemented;

    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    /* XXX: no support for padding when dilation_d > 0 */
    if (!IMPLICATION(jcp.dilate_d > 0, everyone_is(0, jcp.back_pad, jcp.f_pad)))
        return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    jcp.simd_w = simd_w;
    jcp.oc_block = simd_w;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_ncx);
    auto curr_dst_tag
            = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    jcp.is_1stconv = is_1stconv(jcp);

    bool ok_to_pad_channels
            = (jcp.ngroups == 1) && !jcp.is_1stconv && !is_data_layout_nxc;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    auto src_tag = is_data_layout_nxc
            ? dat_tag_nxc
            : (jcp.is_1stconv ? dat_tag_ncx : dat_tag_nCx16c);
    auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
    auto wei_tag = jcp.is_1stconv
            ? pick(2 * ndims - 6 + with_groups, Owi16o, gOwi16o, Ohwi16o,
                    gOhwi16o, Odhwi16o, gOdhwi16o)
            : pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
                    OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);

    if (src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
    } else if (curr_src_tag != src_tag)
        return status::unimplemented;
    jcp.src_tag = src_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dst_tag));
    } else if (curr_dst_tag != dst_tag)
        return status::unimplemented;
    jcp.dst_tag = dst_tag;

    if (diff_weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;
    }

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, x));
    }
    jcp.bia_dt = jcp.with_bias ? diff_bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.nb_oc = utils::div_up(jcp.oc, jcp.oc_block);

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad_h = ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < ext_kw && jcp.r_pad < ext_kw
            && jcp.t_pad <= max_pad_h && jcp.b_pad <= max_pad_h
            && jcp.f_pad < ext_kd && jcp.back_pad < ext_kd
            && IMPLICATION(jcp.is_1stconv && jcp.ow > max_ur_w,
                    jcp.l_pad < max_ur_w && ext_kw <= jcp.ow);
    if (!boundaries_ok) return status::unimplemented;

    const int max_kw = jcp.is_1stconv ? 24 : 14;
    /* yet another common check */
    if (jcp.kw > max_kw) return status::unimplemented;

    jcp.wei_dt = diff_weights_d.data_type();

    jcp.ic_block = jcp.is_1stconv ? jcp.ic : simd_w;
    if (ok_to_pad_channels) jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    jcp.nb_ic = utils::div_up(jcp.ic, jcp.ic_block);
    ok = true && one_of(ndims, 3, 4, 5)
            && everyone_is(
                    data_type::bf16, src_d.data_type(), diff_dst_d.data_type())
            && one_of(diff_weights_d.data_type(), data_type::f32,
                    data_type::bf16);
    if (!ok) return status::unimplemented;

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.ic_block : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.oc_block : 0;

    if (jcp.is_1stconv) {
        jcp.ic_block_step = 24 / jcp.kw;
        while (jcp.ic_block % jcp.ic_block_step != 0)
            jcp.ic_block_step--;
    } else {
        jcp.ic_block_step
                = jcp.kw <= 3 ? 8 : (jcp.kw < 7 ? 4 : (jcp.kw <= 12 ? 2 : 1));
    }

    // jcp.uses_permw_transposition = false shows better performance for
    // resnet50 v1.5 problems
    // jcp.uses_permw_transposition = true works better for 3d 1x1x1 problems
    const bool is_permw_applicable
            = !jcp.is_1stconv && jcp.stride_w == 1 && jcp.dilate_w == 0;
    const bool apply_permw_blocked = !is_data_layout_nxc && ndims == 5
            && jcp.kw == 1 && jcp.ic_block_step > 4;
    // Threshold is based on performance measurements
    const bool apply_permw_nxc = is_data_layout_nxc && ndims == 3
            && nstl::max(jcp.ic, jcp.oc) <= 32;
    jcp.uses_permw_transposition
            = is_permw_applicable && (apply_permw_blocked || apply_permw_nxc);

    jcp.kernel_kind = embd_bcast;
    if (jcp.uses_permw_transposition && jcp.kw <= 3)
        jcp.kernel_kind = expl_bcast;
    if (jcp.uses_permw_transposition && jcp.kernel_kind == expl_bcast)
        jcp.ic_block_step = jcp.kw <= 3 ? 4 : (jcp.kw < 7 ? 2 : 1);

    if (jcp.uses_permw_transposition) {
        jcp.transpose_src = false;
        jcp.transpose_dst = false;
    } else if (jcp.is_1stconv && IMPLICATION(is_data_layout_nxc, jcp.ic == 1)) {
        jcp.transpose_src = false;
        jcp.transpose_dst = true;
    } else {
        jcp.transpose_src = true;
        jcp.transpose_dst = true;
    }

    const bool is_2d = (ndims == 4);
    const bool is_3d = (ndims == 5);
    jcp.typesize_in = sizeof(bfloat16_t);
    jcp.typesize_out = sizeof(float);
    const dim_t cache_l2
            = platform::get_per_core_cache_size(2) / jcp.typesize_out;

    // Observation: Given large 3D shapes with large filter size, 1st nspc
    // bwd_w convolution benefits from non-temporal stores in diff_dst
    // transformation but not so much from blocking w.r.t. depth dimension
    // In particular, it's optimized for i3D 1st convolution
    const bool nt_stores_ok = is_data_layout_nxc
            && dim_t(jcp.oc) * jcp.od * jcp.oh * jcp.ow >= 2 * cache_l2
            && jcp.kd >= 6 && jcp.kh >= 6 && jcp.kw >= 6;

    // Performancewise transposition of diff_dst tensor is one of the major
    // bottleneck in 1st convolution. Thus for large diff_dst size we can
    // potentially further split up transposition in smaller chunks to achieve
    // better cache reuse
    const bool large_diff_dst_size
            = dim_t(jcp.oc) * jcp.od * jcp.oh * jcp.ow >= cache_l2;

    // For two dimensional diff_dst tensor blocking along height demands
    // non-trivial work along width dimension. Similarly, for three dimensional
    // diff_dst tensor enough work must be present in the joint width-height
    // dimension. Finally, there is no blocking along the width dimension
    const bool blocking_ok = large_diff_dst_size
            && IMPLICATION(is_2d, jcp.ow >= 124 && jcp.oh > 1)
            && IMPLICATION(is_3d, jcp.ow * jcp.oh >= 64 * 124 && jcp.od > 1)
            && (is_2d || is_3d);

    // TODO: Find more shapes (especially 3D with large spatials) for which
    // local transposition will be beneficial. Furthermore, for TBB threads
    // more shapes can potentially benefit from spatial blocking
    bool use_spatial_blocking = jcp.is_1stconv && !nt_stores_ok && blocking_ok;
    int optimal_blk_size = is_3d ? jcp.od : is_2d ? jcp.oh : jcp.ow;
    if (use_spatial_blocking) {
        // Default value, works best most of the times
        // TODO: For 3D shapes with intermediate sizes especially the ones not
        // belonging to the 1st convolution, we potentially have more scope
        // for optimization
        optimal_blk_size = 1;

        // Diff_weights computation can be roughly broken down into
        // the following three steps
        // = [Src transform*] + [Diff_dst transform] + [Weights computation]
        //
        // where the bottleneck lies with diff_dst transform that spatial
        // blocking tries to mitigate by avoiding cache thrashing.
        // *note: Src transform may not always be needed.
        //
        // In an idealistic scenario, optimal_blk_size will be an explicit
        // function of the following form
        // optimal_blk_size = f(od, oh, ow, oc)
        //
        // though owing to lack of data points w.r.t. 1st convolution shapes it
        // is approximated by one with few exceptional cases [found by manual
        // optimization] as written below

        if (is_2d && utils::one_of(jcp.oh, 149, 300, 224, 512, 608)) {
            switch (jcp.oh) {
                case 149: optimal_blk_size = 10; break;
                case 224: optimal_blk_size = 56; break;
                case 300: optimal_blk_size = 30; break;
                case 512: optimal_blk_size = 8; break;
                case 608: optimal_blk_size = 10; break;
            }
        }
    }

    jcp.global_transpose = dnnl_thr_syncable() && !use_spatial_blocking;
    jcp.use_nt_stores_ddst = jcp.global_transpose && nt_stores_ok;
    jcp.spatial_blk_size = optimal_blk_size;

    const bool padding_ok = IMPLICATION(!jcp.transpose_src,
            jcp.l_pad < max_ur_w && jcp.r_pad < max_ur_w
                    && ext_kw <= jcp.iw + 1);
    if (!padding_ok) return status::unimplemented;

    const int tr_round = 2;
    // Logic for tr_pad calculation: transpose is used in the extern kernel.
    // There is a memory usage optimization where physical padding is shared
    // between transpose buffers. In calculating on a row, data is read from the
    // src 2 elements at a time due to the bf16 broadcast. Calculation starts
    // at the beginning of the left padding and ends at the end of the right
    // padding. Because elements are read two at a time, we may need r_pad + 1
    // padding on the right. As such, the shared padding is the max of l_pad and
    // r_pad + 1, rounded as necessary for the transpose data format.
    int tr_pad = rnd_up(nstl::max(jcp.l_pad, jcp.r_pad + 1), tr_round);
    jcp.tr_iw = jcp.transpose_src
            ? rnd_up(div_up(jcp.iw, jcp.stride_w) + tr_pad, tr_round)
                    * jcp.stride_w
            : jcp.iw;

    jcp.tr_src_num_guard_elems = tr_pad; // upper bound
    jcp.tr_ow = jcp.transpose_dst ? rnd_up(jcp.ow, 2) : jcp.ow;

    bool args_ok = true
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.ic % jcp.ic_block == 0 && jcp.oc % jcp.oc_block == 0)
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    int inp_row_size = jcp.ic_block * jcp.tr_iw * jcp.typesize_in;
    int out_row_size = jcp.oc_block * jcp.tr_ow * jcp.typesize_in;
    int full_spat_min_h_block_size
            = nstl::max(1, nstl::max(jcp.b_pad, jcp.t_pad));
    int full_spat_working_set_size
            = (inp_row_size + out_row_size) * full_spat_min_h_block_size;
    bool use_full_spat_loop = isa_has_bf16(jcp.isa) && jcp.ndims < 5
            && jcp.ih == jcp.oh && jcp.iw == jcp.ow
            && !one_of(1, jcp.kh, jcp.kw)
            && everyone_is(1, jcp.stride_h, jcp.stride_w)
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            && jcp.l_pad == jcp.kw / 2 && jcp.t_pad == jcp.kh / 2
            && !jcp.uses_permw_transposition && !jcp.is_1stconv
            && full_spat_working_set_size <= full_spat_opt_working_set_size
            && jcp.ic >= 128;

    jcp.harness = ndims == 5
            ? harness_3d_reduction
            : (use_full_spat_loop ? harness_compute_full_spatial
                                  : (ndims == 4) ? harness_2d_reduction
                                                 : harness_mb_reduction);

    switch (jcp.harness) {
        case harness_2d_reduction: jcp.nthr_mb_work = jcp.mb * jcp.oh; break;
        case harness_3d_reduction: jcp.nthr_mb_work = jcp.mb * jcp.od; break;
        case harness_compute_full_spatial:
        case harness_mb_reduction: jcp.nthr_mb_work = jcp.mb; break;
        default: assert(!"Invalid harness"); jcp.nthr_mb_work = jcp.mb;
    }
    { // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;

        // TODO: Optimize memory allocation when threaded on height and depth
        if (jcp.transpose_src) {
            jcp.tr_src_buf_size = jcp.tr_iw * jcp.ic_block * jcp.ih * jcp.id;
            jcp.tr_src_buf_count = jcp.global_transpose
                    ? jcp.nthr_mb * jcp.nb_ic * jcp.ngroups
                    : jcp.nthr;
        }
        if (jcp.transpose_dst) {
            jcp.tr_diff_dst_buf_size
                    = jcp.tr_ow * jcp.oc_block * jcp.oh * jcp.od;
            jcp.tr_diff_dst_buf_count = jcp.global_transpose
                    ? jcp.nthr_mb * jcp.nb_oc * jcp.ngroups
                    : jcp.nthr;
        }
    }

    jcp.nb_ic_blocking_max = 1;
    if (is_data_layout_nxc && jcp.uses_permw_transposition
            && (jcp.ow > max_ur_w || jcp.ndims == 5))
        jcp.nb_ic_blocking_max = nstl::min(8, div_up(jcp.nb_ic, jcp.nthr_ic_b));
    return status::success;
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {

    if (!jcp.uses_permw_transposition) {
        // XXX: See the comment about tr_iw and guarding elements in
        // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
        const size_t tr_src_size = jcp.tr_src_buf_count * jcp.tr_src_buf_size
                + jcp.tr_src_num_guard_elems;
        scratchpad.book(key_conv_tr_src, tr_src_size, jcp.typesize_in);

        /* prepare synchronization contexts */
        if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            scratchpad.book<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx, tr_src_bctx_size);
        }

        const size_t tr_diff_dst_size
                = jcp.tr_diff_dst_buf_count * jcp.tr_diff_dst_buf_size;

        const size_t min_align = jcp.use_nt_stores_ddst ? 64 : jcp.typesize_in;
        scratchpad.book(key_conv_tr_diff_dst, tr_diff_dst_size, jcp.typesize_in,
                min_align);

        /* prepare synchronization contexts */
        if (jcp.global_transpose && jcp.nthr_ic_b > 1) {
            const size_t tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
            scratchpad.book<simple_barrier::ctx_t>(
                    key_conv_tr_diff_dst_bctx, tr_diff_dst_bctx_size);
        }
    }

    if (IMPLICATION(jcp.nthr_mb == 1,
                (jcp.with_bias && jcp.bia_dt == data_type::bf16)
                        || jcp.wei_dt == data_type::bf16)) {
        const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size
                = jcp.with_bias * jcp.ngroups * jcp.nb_oc * jcp.oc_block;

        const int num_wei_buffers
                = jcp.wei_dt == data_type::bf16 ? jcp.nthr_mb : jcp.nthr_mb - 1;
        const int num_bia_buffers = jcp.with_bias
                ? (jcp.bia_dt == data_type::bf16 ? jcp.nthr_mb
                                                 : jcp.nthr_mb - 1)
                : 0;

        const size_t wei_bia_reduction_size
                = wei_size * num_wei_buffers + bia_size * num_bia_buffers;

        scratchpad.book<float>(
                key_conv_wei_bia_reduction, wei_bia_reduction_size);

        if (jcp.global_transpose)
            scratchpad.book<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx, 1);
    }

    if (jcp.with_bias) {
        if ((jcp.oc_without_padding % jcp.oc_block != 0)
                && jcp.bia_dt == data_type::f32)
            scratchpad.book(key_conv_padded_bias,
                    jcp.ngroups * jcp.nb_oc * jcp.oc_block, jcp.typesize_bia);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_) {
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = dnnl_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        nthr_ = nthr_g_ = max_threads;
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) if weights tensor size is less than source and destination
         *       tensors we apply the ratio of the source and destination
         *       tensor sizes to weights one as compensation coefficient to
         *       avoid parallelization across batch size only, othervise we
         *       apply additional coefficient to source component based on
         *       performance measurements
         *  (n2) use scales based on output vs input channels ratio for source
         *       and destination componets to imporve threading balance across
         *       input and output channels */

        const dim_t src_type_size = 2;
        const dim_t wei_type_size = 4;

        dim_t src_size
                = (dim_t)j.mb * j.ic * j.id * j.ih * j.tr_iw * src_type_size;
        dim_t dst_size
                = (dim_t)j.mb * j.oc * j.od * j.oh * j.tr_ow * src_type_size;
        dim_t wei_size
                = (dim_t)j.oc * j.ic * j.kd * j.kh * j.kw * wei_type_size;

        float wei_compensation_scale = 0.5f * (dst_size + src_size) / wei_size;
        float oi_channels_ratio = (float)j.nb_oc / j.nb_ic;
        auto get_src_coef = [=]() {
            float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
            if (wei_compensation_scale < 1.0f) src_coef *= 4.0f;

            return src_coef;
        };

        auto get_dst_coef
                = [=]() { return nstl::max(oi_channels_ratio, 1.0f); };

        auto get_wei_coef
                = [=]() { return nstl::max(wei_compensation_scale, 1.0f); };

        const float src_coef = get_src_coef();
        const float dst_coef = get_dst_coef();
        const float wei_coef = get_wei_coef();

        float src_v = src_coef * div_up(j.nthr_mb_work, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_ic, nthr_ic_b) * j.mb
                * j.ic_block * j.id * j.ih * j.tr_iw / j.nthr_mb_work
                / j.stride_d / j.stride_h / j.stride_w;
        float wei_v = wei_coef * div_up(j.ngroups, nthr_g_)
                * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b) * j.kh
                * j.kw * j.kd * j.ic_block * j.oc_block;
        float dst_v = dst_coef * div_up(j.nthr_mb_work, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b) * j.mb
                * j.oc_block * j.od * j.oh * j.tr_ow / j.nthr_mb_work;

        return src_v + dst_v + wei_v;
    };

    float best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.nthr_mb_work);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            float mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > nthr / 2 && nthr_mb_ < nthr)
        nthr_mb_ = nstl::min(j.nthr_mb_work, nthr);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
}

template struct _jit_avx512_core_bf16_fwd_kernel<Xbyak::Zmm>;
template struct _jit_avx512_core_bf16_fwd_kernel<Xbyak::Ymm>;
template struct _jit_avx512_core_bf16_fwd_kernel<Xbyak::Xmm>;
template struct _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Zmm>;
template struct _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Ymm>;
template struct _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Xmm>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
