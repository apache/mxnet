/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_sse41_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

using namespace Xbyak;

void jit_sse41_conv_fwd_kernel_f32::oh_step_unroll_kw(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
                - nstl::max(0,
                        div_up(ki * dilate_w + pad_r - (kw - 1) * dilate_w,
                                stride_w));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off = get_input_offset(
                        ifm2, filter_w_to_input(ki, jj, pad_l));
                movss(Xmm(oc_blocks * ur_w + jj + 1),
                        ptr[aux_reg_input + inp_off]);
                shufps(Xmm(oc_blocks * ur_w + jj + 1),
                        Xmm(oc_blocks * ur_w + jj + 1), 0x0);
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                for (int jj = jj_start; jj < jj_end; jj++) {
                    movups(xmm0,
                            ptr[aux_reg_kernel
                                    + get_kernel_offset(ii, ki, ifm2)]);
                    mulps(xmm0, Xmm(oc_blocks * ur_w + jj + 1));
                    addps(Xmm(ur_w * ii + jj + 1), xmm0);
                }
            }
        }
    }
}

void jit_sse41_conv_fwd_kernel_f32::oh_step_nopad(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    Label kw_loop;

    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;

    xor_(ki_iter, ki_iter);
    L(kw_loop);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off = get_input_offset(
                        ifm2, filter_w_to_input(0, jj, pad_l));
                movss(Xmm(oc_blocks * ur_w + jj + 1),
                        ptr[aux_reg_input + inp_off]);
                shufps(Xmm(oc_blocks * ur_w + jj + 1),
                        Xmm(oc_blocks * ur_w + jj + 1), 0x0);
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                for (int jj = jj_start; jj < jj_end; jj++) {
                    movups(xmm0,
                            ptr[aux_reg_kernel
                                    + get_kernel_offset(ii, 0, ifm2)]);
                    mulps(xmm0, Xmm(oc_blocks * ur_w + jj + 1));
                    addps(Xmm(ur_w * ii + jj + 1), xmm0);
                }
            }
        }
        add(aux_reg_kernel, get_kernel_offset(0, 1, 0));
        add(aux_reg_input, get_input_offset(0, filter_w_to_input(1)));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_loop, T_NEAR);
    }
}

void jit_sse41_conv_fwd_kernel_f32::width_blk_step(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int oc_blk = jcp.oc_block;

    xor_(simd_iter, simd_iter);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    Label init_simd_iter_loop;
    Label init_done;
    Label init_first;

    L(init_simd_iter_loop);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++)
        for (int jj = 0; jj < ur_w; jj++)
            movups(Xmm(ur_w * ii + jj + 1),
                    xword[reg_output + get_output_offset(ii, jj)]);

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done, T_NEAR);

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                addps(Xmm(ur_w * ii + jj + 1),
                        xword[reg_bias + sizeof(float) * ii * oc_blk]);
    }

    jmp(init_done);

    L(init_first);
    if (this->jcp.with_bias) {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                movups(Xmm(ur_w * ii + jj + 1),
                        xword[reg_bias + sizeof(float) * ii * oc_blk]);
    } else {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                pxor(Xmm(ur_w * ii + jj + 1), Xmm(ur_w * ii + jj + 1));
    }

    L(init_done);

    Label skip_kh_loop;
    mov(kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    Label kh_loop;
    L(kh_loop);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, oc_blocks);
            sub(aux_reg_input, get_input_offset(0, filter_w_to_input(kw)));
            add(aux_reg_input, get_input_offset(0, filter_h_to_input(1)));
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_kernel, get_kernel_offset(0, kw, 0));
            add(aux_reg_input, get_input_offset(0, filter_h_to_input(1)));
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.with_eltwise) {
        Label regular_store;
        test(reg_ci_flag, FLAG_IC_LAST);
        je(regular_store, T_NEAR);

        eltwise_injector_->compute_vector_range(1, oc_blocks * ur_w + 1);

        L(regular_store);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            Xmm reg_out = Xmm(ur_w * ii + jj + 1);
            movups(xword[reg_output + get_output_offset(ii, jj)], reg_out);
        }
    }

    mov(aux_reg_kernel, reg_kernel);
    mov(aux_reg_input, reg_input);
    add(aux_reg_kernel, sizeof(float) * 4);
    add(reg_output, sizeof(float) * 4);
    add(reg_bias, sizeof(float) * 4);
    inc(simd_iter);
    cmp(simd_iter, 2);
    jl(init_simd_iter_loop, T_NEAR);

    sub(reg_output, sizeof(float) * 8);
    sub(reg_bias, sizeof(float) * 8);
}

inline void jit_sse41_conv_fwd_kernel_f32::solve_common(int oc_blocks) {
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int str_w = jcp.stride_w;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, jcp.r_pad);
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, str_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks); // "lpad"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w, l_pad)));
        add(reg_output, get_output_offset(0, ur_w));
    }

    Label ow_loop;
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop);

        width_blk_step(ur_w, 0, 0, oc_blocks); // "middle"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w)));
        add(reg_output, get_output_offset(0, ur_w));

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks); // "rpad"
        add(reg_input, get_input_offset(0, filter_w_to_input(0, ur_w)));
        add(reg_output, get_output_offset(0, ur_w));
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks); // "tail"
}

void jit_sse41_conv_fwd_kernel_f32::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias) mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;
    Label tail, exit;

    cmp(reg_oc_blocks, jcp.nb_oc_blocking);
    jne(nb_oc_tail ? tail : exit, T_NEAR);

    solve_common(jcp.nb_oc_blocking);
    jmp(exit, T_NEAR);

    if (nb_oc_tail) {
        L(tail);
        cmp(reg_oc_blocks, nb_oc_tail);
        jne(exit, T_NEAR);
        solve_common(nb_oc_tail);
    }

    L(exit);

    this->postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_sse41_conv_fwd_kernel_f32::post_ops_ok(
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

status_t jit_sse41_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads) {
    if (!mayiuse(sse41)) return status::unimplemented;

    jcp.nthr = nthreads;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[0];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad
            || ext_kh <= jcp.b_pad;
    if (kernel_outside_src) return status::unimplemented;

    const auto dat_tag_nxc = (ndims == 3 ? nwc : nhwc);
    const auto dat_tag_ncx = (ndims == 3 ? ncw : nchw);
    const auto dat_tag_nCx8c = (ndims == 3 ? nCw8c : nChw8c);
    const auto wei_tag_OIxio = with_groups
            ? pick(ndims - 3, gOIw8i8o, gOIhw8i8o)
            : pick(ndims - 3, OIw8i8o, OIhw8i8o);
    const auto wei_tag_Oxio = with_groups ? pick(ndims - 3, gOwi8o, gOhwi8o)
                                          : pick(ndims - 3, Owi8o, Ohwi8o);

    jcp.src_tag
            = src_d.matches_one_of_tag(dat_tag_ncx, dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag_OIxio, wei_tag_Oxio);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);

    const bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    bool args_ok = true
            && IMPLICATION(flat,
                    jcp.wei_tag == wei_tag_Oxio
                            && ((jcp.src_tag == dat_tag_ncx
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo,
                    jcp.wei_tag == wei_tag_OIxio
                            && ((jcp.src_tag == dat_tag_nCx8c
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1];
    if (!args_ok) return status::unimplemented;

    const int simd_w = 8; // 2 SSE vectors processing at once

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.nb_oc_blocking
            = is_data_layout_nxc ? 1 : 4; /* the optimal value for the kernel */

    args_ok = true && jcp.oc % simd_w == 0 && jcp.l_pad <= jcp.ur_w
            && IMPLICATION(jcp.kw > 7,
                    (jcp.t_pad == 0 && jcp.l_pad == 0)
                            || (jcp.stride_w == 1 && jcp.stride_h == 1))
            && IMPLICATION(mimo, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));

    // kernel needs 1 temporary YMM register
    const int num_avail_regs = 15;
    if (r_pad_no_tail > jcp.ur_w * jcp.stride_w && jcp.ow / jcp.ur_w > 1) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = nstl::min(r_pad_no_tail / jcp.stride_w + jcp.ur_w_tail,
                nstl::min(jcp.ow, num_avail_regs / 2));
        jcp.nb_oc_blocking = (num_avail_regs - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0,
                calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                        jcp.stride_w, ext_kw));

        if (jcp.ur_w < nstl::max(jcp.l_pad, r_pad_no_tail))
            return status::unimplemented;
    }
    assert(jcp.nb_oc_blocking > 0);
    assert(jcp.ur_w * (jcp.nb_oc_blocking + 1) <= num_avail_regs);

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.nb_ic_blocking = 12;
        jcp.nb_ic_blocking_max = 16;
    } else {
        jcp.nb_ic_blocking = 1;
        jcp.nb_ic_blocking_max = jcp.nb_ic_blocking;
    }

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
