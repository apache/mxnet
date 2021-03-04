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
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp) {
    jcp.loop_order = loop_cwgn;
    if (jcp.ngroups > 1) {
        jcp.loop_order = loop_ngcw;
        if (jcp.mb < jcp.nthr)
            jcp.loop_order = jcp.ndims == 3 ? loop_nwcg : loop_nhwcg;
    } else if (jcp.mb >= jcp.nthr && jcp.ic_without_padding <= 8) {
        jcp.loop_order = loop_ngcw;
    }
}
} // namespace

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::prepare_output(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    for (int k = 0; k < nb_oc_block; ++k)
        for (int j = 0; j < ur_w; ++j) {
            Vmm vmm = vmm_out(j, k);
            uni_vpxor(vmm, vmm, vmm);
        }
    if (jcp.signed_input) {
        auto xmm_shift = Xbyak::Xmm(vmm_shift.getIdx());
        if (jcp.is_depthwise)
            mov(reg_scratch, 128);
        else
            mov(reg_scratch, 0x80808080);
        uni_vmovq(xmm_shift, reg_scratch);
        uni_vpbroadcastd(vmm_shift, xmm_shift);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::cvt2ps(data_type_t type_in,
        const Vmm &vmm_in, const Reg64 &reg, int offset, int load_size) {

    load_data(type_in, vmm_in, reg, offset, load_size);
    if (type_in != data_type::f32) uni_vcvtdq2ps(vmm_in, vmm_in);
}
template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::compute_eltwise(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    // avoid passing xmm0 to eltwise injector for sse41
    eltwise_injector_->compute_vector_range(16 - nb_oc_block * ur_w, 16);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::store_output(
        int ur_w, bool last_oc_block_flag) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    int oc_block = jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    if (jcp.signed_input)
        mov(reg_compensation, ptr[param1 + GET_OFF(compensation)]);

    if (jcp.src_zero_point) {
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        uni_vpbroadcastd(vmm_zp, ptr[reg_src_zero_point]);
    }

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }
    if (jcp.signed_input) {
        /* put 'wei_adj_scale = 0.5' for bias calculation */
        mov(reg_bias_alpha, float2int(jcp.wei_adj_scale));
        uni_vmovq(xmm_bias_alpha(), reg_bias_alpha);
        uni_vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
    }

    for (int k = 0; k < nb_oc_block; ++k) {
        const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
        const int load_size = mask_flag ? get_tail_size() : get_blocking_size();
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * oc_block);
        if (jcp.with_bias) {

            int bias_offset = jcp.typesize_bia * k * oc_block;
            cvt2ps(jcp.bia_dt, vmm_bias, reg_bias, bias_offset, load_size);
            if (jcp.signed_input) /* bias *= 0.5 */
                uni_vmulps(vmm_bias, vmm_bias, vmm_bias_alpha());
        }
        if (jcp.signed_input) {
            const int comp_offset = sizeof(int32_t) * k * oc_block;
            cvt2ps(data_type::s32, vmm_comp, reg_compensation, comp_offset,
                    load_size);
        }
        if (jcp.src_zero_point) {
            const int zp_offset = sizeof(int32_t) * k * oc_block;
            load_data(data_type::s32, vmm_zp_comp, reg_zp_compensation,
                    zp_offset, load_size);
            uni_vpmulld(vmm_zp_comp, vmm_zp_comp, vmm_zp);
            uni_vcvtdq2ps(vmm_zp_comp, vmm_zp_comp);
        }
        /* add to ymm_accum: compensation, zero_point, bias and permute */
        if (mask_flag) {
            uni_vpxor(vmm_scale, vmm_scale, vmm_scale);
            load_data(data_type::s32, vmm_scale, reg_ptr_scales, scale_offset,
                    get_tail_size());
        } else {
            uni_vmovups(vmm_scale, ptr[reg_ptr_scales + scale_offset]);
        }

        for (int j = 0; j < ur_w; ++j) {
            const Vmm vmm = vmm_out(j, k);

            uni_vcvtdq2ps(vmm, vmm);
            if (jcp.signed_input) uni_vaddps(vmm, vmm, vmm_comp);
            if (jcp.src_zero_point) uni_vaddps(vmm, vmm, vmm_zp_comp);
            if (jcp.with_bias) uni_vaddps(vmm, vmm, vmm_bias);

            uni_vmulps(vmm, vmm, vmm_scale);
        }
    }

    /* Do post-ops */
    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
    if (maybe_eltwise(0)) compute_eltwise(ur_w);
    if (p_sum_scale) { // post_op: sum
        for (int k = 0; k < nb_oc_block; ++k) {
            const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
            for (int j = 0; j < ur_w; ++j) {
                int aux_output_offset = jcp.typesize_out
                        * (k * oc_block
                                + j * jcp.oc_without_padding * jcp.ngroups);
                cvt2ps(jcp.dst_dt, vmm_prev_dst, reg_out, aux_output_offset,
                        mask_flag ? get_tail_size() : get_blocking_size());
                Vmm vmm = vmm_out(j, k);
                if (*p_sum_scale == 1.f)
                    uni_vaddps(vmm, vmm, vmm_prev_dst);
                else {
                    uni_vbroadcastss(vmm_tmp, ptr[reg_ptr_sum_scale]);
                    uni_vfmadd231ps(vmm, vmm_prev_dst, vmm_tmp);
                }
            }
        }
    }
    if (maybe_eltwise(1)) compute_eltwise(ur_w);

    if (jcp.dst_zero_point) {
        mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
        uni_vpbroadcastd(vmm_zp, ptr[reg_dst_zero_point]);
        uni_vcvtdq2ps(vmm_zp, vmm_zp);

        /* Add dst zero_point to accumulator */
        for (int k = 0; k < nb_oc_block; k++) {
            for (int j = 0; j < ur_w; j++) {
                const Vmm vmm = vmm_out(j, k);
                uni_vaddps(vmm, vmm, vmm_zp);
            }
        }
    }

    // Properly saturate the accumulators for integer datatypes

    // No need to saturate on lower bound for signed integer types, as
    // the conversion to int would return INT_MIN, and then proper
    // saturation will happen in store_data
    if (jcp.dst_dt == data_type::u8) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        for (int k = 0; k < nb_oc_block; ++k) {
            for (int j = 0; j < ur_w; ++j) {
                Vmm vmm = vmm_out(j, k);
                uni_vmaxps(vmm, vmm, vmm_zero);
            }
        }
    }
    if (utils::one_of(
                jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        float saturation_ubound = types::max_value<float>(jcp.dst_dt);
        Xmm xmm_saturation(vmm_saturation.getIdx());
        mov(reg_ptr_saturation_ubound, float2int(saturation_ubound));
        uni_vmovq(xmm_saturation, reg_ptr_saturation_ubound);
        uni_vbroadcastss(vmm_saturation, xmm_saturation);

        for (int k = 0; k < nb_oc_block; ++k) {
            for (int j = 0; j < ur_w; ++j) {
                Vmm vmm = vmm_out(j, k);
                uni_vminps(vmm, vmm, vmm_saturation);
            }
        }
    }

    // Convert float accumulators to int datatype if needed
    if (utils::one_of(jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32))
        for (int k = 0; k < nb_oc_block; ++k)
            for (int j = 0; j < ur_w; ++j) {
                Vmm vmm = vmm_out(j, k);
                uni_vcvtps2dq(vmm, vmm);
            }

    /* write out register to output_addr */
    for (int k = 0; k < nb_oc_block; ++k) {
        const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
        for (int j = 0; j < ur_w; ++j) {
            Vmm r_vmm = vmm_out(j, k);
            int aux_output_offset = jcp.typesize_out
                    * (k * oc_block + j * jcp.oc_without_padding * jcp.ngroups);
            store_data(jcp.dst_dt, r_vmm, reg_out, aux_output_offset,
                    mask_flag ? get_tail_size() : get_blocking_size());
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::compute_ker_dw(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {

    if (!(utils::one_of(isa, avx2) && std::is_same<Vmm, Xbyak::Ymm>::value)
            && !(utils::one_of(isa, sse41)
                    && std::is_same<Vmm, Xbyak::Xmm>::value))
        assert(!"invalid group blocking for depthwise convolution");

    const bool compute_kernel = IMPLICATION(h_padded, jcp.signed_input);

    if (jcp.src_zero_point) {
        push(aux_reg_ker_d);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        uni_vpbroadcastd(vmm_zp, ptr[reg_src_zero_point]);
    }

    auto input_spatial_index = [=](int oi, int ki) {
        return (ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l);
    };

    auto input_offset2 = [=](int ii, int ci) {
        if (jcp.is_fused_conv)
            return jcp.typesize_in
                    * (ii * jcp.dw_conv_buffer_oc + ci * jcp.ch_block);
        else
            return jcp.typesize_in * (ii * jcp.ngroups + ci * jcp.ch_block);
    };

    auto input_offset3 = [=](int oi, int ci, int ki) {
        return jcp.typesize_in * input_offset2(input_spatial_index(oi, ki), ci);
    };

    auto kernel_offset = [=](int ci, int ki) {
        return jcp.typesize_in * ((ci * jcp.kh * jcp.kw + ki) * jcp.ch_block);
    };

    auto compute = [=](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        // okay for depthwise since src is zero-extended
        uni_vpmaddwd(vmm_dw_tmp, vreg_src, vreg_wei);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_dw_tmp);
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise && !h_padded) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ++ki) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; ++oi) {
                int ii = input_spatial_index(oi, ki);
                if (first || ii < ii_start) ii_start = ii;
                if (first || ii > ii_end) ii_end = ii;
                first = false;
            }
        }
    }

    if (jcp.signed_input) uni_vmovups(vmm_dw_shifted_zero, vmm_shift);

    for (int ci = 0; ci < jcp.nb_ch_blocking; ++ci) {
        const bool mask_flag = last_ic_block_flag != no_last_block
                && ci == jcp.nb_ch_blocking - 1;
        if (jcp.is_resrc_depthwise && !h_padded) {
            // now we can load input once and reuse up to jcp.kw times
            for (int ii = ii_start; ii <= ii_end; ++ii) {
                int aux_input_offset = input_offset2(ii, ci);
                const Vmm vmm_inp_tmp = vmm_inp(ii, jcp.nb_ch_blocking);

                uni_vpxor(vmm_inp_tmp, vmm_inp_tmp, vmm_inp_tmp);
                load_data(data_type::u8, vmm_inp_tmp, aux_reg_inp,
                        aux_input_offset,
                        mask_flag ? get_tail_size() : get_blocking_size());
                if (jcp.signed_input)
                    uni_vpaddb(vmm_inp_tmp, vmm_inp_tmp, vmm_shift);
            }
        }
        for (int ki = 0; ki < jcp.kw; ++ki) {
            int aux_kernel_offset = kernel_offset(ci, ki);
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);

            if (compute_kernel) {
                uni_vpmovsxbd(vmm_dw_wei, ptr[aux_reg_ker + aux_kernel_offset]);
                if (h_padded) {
                    assert(jcp.signed_input);
                    for (int oi = 0; oi < ur_w; ++oi)
                        compute(vmm_out(oi, ci), vmm_dw_wei,
                                vmm_dw_shifted_zero);
                } else {
                    int start = jcp.signed_input ? 0 : oi_start;
                    int end = jcp.signed_input ? ur_w : oi_end;
                    for (int oi = start; oi < end; ++oi) {
                        if (oi >= oi_start && oi < oi_end) {
                            if (jcp.is_resrc_depthwise) {
                                int ii = input_spatial_index(oi, ki);
                                vmm_dw_src = vmm_inp(ii, jcp.nb_ch_blocking);
                            } else {
                                int aux_input_offset
                                        = input_offset3(oi, ci, ki);
                                load_data(data_type::u8, vmm_dw_src,
                                        aux_reg_inp, aux_input_offset,
                                        mask_flag ? get_tail_size()
                                                  : get_blocking_size());
                                if (jcp.signed_input)
                                    uni_vpaddb(
                                            vmm_dw_src, vmm_dw_src, vmm_shift);
                            }
                            compute(vmm_out(oi, ci), vmm_dw_wei, vmm_dw_src);
                        } else {
                            assert(jcp.signed_input);
                            compute(vmm_out(oi, ci), vmm_dw_wei,
                                    vmm_dw_shifted_zero);
                        }
                    }
                }
            }
            if (jcp.src_zero_point) {
                /* calculate src_zero_point padding as:
                *      (is_padding ?
                *           src_zero_point_s32 * conv(1, wei_s32) : 0) */
                if (!compute_kernel) {
                    uni_vpmovsxbd(
                            vmm_dw_wei, ptr[aux_reg_ker + aux_kernel_offset]);
                } // else: already loaded weights from previous block
                for (int oi = 0; oi < ur_w; oi++) {
                    if (oi < oi_start || oi >= oi_end || h_padded) {
                        uni_vpmulld(vmm_zp_dw_tmp, vmm_dw_wei, vmm_zp);
                        uni_vpaddd(vmm_out(oi, ci), vmm_out(oi, ci),
                                vmm_zp_dw_tmp);
                    }
                }
            }
        }
    }

    if (jcp.src_zero_point) pop(aux_reg_ker_d);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::compute_ker(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {
    if (jcp.is_depthwise)
        return compute_ker_dw(ur_w, pad_l, pad_r, last_ic_block_flag, h_padded);

    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;

    const bool compute_kernel = IMPLICATION(h_padded, jcp.signed_input);

    assert(IMPLICATION(h_padded, jcp.src_zero_point || jcp.signed_input));

    if (jcp.src_zero_point) {
        push(aux_reg_ker_d);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
    }

    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                                * jcp.ic_without_padding * jcp.ngroups
                        + ic_sub_step * ic);
    };
    auto kernel_offset = [=](int ii, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw + ki)
                                * ch_block_all
                        + ic_sub_step * ic * oc_block);
    };
    auto compute = [=](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        uni_vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
        uni_vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
        uni_vpaddd(vreg_acc, vreg_acc, vmm_tmp);
    };

    for (int ki = 0; ki < kw; ++ki) {
        const int ow_start = get_ow_start(ki, pad_l);
        const int ow_end = get_ow_end(ur_w, ki, pad_r);
        const int ic_tail_size = jcp.ic_without_padding % ic_sub_step;

        const int _start = jcp.signed_input ? 0 : ow_start;
        const int _end = jcp.signed_input ? ur_w : ow_end;

        /* Skip the last loads of input
            if (ic % 8) / ic_sub_step < ic_block / ic_sub_step */
        const int icb = (last_ic_block_flag != no_last_block)
                ? div_up((jcp.ic_without_padding % ic_block), ic_sub_step)
                : ic_block / ic_sub_step;

        if (compute_kernel) {
            for (int ic = 0; ic < icb; ++ic) {
                if (h_padded) {
                    /* fill padded area with shifted values */
                    const Vmm inp = vmm_inp(0, nb_oc_block);
                    uni_vmovups(inp, vmm_shift);
                } else {
                    for (int jj = _start; jj < _end; ++jj) {
                        int aux_input_offset = input_offset(jj, ic, ki);
                        if (jj >= ow_start && jj < ow_end) {
                            const bool need_partial_ic_bcast = true
                                    && last_ic_block_flag == last_sp_block
                                    && ic_tail_size != 0 && ic == icb - 1;

                            if (need_partial_ic_bcast) {
                                const auto inp_bcastd_vmm
                                        = vmm_inp(jj, nb_oc_block);
                                const auto inp_bcastd
                                        = Xmm(inp_bcastd_vmm.getIdx());
                                load_bytes(inp_bcastd_vmm, aux_reg_inp,
                                        aux_input_offset, ic_tail_size,
                                        isa == sse41);
                                uni_vpbroadcastd(
                                        vmm_inp(jj, nb_oc_block), inp_bcastd);
                            } else {
                                uni_vpbroadcastd(vmm_inp(jj, nb_oc_block),
                                        ptr[aux_reg_inp + aux_input_offset]);
                            }
                            if (jcp.signed_input)
                                uni_vpaddb(vmm_inp(jj, nb_oc_block),
                                        vmm_inp(jj, nb_oc_block), vmm_shift);
                        } else {
                            /* fill padded area with shifted values */
                            if (jcp.signed_input) {
                                const Vmm inp = vmm_inp(jj, nb_oc_block);
                                uni_vmovups(inp, vmm_shift);
                            }
                        }
                    }
                }
                for (int ii = 0; ii < nb_oc_block; ++ii) {
                    const int aux_kernel_offset = kernel_offset(ii, ic, ki);
                    uni_vmovdqu(vmm_wei, ptr[aux_reg_ker + aux_kernel_offset]);
                    for (int jj = _start; jj < _end; ++jj) {
                        const Vmm inp = vmm_inp(h_padded ? 0 : jj, nb_oc_block);
                        compute(vmm_out(jj, ii), vmm_wei, inp);
                    }
                }
            }
        }
        if (jcp.src_zero_point) {
            uni_vpbroadcastd(vmm_zp, ptr[reg_src_zero_point]);
            /* calculate src_zero_point padding as:
             *      (is_padding ? src_zero_point_s32 * conv(1, wei_s8) : 0) */
            const Vmm vmm_wacc = vmm_inp(0, nb_oc_block);
            for (int jj = 0; jj < ur_w; jj++) {
                if (jj < ow_start || jj >= ow_end || h_padded) {
                    for (int ii = 0; ii < nb_oc_block; ii++) {
                        uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                        for (int ic = 0; ic < icb; ic++) {
                            const int aux_kernel_offset
                                    = kernel_offset(ii, ic, ki);
                            uni_vpmaddubsw(vmm_wacc, vmm_zp_one,
                                    ptr[aux_reg_ker + aux_kernel_offset]);
                            uni_vpmaddwd(vmm_wacc, vmm_wacc, vmm_one);
                            uni_vpaddd(vmm_tmp, vmm_tmp, vmm_wacc);
                        }
                        uni_vpmulld(vmm_tmp, vmm_tmp, vmm_zp);
                        uni_vpaddd(vmm_out(jj, ii), vmm_out(jj, ii), vmm_tmp);
                    }
                }
            }
        }
    }
    if (jcp.src_zero_point) pop(aux_reg_ker_d);
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::kh_loop(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag) {

    Label kd_label, kh_label, skip_kd_loop, skip_kh_loop;
    Label f_overflow_label, no_f_overflow_label, d_h_f_overflow_label,
            t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label, back_overflow_label, no_back_overflow_label,
            d_h_back_overflow_label;

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr
            = jcp.typesize_in * jcp.iw * jcp.ic_without_padding * jcp.ngroups;

    if (jcp.src_zero_point && !jcp.is_depthwise) {
        const auto xmm_one = Xbyak::Xmm(vmm_zp_one.getIdx());
        mov(reg_scratch, 0x01010101);
        uni_vmovq(xmm_one, reg_scratch);
        uni_vpbroadcastd(vmm_zp_one, xmm_one);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_inp_d, reg_inp);
        if (jcp.signed_input || jcp.src_zero_point) {
            //TODO: May be avoided when f_pad=0 and dd0
            //TODO: Potential optimization by precomputing, when kd <<< od?
            mov(reg_ki, ptr[param1 + GET_OFF(f_overflow)]);
            cmp(reg_ki, 0);
            je(no_f_overflow_label, T_NEAR);
            L(f_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov(reg_kj, jcp.kh);
                L(d_h_f_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    add(aux_reg_ker, shift_kernel_ptr);
                    dec(reg_kj);
                    jne(d_h_f_overflow_label);
                }
                add(aux_reg_ker_d, shift_kernel_ptr * jcp.kh);
                dec(reg_ki);
                jne(f_overflow_label);
            }
            L(no_f_overflow_label);
        }

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        if ((jcp.signed_input || jcp.src_zero_point) || (jcp.dilate_d >= jcp.id)
                || (!(jcp.signed_input || jcp.src_zero_point)
                        && (jcp.kd - 1) * (jcp.dilate_d + 1)
                                < nstl::max(jcp.f_pad, jcp.back_pad))) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        if (jcp.is_fused_conv) {
            mov(aux_reg_inp_buffer_ptr, reg_inp_buffer_ptr);
        } else {
            mov(aux_reg_inp, reg_inp);
        }
        mov(aux_reg_ker, reg_ker);
    }

    if ((jcp.signed_input || jcp.src_zero_point) && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }
    mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    if ((jcp.signed_input || jcp.src_zero_point) || (jcp.dilate_h >= jcp.ih)
            || (!(jcp.signed_input || jcp.src_zero_point)
                    && (jcp.kh - 1) * (jcp.dilate_h + 1)
                            < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            mov(aux_reg_inp, ptr[aux_reg_inp_buffer_ptr]);
            add(aux_reg_inp, reg_inp);
        }
        compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, false);

        add(aux_reg_ker, shift_kernel_ptr);
        if (jcp.is_fused_conv) {
            add(aux_reg_inp_buffer_ptr, sizeof(void *));
        } else {
            add(aux_reg_inp, shift_input_ptr * (jcp.dilate_h + 1));
        }
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);
    if ((jcp.signed_input || jcp.src_zero_point) && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, shift_input_ptr * jcp.ih * (jcp.dilate_d + 1));
        add(aux_reg_ker_d, shift_kernel_ptr * jcp.kh);
        dec(reg_ki);
        jne(kd_label, T_NEAR);

        L(skip_kd_loop);
        if (jcp.signed_input || jcp.src_zero_point) {
            mov(reg_ki, ptr[param1 + GET_OFF(back_overflow)]);
            cmp(reg_ki, 0);
            je(no_back_overflow_label, T_NEAR);
            L(back_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov(reg_kj, jcp.kh);
                L(d_h_back_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    add(aux_reg_ker, shift_kernel_ptr);
                    dec(reg_kj);
                    jne(d_h_back_overflow_label);
                }
                add(aux_reg_ker_d, shift_kernel_ptr * jcp.kh);
                dec(reg_ki);
                jne(back_overflow_label);
            }
            L(no_back_overflow_label);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::icb_loop(
        int ur_w, int pad_l, int pad_r, bool is_last_sp_block) {
    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);
    const bool do_icb_loop
            = jcp.is_depthwise ? jcp.nb_ch > jcp.nb_ch_blocking : jcp.nb_ic > 1;
    if (jcp.ngroups % jcp.ch_block != 0 || jcp.ic_without_padding != jcp.ic) {
        Label common_ker, end_ker;
        if (do_icb_loop) {
            if (jcp.is_depthwise)
                cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
            else
                cmp(reg_icb, 1); // The last IC block
            jne(common_ker, T_NEAR);
        }
        kh_loop(ur_w, pad_l, pad_r,
                is_last_sp_block ? last_sp_block : last_ic_block);
        if (do_icb_loop) {
            jmp(end_ker, T_NEAR);

            L(common_ker);
            kh_loop(ur_w, pad_l, pad_r, no_last_block);

            L(end_ker);
        }
    } else {
        kh_loop(ur_w, pad_l, pad_r, no_last_block);
    }
    // End of IC Loop
    if (do_icb_loop) {
        int inp_step = jcp.ic_block;
        int ker_step = jcp.kd * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
        add(reg_inp, jcp.typesize_in * inp_step);
        add(reg_ker, jcp.typesize_in * ker_step);

        dec(reg_icb);
        cmp(reg_icb, 0);
        jg(icb_label, T_NEAR);

        sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
        sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);
    }

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);

        jne(common_store, T_NEAR);

        store_output(ur_w, true); // last oc block
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);
    } else {
        store_output(ur_w, false);
    }
}

template <cpu_isa_t isa, typename Vmm>
void _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::generate() {
    Label permute_index_table;
    int in_ic_shift = jcp.is_fused_conv ? jcp.dw_conv_buffer_oc
                                        : jcp.ic_without_padding * jcp.ngroups;
    const int urw_inp_stride = jcp.ur_w * jcp.stride_w;
    const int n_urw_l_pad
            = nstl::min(div_up(jcp.l_pad, urw_inp_stride), jcp.ow / jcp.ur_w);
    const int inp_shift_pad = nstl::max(0,
            jcp.typesize_in * (n_urw_l_pad * urw_inp_stride - jcp.l_pad)
                    * in_ic_shift);
    int inp_shift = jcp.typesize_in * (jcp.ur_w * jcp.stride_w * in_ic_shift);
    int out_shift = jcp.typesize_out
            * (jcp.ur_w * jcp.oc_without_padding * jcp.ngroups);
    preamble();

    if (jcp.is_depthwise) {
        const bool is_zero_point = jcp.src_zero_point || jcp.dst_zero_point;
        int idx = ker_max_reg + 1 - jcp.max_regs_ur - 2 * is_zero_point;
        if (!jcp.is_resrc_depthwise) vmm_dw_src = Vmm(--idx);
        vmm_dw_tmp = Vmm(--idx);
        if (jcp.signed_input) {
            vmm_dw_shifted_zero = Vmm(--idx);
            --idx; // due to extra register used for shifts and compensations
        }
        assert(IMPLICATION(
                !is_zero_point, idx == ker_max_reg - ker_dw_reg_base_idx));
    }

    if (!jcp.is_depthwise) {
        auto vmm_one_128 = Xbyak::Xmm(vmm_one.getIdx());
        mov(reg_scratch, 0x10001);
        uni_vmovq(vmm_one_128, reg_scratch);
        uni_vpbroadcastd(vmm_one, vmm_one_128);
    }

    if (jcp.is_fused_conv) {
        mov(reg_inp_buffer_ptr, ptr[param1 + GET_OFF(src)]);
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format wc with c=jcp.dw_conv_buffer_oc.
        Example: [ptr_to_inp_row0, ptr_to_inp_row1, ptr_to_inp_row2].
        Traverse the data as
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row0 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row1 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row2 ...
        */
        xor_(reg_inp, reg_inp);
    } else {
        mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    }
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
    }

    const int extended_filter_size
            = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int r_pad = nstl::max(0, jcp.r_pad);
    const int ow_with_no_rpad = 1
            + (jcp.iw + jcp.l_pad + nstl::min(0, jcp.r_pad)
                      - extended_filter_size)
                    / jcp.stride_w;
    const int n_urw_per_ow_block = jcp.ow_block / jcp.ur_w;
    const int max_safe_iw = nstl::max(
            0, jcp.iw - div_up(ic_sub_step, jcp.ic_without_padding));
    const int max_safe_ow = jcp.ic_without_padding % ic_sub_step == 0
            ? jcp.ow
            : (max_safe_iw + jcp.l_pad - extended_filter_size) / jcp.stride_w;
    Label middle_block_label, done_compute;
    std::vector<Label> ow_block_jmp_table;

    // r_pad_fall_through is a special ow_block, where the block overlaps
    // both middle_block and r_pad/ur_w_tail region when it exists.
    // The number of ur_w's to compute in middle_block before executing
    // r_pad region is stored in r_pad_fall_through_n_urw and the ow_block
    // number is stored in r_pad_fall_through_ow_block.
    int r_pad_fall_through_ow_block = 0;
    int r_pad_fall_through_n_urw = 0;

    if (jcp.nb_ow > 1) {
        // Only one ow block is processed, per jit call.
        // Number of this ow block is passed as parameter owb,
        // and padding processing depends on this number.
        //
        // The compute block to run is determined by using a jmp-table.
        // jmp-table Layout:
        //  idx -> addr
        //  0   -> [...l_pad_region label[0]...]
        //         : : : : : : : : : : : : : : :
        //  L ->   [...l_pad_region label[L]...]
        //  L+1 -> [...r_pad_region label[0]...]
        //         : : : : : : : : : : : : : : :
        //  L+R -> [...r_pad_region label[R]...]
        //
        // Note: Label for middle_block is not stored in the jmp-table.
        //
        // During jit call, the jump address is calculated as below:
        // if (owb < n) {
        //   jmp([jmp_table + owb*sizeof(void*)]);
        // } else if (owb < X) {
        //   // X is the number of ow_blocks before r_pad region (see below).
        //   jmp(middle_block);
        // } else {
        //   sub(owb, X);
        //   jmp([jmp_table + owb*sizeof(void*) + L*sizeof(void)]);
        // }
        //
        // To configure the jmp-table, we need to determine some constants
        // (namely, r_pad_fall_through_n_urw, r_pad_fall_through_ow_block,
        // n_l_pad_labels, n_labels) ahead of writing the compute assembly. So,
        // we simulate the filter path without writing the assembly initially.
        // This makes the math for calculating the constants become simple and
        // self explanatory.

        // Begin simulation without writing assembly
        int n_l_pad_labels = 0;
        int n_labels = 0;
        int cur_ow = 0;

        // l_pad region:
        n_l_pad_labels = div_up(n_urw_l_pad, n_urw_per_ow_block);
        n_labels = n_l_pad_labels;
        cur_ow += n_urw_l_pad * jcp.ur_w;

        // middle_region:
        int n_urw_middle_block_loop = 0;
        int cur_r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, cur_ow + jcp.ur_w, jcp.iw,
                        jcp.stride_w, extended_filter_size));
        if (cur_ow + jcp.ur_w <= jcp.ow && cur_r_pad == 0) {
            n_urw_middle_block_loop
                    = nstl::max(0,
                              nstl::min(ow_with_no_rpad, max_safe_ow) - cur_ow)
                    / jcp.ur_w;
            cur_ow += n_urw_middle_block_loop * jcp.ur_w;
        }
        r_pad_fall_through_n_urw = (cur_ow / jcp.ur_w) % n_urw_per_ow_block;
        r_pad_fall_through_ow_block = cur_ow / (n_urw_per_ow_block * jcp.ur_w);

        // r_pad or last_sp_block
        if (cur_ow + jcp.ur_w <= jcp.ow) {
            if (r_pad_fall_through_n_urw == 0) ++n_labels;
            const int n_urw_r_pad_region = (jcp.ow - cur_ow) / jcp.ur_w;
            n_labels += nstl::max(0,
                    div_up(r_pad_fall_through_n_urw + n_urw_r_pad_region,
                            n_urw_per_ow_block)
                            - 1);
        }

        if (jcp.ur_w_tail != 0) {
            if (jcp.ow % jcp.ow_block == jcp.ur_w_tail) ++n_labels;
        }
        // End of simulation

        ow_block_jmp_table.resize(n_labels);

        // Begin jump-table logic
        Label ow_block_jmp_table_label;
        if (!ow_block_jmp_table.empty())
            mov(reg_jmp_tbl_base, ow_block_jmp_table_label);
        mov(reg_oi, n_urw_per_ow_block);
        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        if (jcp.l_pad > 0) {
            Label middle_or_rpad_check;
            cmp(reg_owb, n_l_pad_labels);
            jge(middle_or_rpad_check, T_NEAR);
            jmp(ptr[reg_jmp_tbl_base + reg_owb * sizeof(void *)]);
            L(middle_or_rpad_check);
            // harness passes shifted src pointer that does not take
            // left-padding into account. So, we must re-shift here.
            const int inp_shift_pad_middle_block = -1 * jcp.typesize_in
                    * nstl::min(jcp.l_pad, n_urw_l_pad * urw_inp_stride)
                    * in_ic_shift;
            add(reg_inp, inp_shift_pad_middle_block);
        }
        if (r_pad_fall_through_n_urw != 0) {
            mov(reg_scratch, r_pad_fall_through_n_urw);
            cmp(reg_owb, r_pad_fall_through_ow_block);
            cmove(reg_oi, reg_scratch);
            if (n_urw_middle_block_loop > 0) {
                sub(reg_owb, r_pad_fall_through_ow_block);
                // simple middle_block
                jle(middle_block_label, T_NEAR);
                dec(reg_owb);
            } else {
                sub(reg_owb, r_pad_fall_through_ow_block + 1);
            }
        } else {
            sub(reg_owb, r_pad_fall_through_ow_block);
            // simple middle_block
            if (n_urw_middle_block_loop) jl(middle_block_label, T_NEAR);
        }
        // r_pad-only region
        if (!ow_block_jmp_table.empty())
            jmp(ptr[reg_jmp_tbl_base + reg_owb * sizeof(void *)
                    + n_l_pad_labels * sizeof(void *)]);

        if (!ow_block_jmp_table.empty()) {
            align(8);
            L(ow_block_jmp_table_label);
            {
                for (size_t i = 0; i < ow_block_jmp_table.size(); ++i) {
                    putL(ow_block_jmp_table[i]);
                }
            }
        }
        // End of jump-table logic
    }

    // Begin kernel
    int cur_ow = 0;
    int cur_n_oi = 0; // used only for jcp.nb_ow > 1 scenario
    int label_cntr = 0;
    int cur_l_pad = 0;
    if (jcp.l_pad > 0) {
        for (cur_l_pad = jcp.l_pad;
                cur_l_pad > 0 && cur_ow + jcp.ur_w <= jcp.ow;
                cur_l_pad -= urw_inp_stride) {
            if (jcp.nb_ow > 1 && cur_n_oi == 0) {
                // cur_n_oi == 0 signifies beginning of new ow_block
                // (or end of previous block)
                const dim_t inp_lpad_region_shift = -label_cntr * jcp.ow_block
                        * jcp.stride_w * in_ic_shift;
                L(ow_block_jmp_table[label_cntr++]);
                // harness passes shifted src pointer that does not take
                // left-padding into account. So, we must re-shift here.
                add(reg_inp, inp_lpad_region_shift);
            }

            cur_ow += jcp.ur_w;
            int cur_r_pad = nstl::max(0,
                    calculate_end_padding(jcp.l_pad, cur_ow, jcp.iw,
                            jcp.stride_w, extended_filter_size));
            icb_loop(jcp.ur_w, cur_l_pad, cur_r_pad, cur_ow > max_safe_ow);
            add(reg_out, out_shift);
            dec(reg_oi);

            if (jcp.nb_ow > 1 && ++cur_n_oi == n_urw_per_ow_block) {
                // We compute one owb per jit call. So, insert an
                // unconditional jmp, after computing one owb.
                jmp(done_compute, T_NEAR);
                cur_n_oi = 0;
            }
        }
        if (jcp.nb_ow == 1 || cur_n_oi != 0) {
            // Let it "fall-through" middle_block_label
            add(reg_inp, inp_shift_pad);
        }
    }

    // middle_block
    {
        int cur_r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, cur_ow + jcp.ur_w, jcp.iw,
                        jcp.stride_w, extended_filter_size));
        if (cur_r_pad == 0 && cur_ow + jcp.ur_w <= jcp.ow) {
            int n_oi_middle_block_loop
                    = nstl::max(0,
                              nstl::min(ow_with_no_rpad, max_safe_ow) - cur_ow)
                    / jcp.ur_w;
            if (jcp.nb_ow == 1 && n_oi_middle_block_loop > 1)
                mov(reg_oi, n_oi_middle_block_loop);
            L(middle_block_label);
            if (n_oi_middle_block_loop > 0) {
                icb_loop(jcp.ur_w, 0, 0, false);
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
                if (n_oi_middle_block_loop > 1) {
                    dec(reg_oi);
                    jg(middle_block_label, T_NEAR);
                }
            }
            cur_ow += n_oi_middle_block_loop * jcp.ur_w;
            cur_n_oi = (cur_n_oi + n_oi_middle_block_loop) % n_urw_per_ow_block;
        }
    }

    // r_pad region or last_sp_block
    if (cur_ow + jcp.ur_w <= jcp.ow) {
        if (jcp.nb_ow > 1) {
            if (cur_n_oi == 0) {
                jmp(done_compute, T_NEAR);
            } else {
                // r_pad fall-through
                mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
                cmp(reg_owb, r_pad_fall_through_ow_block);
                jne(done_compute, T_NEAR);
            }
        }

        while (cur_ow + jcp.ur_w <= jcp.ow) {
            if (jcp.nb_ow > 1 && cur_n_oi == 0) {
                L(ow_block_jmp_table[label_cntr++]);
            }
            cur_ow += jcp.ur_w;
            int cur_r_pad = calculate_end_padding(jcp.l_pad, cur_ow, jcp.iw,
                    jcp.stride_w, extended_filter_size);
            assert(cur_r_pad > 0 || cur_ow > max_safe_ow); // else, why be here?
            icb_loop(jcp.ur_w, 0, cur_r_pad, cur_ow > max_safe_ow);
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);

            if (jcp.nb_ow > 1 && ++cur_n_oi == n_urw_per_ow_block) {
                // We compute one owb per jit call. So, insert an
                // unconditional jmp, after computing one owb.
                jmp(done_compute, T_NEAR);
                cur_n_oi = 0;
            }
        }
        // Let it fall-through ur_w_tail
    }

    // ur_w_tail
    if (jcp.ur_w_tail != 0) {
        if (jcp.nb_ow > 1) {
            if (cur_n_oi == 0) {
                jmp(done_compute, T_NEAR);
                L(ow_block_jmp_table[label_cntr++]);
            } else {
                // In case, when there is no r_pad region, then there exists an
                // ambiguity btw middle_blocks and r_pad_fall_through_ow_block.
                // If not properly distinguished, there can be a race condition
                // as middle_blocks and r_pad_fall_through_ow_block both try to
                // compute ur_w_tail work at the end.
                mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
                cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
                jne(done_compute, T_NEAR);
            }
        }
        icb_loop(jcp.ur_w_tail, nstl::max(0, cur_l_pad), r_pad, true);
    }
    L(done_compute);
    assert(ow_block_jmp_table.size() == static_cast<size_t>(label_cntr));

    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_fwd_kernel<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || p.contain(sum, 0);
        case 2:
            return (p.contain(sum, 0) && is_eltwise(1))
                    || (p.contain(sum, 1) && is_eltwise(0));
        default: return false;
    }

    return false;
}

template <cpu_isa_t isa, typename Vmm>
bool _jit_uni_x8s8s32x_fwd_kernel<isa, Vmm>::maybe_eltwise(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* eltwise before sum */
        return p.contain(eltwise, 0);
    } else if (position == 1) {
        /* eltwise after sum */
        return p.contain(sum, 0) && p.contain(eltwise, 1);
    }

    return false;
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_fwd_kernel<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_2d = ndims == 4;
    const bool is_3d = ndims == 5;
    const bool is_avx2 = isa == avx2;
    assert(is_1d || is_2d || is_3d);

    if (!(mayiuse(isa)
                && one_of(src_d.data_type(), data_type::u8, data_type::s8)
                && weights_d.data_type() == data_type::s8
                && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                        data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.ur_h = 1; /* no code-unrolling by h so far */

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);

    jcp.signed_input = src_d.data_type() == data_type::s8;
    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common
            = zp.common(DNNL_ARG_SRC); // otherwise, it's per-channel
    assert(IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common));

    if ((jcp.dst_zero_point || jcp.src_zero_point) && jcp.is_fused_conv)
        return status::unimplemented;

    if (is_3d && jcp.is_depthwise) return status::unimplemented;

    if (jcp.is_depthwise) {
        jcp.ch_block = is_avx2 ? 8 : 4;
        jcp.ic_block = 1;
        jcp.oc_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.ic_block = is_avx2 ? 8 : 4;
        jcp.oc_block = is_avx2 ? 8 : 4;

        if (jcp.ngroups == 1) {
            /* For non grouped convolutions, pad channels by 8 if needed */
            jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        } else if (jcp.ngroups != 1
                && ((jcp.ic % jcp.ic_block != 0)
                        || (jcp.oc % jcp.oc_block != 0))) {
            /* For grouped convolutions, oneDNN doesn't support padding.
             * When channels per group is not multiple of 8:
             * - Use Xmm when channels per group is multiple of 4.
             * - Otherwise return unimplemented */
            jcp.oc_block = jcp.ic_block = 4;
        }
        if (jcp.ic % jcp.ic_block != 0 || jcp.oc % jcp.oc_block != 0)
            return status::unimplemented;
    }

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.is_resrc_depthwise = true && jcp.is_depthwise && jcp.stride_w < jcp.kw
            && jcp.kw < 4 && jcp.dilate_w == 0;

    if (jcp.is_depthwise) {
        jcp.max_regs_ur = 14 - !jcp.is_resrc_depthwise - 2 * jcp.signed_input;
    } else {
        jcp.max_regs_ur = 12;
    }

    if (jcp.src_zero_point || jcp.dst_zero_point) jcp.max_regs_ur = 9;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        using namespace memory_extra_flags;
        const int c_mask = 0x1,
                  g_mask = 0x3; // mask for i/o-channel and ngroups
        format_tag_t wei_tag;
        if (jcp.ic_block == 8 || jcp.ch_block == 8) {
            if (is_1d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goiw8g : gOIw2i8o4i
                                      : OIw2i8o4i;
            } else if (is_2d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goihw8g : gOIhw2i8o4i
                                      : OIhw2i8o4i;
            } else {
                wei_tag = with_groups ? gOIdhw2i8o4i : OIdhw2i8o4i;
            }
        } else {
            if (is_avx2) {
                assert(with_groups && jcp.ic_block == 4);
                wei_tag = is_3d ? gOIdhw4o4i : is_2d ? gOIhw4o4i : gOIw4o4i;
            } else {
                if (is_1d) {
                    wei_tag = with_groups ? jcp.is_depthwise ? Goiw4g : gOIw4o4i
                                          : OIw4o4i;
                } else if (is_2d) {
                    wei_tag = with_groups
                            ? jcp.is_depthwise ? Goihw4g : gOIhw4o4i
                            : OIhw4o4i;
                } else {
                    wei_tag = with_groups ? gOIdhw4o4i : OIdhw4o4i;
                }
            }
        }
        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

        if (jcp.signed_input) {
            want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::scale_adjust;
            want_wei_md.extra.compensation_mask
                    = (with_groups && !jcp.is_depthwise) ? g_mask : c_mask;
            want_wei_md.extra.scale_adjust = 0.5f;
        }
        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask
                    = (with_groups && !jcp.is_depthwise) ? g_mask : c_mask;
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;
    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag) return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    // Try to use 4 channel-groups at a time to avoid false sharing (depthwise)
    int nb_ch_blocking = 4;
    for (/* init above */; nb_ch_blocking > 1; --nb_ch_blocking)
        if (jcp.nb_ch % nb_ch_blocking == 0) break;
    jcp.nb_ch_blocking = jcp.is_depthwise ? nb_ch_blocking : 1;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    auto is_oc_blocking_ok = [&](int block) {
        int ur_w = nstl::min(jcp.ow, jcp.max_regs_ur / (block + 1));
        return jcp.nb_oc % block == 0 && jcp.l_pad <= ur_w
                && jcp.ow % ur_w != 1;
    };

    // choose nb_oc work chunk size for distribution within threads
    int max_threading_nb_oc_chunk = 4;

    jcp.nb_oc_blocking_thr_chunk
            = nstl::min(max_threading_nb_oc_chunk, jcp.nb_oc);
    for (; jcp.nb_oc_blocking_thr_chunk > 1; --jcp.nb_oc_blocking_thr_chunk) {
        if (is_oc_blocking_ok(jcp.nb_oc_blocking_thr_chunk)) break;
    }

    // choose oc blocking for computational kernel
    jcp.nb_oc_blocking = jcp.nb_oc_blocking_thr_chunk;

    if (jcp.is_resrc_depthwise)
        jcp.ur_w = (jcp.max_regs_ur - jcp.kw + jcp.stride_w)
                / (jcp.nb_ch_blocking + jcp.stride_w);
    else
        jcp.ur_w = jcp.max_regs_ur
                / (jcp.is_depthwise ? jcp.nb_ch_blocking
                                    : jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    int base_work_amount = jcp.mb * jcp.nb_ch * jcp.oh
            * (jcp.nb_oc / jcp.nb_oc_blocking_thr_chunk);
    float best_thr_eff
            = (float)base_work_amount / rnd_up(base_work_amount, jcp.nthr);
    int max_nb_ow = div_up(jcp.ow, jcp.ur_w);
    for (int nb_ow = 1; nb_ow <= max_nb_ow; ++nb_ow) {
        int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), jcp.ur_w), jcp.ow);
        if (ow_block < jcp.nb_oc_blocking_thr_chunk * jcp.oc_block
                && best_thr_eff > 0.8f)
            break;
        if (div_up(jcp.ow, ow_block) != nb_ow) continue;
        auto work_amount = base_work_amount * nb_ow;
        float thr_eff = (float)work_amount / rnd_up(work_amount, jcp.nthr);
        if (ow_block >= jcp.ur_w && thr_eff > 1.1f * best_thr_eff) {
            jcp.ow_block = ow_block;
            best_thr_eff = thr_eff;
        }
        if (best_thr_eff > 0.9f) break;
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    bool args_ok = true && jcp.oc % jcp.oc_block == 0
            && IMPLICATION(!jcp.is_1stconv, jcp.ic % jcp.ic_block == 0);
    if (!args_ok) return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_fwd_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {

    if (jcp.signed_input) {
        dim_t count = attr.output_scales_.count_ == 1
                ? (dim_t)8
                : attr.output_scales_.count_;
        scratchpad.book<float>(key_conv_adjusted_scales, count);
    }
}

template struct _jit_uni_x8s8s32x_fwd_kernel<avx2, Ymm>;
template struct _jit_uni_x8s8s32x_fwd_kernel<avx2, Xmm>;
template struct _jit_uni_x8s8s32x_fwd_kernel<sse41, Xmm>;
template struct jit_uni_x8s8s32x_fwd_kernel<avx2>;
template struct jit_uni_x8s8s32x_fwd_kernel<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
