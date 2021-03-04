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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::load_src(int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc
                        = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);

                int b_off = ch * ch_blk + i * 4;
                if (this->jcp.with_bias)
                    uni_vmovups(
                            vmm_acc, vmmword[reg_bias + b_off * sizeof(float)]);
                else
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);

                int o_off = ch * ocb_stride + ow * ow_stride + i * 4;
                if (this->jcp.with_sum)
                    uni_vaddps(vmm_acc, vmm_acc,
                            vmmword[reg_output + o_off * sizeof(float)]);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, int pad_l, int pad_r) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
            ? ch_blk
            : (jcp.is_fused_conv ? 1 : jcp.ih) * jcp.iw * ch_blk;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input, ptr[aux_reg_input_buffer_ptr]);
            add(aux_reg_input, reg_iw_offset);
        }
        int repeats = isa == sse41 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int kw = 0; kw < jcp.kw; kw++) {
                    int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk
                            + i * 4;

                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker,
                            ptr[aux_reg_kernel + ker_off * sizeof(float)]);

                    int ow_start = get_ow_start(kw, pad_l);
                    int ow_end = get_ow_end(ur_w, kw, pad_r);
                    for (int ow = ow_start; ow < ow_end; ow++) {
                        int inp_off = ch * icb_stride
                                + (ow * stride_w - pad_l) * iw_stride
                                + kw * dilate_w * iw_stride + i * 4;

                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src,
                                ptr[aux_reg_input + inp_off * jcp.typesize_in]);

                        Vmm vmm_acc = get_acc_reg(
                                i * ur_ch_blocks * ur_w + ch * ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * sizeof(float));
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add(aux_reg_input_buffer_ptr, sizeof(void *));
        } else {
            add(aux_reg_input, ih_stride * dilate_h * sizeof(float));
        }

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_activation(
        int ur_ch_blocks, int ur_w) {
    if (this->jcp.with_eltwise) {
        int repeats = isa == sse41 ? 2 : 1;
        eltwise_injector_->compute_vector_range(
                4, repeats * ur_w * ur_ch_blocks + 4);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                const int o_off = ch * ocb_stride + ow * ow_stride + i * 4;

                Vmm vmm_dst
                        = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);

                uni_vmovups(
                        vmmword[reg_output + o_off * sizeof(float)], vmm_dst);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw
            * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w, pad_l, pad_r);
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);
    };

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_tail = jcp.nb_ch % jcp.nb_ch_blocking;

        mov(aux_reg_ch_blocks, reg_ch_blocks);
        push(reg_kernel);
        push(reg_input);
        push(reg_output);
        if (jcp.with_bias) push(reg_bias);

        if (ch_tail) {
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            jl(ch_tail_label, T_NEAR);
        }

        L(ch_loop_label);
        {
            compute(jcp.nb_ch_blocking);
            add(reg_kernel, wei_ch_stride);
            add(reg_input, inp_ch_stride);
            add(reg_output, out_ch_stride);
            if (jcp.with_bias) add(reg_bias, bias_stride);
            sub(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            jge(ch_loop_label, T_NEAR);
        }

        if (ch_tail) {
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            jle(skip_ch_tail_label, T_NEAR);
            compute(ch_tail);
            L(skip_ch_tail_label);
        }

        if (jcp.with_bias) pop(reg_bias);
        pop(reg_output);
        pop(reg_input);
        pop(reg_kernel);

    } else {
        compute(ur_ch_blocks);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::ow_loop(int ur_ch_blocks) {

    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;
    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * dat_c_stride;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

    int inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * dat_c_stride;

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    assert(jcp.nb_ow <= 1);

    if (r_pad1 > 0) n_oi--;
    xor_(reg_oi, reg_oi);
    if (ow == ur_w) {
        compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad1);
            add(reg_input, inp_shift_pad);
            add(reg_output, out_shift);
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        } else {
            if (l_pad > 0) {
                compute_loop(ur_w, ur_ch_blocks, l_pad, 0);
                add(reg_input, inp_shift_pad);
                add(reg_output, out_shift);
                inc(reg_oi);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                Label ow_loop_label;
                L(ow_loop_label);
                {
                    compute_loop(ur_w, ur_ch_blocks, 0, 0);
                    add(reg_input, inp_shift);
                    add(reg_output, out_shift);

                    inc(reg_oi);
                    cmp(reg_oi, n_oi);
                    jl(ow_loop_label, T_NEAR);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(ur_w, ur_ch_blocks, 0, r_pad1);
                add(reg_input, inp_shift);
                add(reg_output, out_shift);
            }
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::generate() {
    this->preamble();

    if (jcp.is_fused_conv) {
        mov(reg_input_buffer_ptr, ptr[this->param1 + GET_OFF(src)]);
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format Cwc with blocking nb_ch_blocking.
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
        xor_(reg_iw_offset, reg_iw_offset);
    } else {
        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    }
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias) mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    if (is_src_layout_nxc()) {
        ow_loop(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, jcp.nb_ch_blocking);
        jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

        ow_loop(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            L(ch_blocks_tail_label);

            cmp(reg_ch_blocks, ch_blocks_tail);
            jne(exit_label, T_NEAR);

            ow_loop(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    this->postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

template struct jit_uni_dw_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_dw_conv_fwd_kernel_f32<sse41>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int w = 0; w < ur_str_w; w++) {
                Vmm vmm_acc = get_acc_reg(
                        i * ur_ch_blocks * ur_str_w + ch * ur_str_w + w);
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_str_w) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label);
        {
            int repeats = isa == sse41 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    int ker_off = ch * kh * kw * ch_blk + i * 4;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker,
                            ptr[aux1_reg_kernel + ker_off * sizeof(float)]);

                    for (int w = 0; w < ur_str_w; w++) {
                        int ddst_off = (ch * oh * ow + w) * ch_blk + i * 4;

                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src,
                                ptr[aux1_reg_ddst + ddst_off * sizeof(float)]);

                        Vmm vmm_acc = get_acc_reg(i * ur_ch_blocks * ur_str_w
                                + ch * ur_str_w + w);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }

            add(aux1_reg_kernel, ch_blk * stride_w * sizeof(float));
            sub(aux1_reg_ddst, ch_blk * sizeof(float));

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw * ch_blk * stride_h * sizeof(float));
        sub(aux_reg_ddst, ow * ch_blk * sizeof(float));

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int w = 0; w < ur_str_w; w++) {
                int dsrc_off = (ch * ih * iw + w * stride_w) * ch_blk + i * 4;
                Vmm vmm_acc = get_acc_reg(
                        i * ur_ch_blocks * ur_str_w + ch * ur_str_w + w);

                uni_vmovups(ptr[reg_dsrc + dsrc_off * sizeof(float)], vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::loop_body(
        int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label);
    {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_str_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label);
    {
        int ur_w = 1;

        cmp(reg_ur_str_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::generate() {
    preamble();

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_str_w, ptr[this->param1 + GET_OFF(ur_str_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<sse41>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_filter() {
    for (int r = 0; r < reg_repeats; ++r) {
        for (int i = 0; i < jcp.kw; ++i) {
            Vmm vmm_acc = get_acc_reg(r * jcp.kw + i);
            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_filter() {
    for (int r = 0; r < reg_repeats; ++r) {
        const int reg_set = r * jcp.kw;
        for (int i = 0; i < jcp.kw; ++i) {
            int off_filter = (reg_set + i) * simd_w;
            Vmm vmm_acc = get_acc_reg(reg_set + i);
            uni_vmovups(vmm_acc,
                    vmmword[reg_tmp_filter + off_filter * sizeof(float)]);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_bias() {
    for (int r = 0; r < reg_repeats; ++r) {
        Vmm vmm_bias = get_bias_reg(r);
        uni_vpxor(vmm_bias, vmm_bias, vmm_bias);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_bias() {
    for (int r = 0; r < reg_repeats; ++r) {
        Vmm vmm_bias = get_bias_reg(r);
        uni_vmovups(
                vmm_bias, vmmword[reg_bias_baddr + r * simd_w * sizeof(float)]);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;

    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int r = 0; r < reg_repeats; ++r) {
        for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
            int off_output = (i_ur * reg_repeats + r) * simd_w;
            Vmm vmm_output = get_output_reg(r);
            uni_vmovups(vmm_output,
                    ptr[reg_tmp_output + off_output * sizeof(float)]);
            if (i_ur == 0) {
                for (int c = 0; c < input_overlap; ++c) {
                    int off_input
                            = ((c - pad_offset) * reg_repeats + r) * simd_w;
                    if (off_input < 0 && unroll_w == jcp.ow) continue;

                    const bool over_steps_bdry = true && is_last_block
                            && (c - pad_offset + r_pad > right_border);
                    if (over_steps_bdry) continue;

                    Vmm vmm_input
                            = get_input_reg((c % jcp.kw) * reg_repeats + r);
                    uni_vmovups(vmm_input,
                            ptr[reg_tmp_input + off_input * sizeof(float)]);
                }
            } else {
                for (int c = 0; c < cascade_input; ++c) {
                    int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                    int off_input
                            = ((overlap + c - pad_offset) * reg_repeats + r)
                            * simd_w;
                    if (off_input < 0 || overlap + c + l_pad > right_border)
                        continue;

                    const bool over_steps_bdry = true && is_last_block
                            && (overlap + c - pad_offset + r_pad
                                    > right_border);
                    if (over_steps_bdry) continue;

                    Vmm vmm_input = get_input_reg(
                            ((overlap + c) % jcp.kw) * reg_repeats + r);
                    uni_vmovups(vmm_input,
                            ptr[reg_tmp_input + off_input * sizeof(float)]);
                }
            }

            for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
                int io_overlap = i_kw + (i_ur * jcp.stride_w);

                /* Don't apply FMAs that fall into the padded region */
                if (io_overlap - l_pad < 0
                        || io_overlap - jcp.l_pad >= right_border)
                    continue;

                const bool over_steps_bdry = true && is_last_block
                        && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
                if (over_steps_bdry) continue;

                Vmm vmm_input = get_input_reg(
                        ((io_overlap - l_pad) % jcp.kw) * reg_repeats + r);
                Vmm vmm_acc = get_acc_reg(i_kw * reg_repeats + r);
                Vmm vmm_aux = isa == sse41 ? get_aux_reg() : vmm_input;
                if (isa == sse41) uni_vmovups(vmm_aux, vmm_input);
                uni_vfmadd231ps(vmm_acc, vmm_aux, vmm_output);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias_step_unroll(
        const int unroll_w) {
    for (int r = 0; r < reg_repeats; ++r) {
        for (int i = 0; i < unroll_w; ++i) {
            Vmm vmm_bias = get_bias_reg(r);
            int off_output = (i * reg_repeats + r) * simd_w;
            if (isa == sse41) {
                /* Need to support unaligned address loads for SSE41 */
                Vmm vmm_output = get_output_reg(1 + r);
                uni_vmovups(vmm_output,
                        ptr[reg_tmp_output + off_output * sizeof(float)]);
                uni_vaddps(vmm_bias, vmm_bias, vmm_output);
            } else {
                uni_vaddps(vmm_bias, vmm_bias,
                        vmmword[reg_tmp_output + off_output * sizeof(float)]);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_filter() {
    for (int r = 0; r < reg_repeats; ++r) {
        const int reg_set = r * jcp.kw;
        for (int i = 0; i < jcp.kw; ++i) {
            int off_filter = (i + reg_set) * simd_w;
            Vmm vmm_acc = get_acc_reg(i + reg_set);
            uni_vmovups(vmmword[reg_tmp_filter + off_filter * sizeof(float)],
                    vmm_acc);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_bias() {
    for (int r = 0; r < reg_repeats; ++r) {
        Vmm vmm_bias = get_bias_reg(r);
        uni_vmovups(
                vmmword[reg_bias_baddr + r * simd_w * sizeof(float)], vmm_bias);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias_loop(
        const int block_size) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(block_size, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > block_size ? jcp.ow % block_size : 0;

    const int ch_offset = jcp.ch_block;

    mov(reg_oh, ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_index)]);
    mov(reg_oh_worksize,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {

            compute_bias_step_unroll(unroll_w);
            add(reg_tmp_output, unroll_w * ch_offset * sizeof(float));

            dec(reg_iter_ow_blk);
            cmp(reg_iter_ow_blk, 0);
            jg(ow_blk_label, T_NEAR);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w);
            add(reg_tmp_output, tail_w * ch_offset * sizeof(float));
        }

        inc(reg_oh);
        cmp(reg_oh, reg_oh_worksize);
        jl(oh_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_zero_filter() {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_zeroing_label;

    mov(reg_exec_flags,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, exec_flags)]);
    and_(reg_exec_flags, FLAG_ZERO_FILTER);
    test(reg_exec_flags, reg_exec_flags);
    je(skip_zeroing_label);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    mov(reg_kh, jcp.kh);
    L(kh_loop_label);
    {
        store_filter();

        add(reg_tmp_filter, jcp.kw * ch_offset * sizeof(float));
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_loop_label);
    }

    /* Comeback pointers */
    sub(reg_tmp_filter, jcp.kh * jcp.kw * ch_offset * sizeof(float));

    L(skip_zeroing_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_h_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh_count, 0);
    je(skip_loop_label, T_NEAR);

    mov(reg_kh, reg_kh_count);
    L(kh_loop_label);
    {
        load_filter();
        compute_ow_step_unroll(unroll_w, l_pad, pad_offset, ow_block);
        store_filter();

        add(reg_tmp_filter, jcp.kw * ch_offset * sizeof(float));
        add(reg_tmp_input, jcp.iw * ch_offset * sizeof(float));
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_loop_label);
    }

    /* Comeback pointers */
    Label kh_comeback_label;
    mov(reg_kh, reg_kh_count);
    L(kh_comeback_label);
    {
        sub(reg_tmp_input, jcp.iw * ch_offset * sizeof(float));
        sub(reg_tmp_filter, jcp.kw * ch_offset * sizeof(float));
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_comeback_label, T_NEAR);
    }

    L(skip_loop_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    // last index of output that is not influenced by right padding
    const size_t io_overlap
            = jcp.oh - 1 - utils::div_up(jcp.b_pad, jcp.stride_h);

    const int ch_offset = jcp.ch_block;
    const int t_overlap_off = jcp.t_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;
    const int b_overlap_off = jcp.b_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;

    Label tpad_loop_label, h_loop_label, skip_tpad_label, skip_bpad_label;

    mov(reg_oh, ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_index)]);
    mov(reg_oh_worksize,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_count)]);
    mov(reg_kh_count,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, kh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    L(h_loop_label);
    {

        compute_h_step(unroll_w, l_pad, pad_offset, ow_block);

        add(reg_tmp_output, jcp.ow * ch_offset * sizeof(float));

        /* If within the top_pad region */
        if (jcp.t_pad > 0) {
            /* Skip t_pad area if no longer in initial h_block */
            cmp(reg_oh, jcp.t_pad);
            jg(skip_tpad_label, T_NEAR);

            cmp(reg_kh_count, jcp.kh);
            jge(skip_tpad_label, T_NEAR);

            add(reg_kh_count, t_overlap_off);
            sub(reg_tmp_filter,
                    t_overlap_off * jcp.kw * ch_offset * sizeof(float));

            /* kernel has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add(reg_tmp_input,
                        inp_corr * jcp.iw * ch_offset * sizeof(float));
            }
            jmp(tpad_loop_label, T_NEAR);
        }

        L(skip_tpad_label);

        cmp(reg_oh, io_overlap);
        jl(skip_bpad_label, T_NEAR);
        sub(reg_kh_count, b_overlap_off);

        L(skip_bpad_label);
        add(reg_tmp_input, jcp.stride_h * jcp.iw * ch_offset * sizeof(float));

        L(tpad_loop_label);

        inc(reg_oh);

        cmp(reg_oh, reg_oh_worksize);
        jl(h_loop_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ow_block_unroll() {

    const int ch_offset = jcp.ch_block;
    int ow = jcp.ow;
    int pad_offset = 0;
    int l_pad = jcp.l_pad;
    int r_pad = jcp.r_pad;

    /* Is this strictly defined by:
     * -code-size (?)
     * -address size (?) */
    const int max_unroll_w = 30;
    const int block_size = 15;

    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_w_trips = 0;
    const bool do_unroll_w = jcp.ow > max_unroll_w;

    if (do_unroll_w) {
        unroll_w = nstl::min(block_size, jcp.ow);
        unroll_w_trips = ow / unroll_w;
        /* calculate tail */
        unroll_w_tail = ow % unroll_w;
        /* Perform some rebalancing if tail too small*/
        if ((unroll_w_tail == 0 && r_pad != 0)
                || (r_pad > 0 && r_pad >= unroll_w_tail)) {
            if (unroll_w_trips > 1) {
                unroll_w_tail += unroll_w;
                unroll_w_trips--;
            } else {
                /* Idealy, this case shouldn't happen */
                unroll_w_tail += (unroll_w - unroll_w / 2);
                unroll_w = unroll_w / 2;
            }
        }
    } else {
        unroll_w_tail = jcp.ow;
    }
    if (jcp.with_bias) {
        Label skip_load_bias;
        mov(reg_bias_baddr,
                ptr[this->param1 + offsetof(jit_dw_conv_call_s, bias)]);

        zero_bias();

        mov(reg_exec_flags,
                ptr[this->param1 + offsetof(jit_dw_conv_call_s, exec_flags)]);
        and_(reg_exec_flags, FLAG_ZERO_BIAS);
        test(reg_exec_flags, reg_exec_flags);
        jne(skip_load_bias);

        load_bias();

        L(skip_load_bias);
        compute_bias_loop(block_size);

        store_bias();
    }

    /* Pass filter address, then offset for h_padding. */
    compute_zero_filter();
    mov(reg_kh_offset,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, filter_pad_off)]);
    add(reg_filter_baddr, reg_kh_offset);

    /* compute left padded block */
    if (l_pad && do_unroll_w) {
        compute_h_loop(unroll_w, l_pad, 0, 0);
        add(reg_output_baddr, unroll_w * ch_offset * sizeof(float));
        add(reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float));
        unroll_w_trips--;
        pad_offset = l_pad;
        l_pad = 0;
    }

    /* compute middle block */
    Label ow_blk_label;

    /* Insert loop for 'ow' block when middle block needs to execute more
     * than once */
    bool do_ow_blk_loop = unroll_w_trips > 1;
    if (do_ow_blk_loop) {
        mov(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
    }
    if (unroll_w_trips > 0) {
        compute_h_loop(unroll_w, l_pad, pad_offset, 0);
        add(reg_output_baddr, unroll_w * ch_offset * sizeof(float));
        add(reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float));
    }
    if (do_ow_blk_loop) {
        dec(reg_iter_ow_blk);
        cmp(reg_iter_ow_blk, 0);
        jg(ow_blk_label, T_NEAR);
    }

    /* compute right padded block */
    if (unroll_w_tail) {
        compute_h_loop(
                unroll_w_tail, l_pad, pad_offset, jcp.ow - unroll_w_tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::generate() {
    preamble();

    mov(reg_input_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, input)]);
    mov(reg_output_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, output)]);
    mov(reg_filter_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, filter)]);

    compute_ow_block_unroll();

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_weights_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_weights_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
