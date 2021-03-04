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

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_amx_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::utils;
using namespace Xbyak;

void jit_avx512_core_amx_copy_to_wbuffer_t::generate() {

    const bool is_bf16 = jcp.src_dt == data_type::bf16;

    // required for use of VPERMB instruction
    assert(IMPLICATION(!is_bf16, cpu().has(Xbyak::util::Cpu::tAVX512_VBMI)));
    assert(jcp.ic_block_int * jcp.typesize_in == 64);

    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);

    // load permute indices from data section
    Label permute_index_table;
    mov(reg_tmp, permute_index_table);
    if (is_bf16)
        vmovdqu16(zmm_idx, ptr[reg_tmp]);
    else
        vmovdqu8(zmm_idx, ptr[reg_tmp]);

    const int vnni_width = is_bf16 ? 2 : 4;
    const int r = jcp.kh * jcp.kw * jcp.ic_without_padding;
    const int nb_r = div_up(r, vnni_width);
    const int rtail = (r % vnni_width) * jcp.oc_block;
    if (rtail > 0) {
        uint64_t mask = (UINT64_C(1) << rtail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask_load, reg_tmp);
    }
    const int nb_z = rnd_up(nb_r, jcp.ic_block);
    if (nb_r < nb_z) vpxord(zmm_zero, zmm_zero, zmm_zero);

    const int tile_size = jcp.ic_block_int * jcp.oc_block * jcp.typesize_in;
    const int ocb_src_step = r * jcp.oc_block * jcp.typesize_in;
    const int ocb_dst_step = rnd_up(ocb_src_step, tile_size);

    // reorder from ~Owhi16o -> ~OR16oVr with r := whi and V := vnni_width
    for (int g = 0; g < jcp.ngroups; g++) {
        for (int ocb = 0; ocb < jcp.nb_oc; ocb++) {
            int offset = 0;
            int rb = 0;
            for (; rb < nb_r; offset += 64, rb++) {
                auto zmm_src_tmp = (rtail > 0 && rb == nb_r - 1)
                        ? zmm_src | kmask_load | T_z
                        : zmm_src;
                if (is_bf16) {
                    vmovdqu16(zmm_src_tmp, ptr[reg_src + offset]);
                    vpermw(zmm_dst, zmm_idx, zmm_src);
                    vmovdqu16(ptr[reg_dst + offset], zmm_dst);
                } else {
                    vmovdqu8(zmm_src_tmp, ptr[reg_src + offset]);
                    vpermb(zmm_dst, zmm_idx, zmm_src);
                    vmovdqu8(ptr[reg_dst + offset], zmm_dst);
                }
            }
            for (; rb < nb_z; offset += 64, rb++) {
                if (is_bf16)
                    vmovdqu16(ptr[reg_dst + offset], zmm_zero);
                else
                    vmovdqu8(ptr[reg_dst + offset], zmm_zero);
            }
            add(reg_src, ocb_src_step);
            add(reg_dst, ocb_dst_step);
        }
    }

    postamble();

    align(64);
    L(permute_index_table);
    const uint8_t no = 16; // 16o
    const uint8_t nr = is_bf16 ? 2 : 4; // 2r or 4r
    for (uint8_t o = 0; o < no; ++o) {
        for (uint8_t r = 0; r < nr; r++) {
            const uint8_t index = o + r * UINT8_C(no);
            if (is_bf16)
                dw(index);
            else
                db(index);
        }
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row_body(
        int lpad, int iw_len, int icb) {

    const bool is_bf16 = jcp.src_dt == data_type::bf16;
    int iwp_idx = 0;
    // there are min(gen_kw, jcp.stride_w) continuous sets of input
    // data (for each stride idx), they are placed one by one
    // without additional padding
    const bool are_sets_interleaved
            = IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1);
    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
    const int num_sets = are_sets_interleaved ? jcp.n_stride_sets : jcp.kw;
    for (int set_idx = 0; set_idx < num_sets; set_idx++) {
        int set_width_padded = !jcp.is_pbuffer_strided
                ? (jcp.ow_block - 1) * jcp.stride_w + gen_kw
                : are_sets_interleaved ? jcp.ow_block - 1 + gen_kw / num_sets
                                + (set_idx < gen_kw % num_sets ? 1 : 0)
                                       : jcp.ow_block;
        for (int set_shift = 0; set_shift < set_width_padded;
                set_shift++, iwp_idx++) {
            int iw_idx = set_idx * (jcp.dilate_w + 1)
                    + set_shift * (jcp.is_pbuffer_strided ? jcp.stride_w : 1)
                    - lpad;
            size_t out_base_offset
                    = (size_t)jcp.typesize_in * iwp_idx * jcp.ic_block_int_np;
            if (iw_idx < 0 || iw_idx >= iw_len) {
                // left or right padding
                vmovups(ptr[aux_out_ptr + out_base_offset], zmm_zero);
            } else if (jcp.is_nspc) {
                size_t inp_w_offset = (size_t)jcp.typesize_in * iw_idx
                        * jcp.ngroups * jcp.ic_without_padding;
                int ic = icb * jcp.ic_block_int_np;
                // TODO: use Xmm or Ymm moves for better small ic efficiency
                auto zmm_tmp_mask
                        = ic + jcp.ic_block_int <= jcp.ic_without_padding
                        ? zmm_tmp
                        : zmm_tmp | ktail_mask | T_z;
                if (is_bf16) {
                    vmovdqu16(zmm_tmp_mask, ptr[aux_inp_ptr + inp_w_offset]);
                    vmovdqu16(ptr[aux_out_ptr + out_base_offset], zmm_tmp);
                } else {
                    vmovdqu8(zmm_tmp_mask, ptr[aux_inp_ptr + inp_w_offset]);
                    vmovdqu8(ptr[aux_out_ptr + out_base_offset], zmm_tmp);
                }
            } else {
                assert(is_bf16);
                size_t inp_w_offset
                        = (size_t)jcp.typesize_in * iw_idx * jcp.ic_block;
                for (int j = 0; j < jcp.ic_block_int_np / jcp.ic_block; j++) {
                    int ic = icb * jcp.ic_block_int_np + j * jcp.ic_block;
                    size_t inp_c_w_offset = (size_t)jcp.typesize_in * j * jcp.ih
                                    * jcp.iw * jcp.ic_block
                            + inp_w_offset;
                    if (ic + jcp.ic_block <= jcp.ic) {
                        vmovdqu16(ymm_tmp, ptr[aux_inp_ptr + inp_c_w_offset]);
                    } else {
                        vpxord(ymm_tmp, ymm_tmp, ymm_tmp);
                    }
                    size_t out_offset = out_base_offset
                            + (size_t)jcp.typesize_in * j * jcp.ic_block;
                    vmovdqu16(ptr[aux_out_ptr + out_offset], ymm_tmp);
                }
            }
        }
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row(int icb) {
    if (jcp.nb_ow == 1) {
        copy_row_body(jcp.l_pad, jcp.iw, icb);
    } else {
        auto get_iw_len_required = [&](int cur_ow_block, int cur_lpad) {
            return (cur_ow_block - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1) + 1 - cur_lpad;
        };

        auto get_iw_len_limited = [&](int owb, int cur_ow_block, int cur_lpad) {
            auto len_req = get_iw_len_required(cur_ow_block, cur_lpad);
            if (owb < 0) return len_req;
            int ow_block_start = nstl::max(
                    0, owb * jcp.ow_block * jcp.stride_w - jcp.l_pad);
            return nstl::min(jcp.iw - ow_block_start, len_req);
        };

        int general_owb_cases = jcp.nb_ow;
        Xbyak::Label copy_row_done_label;
        bool special_first_block_case = jcp.l_pad > 0;
        if (special_first_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_first_block_case_label;
            cmp(reg_owb, 0);
            jne(skip_first_block_case_label, T_NEAR);
            copy_row_body(jcp.l_pad,
                    get_iw_len_limited(0, jcp.ow_block, jcp.l_pad), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_first_block_case_label);
        }
        bool special_last_block_case = false
                // has ow_block_tail
                || jcp.ow % jcp.ow_block != 0
                // there is no ow_block_tail but right padding exists
                || get_iw_len_limited(jcp.nb_ow - 1, jcp.ow_block, 0)
                        != get_iw_len_required(jcp.ow_block, 0);
        if (special_last_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_last_block_case_label;
            cmp(reg_owb, jcp.nb_ow - 1);
            jne(skip_last_block_case_label, T_NEAR);
            int ow_block_tail = jcp.ow % jcp.ow_block;
            int cur_ow_block = ow_block_tail > 0 ? ow_block_tail : jcp.ow_block;
            copy_row_body(
                    0, get_iw_len_limited(jcp.nb_ow - 1, cur_ow_block, 0), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_last_block_case_label);
        }

        bool special_penult_block_case = true
                // if nb_ow = 2 and l_pad > 0 it's the same as
                // special_first_block_case
                && jcp.nb_ow >= (special_first_block_case ? 3 : 2)
                // right padding exists in penult block
                && get_iw_len_limited(jcp.nb_ow - 2, jcp.ow_block, 0)
                        != get_iw_len_required(jcp.ow_block, 0);
        if (special_penult_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_penult_block_case_label;
            cmp(reg_owb, jcp.nb_ow - 2);
            jne(skip_penult_block_case_label, T_NEAR);
            copy_row_body(
                    0, get_iw_len_limited(jcp.nb_ow - 2, jcp.ow_block, 0), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_penult_block_case_label);
        }

        if (general_owb_cases > 0) // general case
            copy_row_body(0, get_iw_len_required(jcp.ow_block, 0), icb);

        L(copy_row_done_label);
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row_reduced_lowering() {
    assert(jcp.nb_ic_int == 1);
    assert(jcp.ic_block_int * jcp.typesize_in == 64);
    assert(jcp.is_nspc);

    auto load_mask = [=](int tail, Opmask kmask) {
        uint64_t mask = (UINT64_C(1) << tail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask, reg_tmp);
    };

    const bool is_bf16 = jcp.src_dt == data_type::bf16;
    const int inp_w_step
            = jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    const int inp_h_step = jcp.iw * inp_w_step;
    const int out_h_step = jcp.ic_without_padding * jcp.typesize_in;
    const int out_w_step = jcp.kh * out_h_step;
    const int tail_size = jcp.ic_without_padding % jcp.ic_block_int;
    if (tail_size > 0) load_mask(tail_size, ktail_mask);

    auto zero_it = [=](reg64_t tmp_out_ptr) {
        for (int ic = 0; ic < jcp.ic_without_padding; ic += jcp.ic_block_int) {
            const int offset = ic * jcp.typesize_in;
            const bool masked = ic + jcp.ic_block_int > jcp.ic_without_padding;
            Zmm zmm = masked ? zmm_zero | ktail_mask : zmm_zero;
            if (is_bf16)
                vmovdqu16(ptr[tmp_out_ptr + offset], zmm);
            else
                vmovdqu8(ptr[tmp_out_ptr + offset], zmm);
        }
    };

    // pointer to 1st needed element in src buffer
    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    // pointer to 1st needed element in dst buffer
    mov(out_ptr, ptr[param1 + GET_OFF(dst)]);

    // total number of rows to copy
    mov(reg_kht, ptr[param1 + GET_OFF(kh_offset)]);

    // number of rows of src buffer to copy
    mov(reg_khp, ptr[param1 + GET_OFF(kh_padding)]);
    // number of zero-padded rows above src buffer to copy
    mov(reg_tov, ptr[param1 + GET_OFF(t_overflow)]);
    // number of zero-padded rows below src buffer to copy
    mov(reg_bov, ptr[param1 + GET_OFF(b_overflow)]);

    // number of columns of src buffer to copy
    mov(reg_kwp, ptr[param1 + GET_OFF(kw_padding)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_lov, ptr[param1 + GET_OFF(f_overflow)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_rov, ptr[param1 + GET_OFF(back_overflow)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    { // Handle Left Overflow
        Label label_lov, label_lov_skip;
        test(reg_lov, reg_lov);
        jz(label_lov_skip, T_NEAR);
        L(label_lov); // handle left or right overflow
        {
            Label label_lov_inner;
            mov(aux_out_ptr, out_ptr);
            mov(reg_cnt, reg_kht);
            L(label_lov_inner);
            {
                zero_it(aux_out_ptr);
                add(aux_out_ptr, out_h_step);
                dec(reg_cnt);
                jnz(label_lov_inner, T_NEAR);
            }
            add(out_ptr, out_w_step);
            dec(reg_lov);
            jnz(label_lov, T_NEAR);
        }
        L(label_lov_skip);
    }

    // save output pointer for later use
    mov(save_out_ptr, out_ptr);

    // just in case there is no meat...
    Label label_kwp_end;
    test(reg_kwp, reg_kwp);
    jz(label_kwp_end, T_NEAR);

    // Unroll over W-dimension in powers of 2
    Label label_tov;
    Label label_khp, label_no_khp;
    Label label_bov;
    test(reg_tov, reg_tov);
    jnz(label_tov, T_NEAR);
    test(reg_khp, reg_khp);
    jnz(label_khp, T_NEAR);
    test(reg_bov, reg_bov);
    jnz(label_bov, T_NEAR);
    jmp(label_kwp_end, T_NEAR); // safe exit in case of bad parameters

    L(label_tov); // handle top overflow
    {
        Label label_tov_inner;
        mov(aux_out_ptr, out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_tov_inner);
        {
            zero_it(aux_out_ptr);
            add(aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_tov_inner, T_NEAR);
        }
        add(out_ptr, out_h_step);
        dec(reg_tov);
        jnz(label_tov, T_NEAR);
    }
    test(reg_khp, reg_khp);
    jz(label_no_khp, T_NEAR);
    L(label_khp); // handle kh padding (not fully unrolled)
    {
        Label label_khp_inner;
        mov(aux_inp_ptr, inp_ptr);
        mov(aux_out_ptr, out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_khp_inner);
        {
            for (int ic = 0; ic < jcp.ic_without_padding;
                    ic += jcp.ic_block_int) {
                const int offset = ic * jcp.typesize_in;
                const bool masked
                        = ic + jcp.ic_block_int > jcp.ic_without_padding;
                // zero masking is needed to avoid dependency on destination
                Zmm zmm_load = masked ? zmm_tmp | ktail_mask | T_z : zmm_tmp;
                Zmm zmm_store = masked ? zmm_tmp | ktail_mask : zmm_tmp;
                if (is_bf16) {
                    vmovdqu16(zmm_load, ptr[aux_inp_ptr + offset]);
                    vmovdqu16(ptr[aux_out_ptr + offset], zmm_store);
                } else {
                    vmovdqu8(zmm_load, ptr[aux_inp_ptr + offset]);
                    vmovdqu8(ptr[aux_out_ptr + offset], zmm_store);
                }
            }
            add(aux_inp_ptr, inp_w_step);
            add(aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_khp_inner, T_NEAR);
        }
        add(inp_ptr, inp_h_step);
        add(out_ptr, out_h_step);
        dec(reg_khp);
        jnz(label_khp, T_NEAR);
    }
    L(label_no_khp);
    test(reg_bov, reg_bov);
    jz(label_kwp_end, T_NEAR);
    L(label_bov); // handle bottom overflow
    {
        Label label_bov_inner;
        mov(aux_out_ptr, out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_bov_inner);
        {
            zero_it(aux_out_ptr);
            add(aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_bov_inner, T_NEAR);
        }
        add(out_ptr, out_h_step);
        dec(reg_bov);
        jnz(label_bov, T_NEAR);
    }
    L(label_kwp_end);

    { // Handle Right Overflow
        Label label_rov, label_rov_skip;
        // retrieve output pointer
        mov(out_ptr, save_out_ptr);
        // calculate the shift
        imul(reg_tmp, reg_kwp, out_w_step);
        // shift past the body
        add(out_ptr, reg_tmp);
        // skip if no right overflow
        test(reg_rov, reg_rov);
        jz(label_rov_skip, T_NEAR);

        L(label_rov); // handle left or right overflow
        {
            Label label_rov_inner;
            mov(aux_out_ptr, out_ptr);
            mov(reg_cnt, reg_kht);
            L(label_rov_inner);
            {
                zero_it(aux_out_ptr);
                add(aux_out_ptr, out_h_step);
                dec(reg_cnt);
                jnz(label_rov_inner, T_NEAR);
            }
            add(out_ptr, out_w_step);
            dec(reg_rov);
            jnz(label_rov, T_NEAR);
        }
        L(label_rov_skip);
    }

    // For bf16, zero-pad an extra cacheline to avoid NaNs
    // For int8, it is sufficient to zero-pad the weights only
    if (is_bf16) {
        // shift forward to align h index to end of needed buffer
        imul(reg_tmp, reg_kht, out_h_step);
        add(out_ptr, reg_tmp);
        // shift backward to align w index to end of needed buffer
        sub(out_ptr, out_w_step);
        vmovdqu16(ptr[out_ptr], zmm_zero);
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::generate() {

    // Special copy kernel for reduced lowering
    if (jcp.is_relo) {
        assert(jcp.nb_ic_int == 1);
        preamble();
        copy_row_reduced_lowering();
        postamble();
        return;
    }

    preamble();

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(out_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(khp, ptr[param1 + GET_OFF(kh_padding)]);
    mov(tover, ptr[param1 + GET_OFF(t_overflow)]);
    mov(bover, ptr[param1 + GET_OFF(b_overflow)]);
    mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    if (jcp.is_nspc && jcp.ic_without_padding % jcp.ic_block_int) {
        int tail_size = jcp.ic_without_padding % jcp.ic_block_int;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    for (int icb = 0; icb < jcp.nb_ic_int; icb++) {
        Xbyak::Label kh_label, no_kh_label, icb_label;
        Xbyak::Label kh_tover_label, kh_bover_label;
        Xbyak::Label no_kh_tover_label, no_kh_bover_label;

        mov(aux_inp_ptr, inp_ptr);
        mov(aux_out_ptr, out_ptr);

        cmp(khp, 0);
        jle(no_kh_bover_label, T_NEAR); // nothing to do
        mov(khc, khp);

        cmp(tover, 0);
        jle(no_kh_tover_label, T_NEAR);

        mov(kh_over, tover);
        L(kh_tover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (int iw = 0; iw < jcp.iwp; iw++)
                vmovups(ptr[aux_out_ptr
                                + jcp.typesize_in * iw * jcp.ic_block_int_np],
                        zmm_zero);
            int out_h_offset = jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;
            add(aux_out_ptr, out_h_offset);

            dec(kh_over);
            jnz(kh_tover_label, T_NEAR);
        }
        sub(khc, tover);
        L(no_kh_tover_label);

        cmp(khc, bover);
        jle(no_kh_label, T_NEAR);

        L(kh_label);
        {
            copy_row(icb);
            size_t inp_h_offset = !jcp.is_nspc
                    ? (size_t)jcp.typesize_in * jcp.iw * jcp.ic_block
                    : (size_t)jcp.typesize_in * jcp.iw * jcp.ngroups
                            * jcp.ic_without_padding;
            size_t out_h_offset
                    = (size_t)jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;

            add(aux_inp_ptr, inp_h_offset);
            add(aux_out_ptr, out_h_offset);

            dec(khc);
            cmp(khc, bover);
            jg(kh_label, T_NEAR);
        }
        L(no_kh_label);

        cmp(khc, 0);
        jle(no_kh_bover_label, T_NEAR);

        L(kh_bover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (int iw = 0; iw < jcp.iwp; iw++)
                vmovups(ptr[aux_out_ptr
                                + jcp.typesize_in * iw * jcp.ic_block_int_np],
                        zmm_zero);
            int out_h_offset = jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;
            add(aux_out_ptr, out_h_offset);

            dec(khc);
            jnz(kh_bover_label, T_NEAR);
        }
        L(no_kh_bover_label);

        // End IC Loop
        size_t inp_cb_offset = !jcp.is_nspc
                ? (size_t)jcp.typesize_in * (jcp.ic_block_int_np / jcp.ic_block)
                        * jcp.ih * jcp.iw * jcp.ic_block
                : (size_t)jcp.typesize_in * jcp.ic_block_int_np;
        size_t out_cb_offset = (size_t)jcp.typesize_in * jcp.ihp * jcp.iwp
                * jcp.ic_block_int_np;

        add(inp_ptr, inp_cb_offset);
        add(out_ptr, out_cb_offset);
    }

    postamble();
}

// Tile register decomposition
// { C_BASE = 0, I_BASE = 4, W_BASE = 6, }
int jit_avx512_core_amx_fwd_kernel_t::get_out_tensor(
        int h, int i, bool is_h_tail) const {
    const int C_BASE = 0;
    const int C_LAST = 4;
    assert(0 <= C_BASE && C_BASE < C_LAST && C_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(C_LAST);
    const int tile = C_BASE
            + (jcp.nb_oh_blocking > 1
                            ? h * jcp.nb_oh_blocking + i
                            : (int)is_h_tail * jcp.nb_oc_blocking + i);
    assert(C_BASE <= tile && tile < C_LAST);
    return tile;
}
int jit_avx512_core_amx_fwd_kernel_t::get_inp_tensor(
        int h, bool is_h_tail) const {
    const int I_BASE = 4;
    const int I_LAST = 6;
    assert(0 <= I_BASE && I_BASE < I_LAST && I_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(I_LAST);
    const int tile = I_BASE + (jcp.nb_oh_blocking > 1 ? h : (int)is_h_tail);
    assert(I_BASE <= tile && tile < I_LAST);
    return tile;
}
int jit_avx512_core_amx_fwd_kernel_t::get_wei_tensor(int i) const {
    const int W_BASE = 6;
    const int W_LAST = 8;
    assert(0 <= W_BASE && W_BASE < W_LAST && W_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(W_LAST);
    const int tile = W_BASE + i;
    assert(W_BASE <= tile && tile < W_LAST);
    return tile;
}

// Shifts and offsets
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_icb_step() const {
    return (size_t)jcp.typesize_in * jcp.ihp * jcp.iwp * jcp.ic_block_int_np;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_icb_step() const {
    return (size_t)jcp.typesize_in * jcp.kh * jcp.kw * jcp.ic_block_int_np
            * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_h_step() const {
    return (size_t)jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np
            * (jcp.dilate_h + 1);
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_h_step() const {
    return (size_t)jcp.typesize_in * jcp.kw * jcp.ic_block_int_np
            * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_ocb_offset(
        int ohb, int ocb) const {
    size_t el_offset = jcp.is_nspc
            ? (size_t)ocb * jcp.oc_block
                    + (size_t)ohb * jcp.ow * jcp.ngroups
                            * jcp.oc_without_padding
            : (size_t)ocb * jcp.oh * jcp.ow * jcp.oc_block
                    + (size_t)ohb * jcp.ow * jcp.oc_block;
    return (size_t)jcp.typesize_out * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_row_offset(
        int ohb, int ocb, int j) const {
    size_t offset_w = jcp.is_nspc ? (size_t)jcp.typesize_out * j * jcp.ngroups
                    * jcp.oc_without_padding
                                  : (size_t)jcp.typesize_out * j * jcp.oc_block;
    return get_out_ocb_offset(ohb, ocb) + offset_w;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_shift(int width) const {
    return jcp.is_nspc ? (size_t)jcp.typesize_out * width * jcp.ngroups
                    * jcp.oc_without_padding
                       : (size_t)jcp.typesize_out * width * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_ocb_offset(
        int ohb, int ocb) const {
    size_t el_offset = (size_t)ocb * prv_width_ * jcp.oc_block
            + (size_t)ohb * jcp.nb_oc_blocking * jcp.full_tile_width
                    * jcp.oc_block;
    return jcp.typesize_acc * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_row_offset(
        int ohb, int ocb, int j) const {
    return get_wsp_ocb_offset(ohb, ocb)
            + (size_t)jcp.typesize_acc * j * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_shift() const {
    return (size_t)jcp.typesize_acc * jcp.nb_oh_blocking * jcp.full_tile_width
            * jcp.oc_block * jcp.nb_oc_blocking;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_offset(int ocb, int kw) const {
    size_t el_offset = (size_t)kw * jcp.ic_block_int_np * jcp.oc_block;
    size_t raw_oc_subblock_step
            = jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;
    size_t oc_subblock_step = jcp.is_relo
            ? rnd_up(raw_oc_subblock_step, jcp.ic_block_int * jcp.oc_block)
            : raw_oc_subblock_step;
    el_offset += (size_t)ocb * jcp.nb_ic_int * oc_subblock_step;
    return jcp.typesize_in * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_shift() const {
    size_t w_step = (jcp.is_relo ? jcp.stride_w * jcp.kh
                                 : jcp.is_pbuffer_strided ? 1 : jcp.stride_w)
            * jcp.ic_block_int_np;
    return (size_t)jcp.typesize_in * jcp.tile_width * w_step;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_offset(int ohb, int kw) const {
    if (jcp.is_relo)
        return ohb * jcp.iwp * jcp.kh * jcp.ic_block_int_np * jcp.typesize_in;
    // calculate offset by height dimension
    const int gen_stride_h = nstl::min(jcp.stride_h, jcp.kh);
    size_t el_offset = (size_t)ohb * jcp.oh_per_tile * gen_stride_h * jcp.iwp
            * jcp.ic_block_int_np;

    // add offset by width dimension
    if (IMPLICATION(jcp.is_pbuffer_strided, jcp.stride_w == 1)) {
        el_offset += (size_t)kw * (jcp.dilate_w + 1) * jcp.ic_block_int_np;
    } else if (jcp.dilate_w > 0) {
        el_offset += (size_t)kw * jcp.ow_block * jcp.ic_block_int_np;
    } else {
        // dilate_w == 0 && stride_w > 1
        // there are min(jcp.kw, jcp.stride_w) continuous sets of input data
        // (foreach stride idx), they are placed one by one without additional
        // padding

        // calculate set idx for current kw value
        int set_idx = kw % jcp.stride_w;
        // calculate shift within set for current kw value
        int set_shift = kw / jcp.stride_w;

        // calculate the beginning of the current set along width, each set
        // with index set_i contains number of elements along width equal to
        // jcp.ow - 1 + jcp.kw / jcp.stride_w
        //     + (set_i < jcp.kw % jcp.stride_w)
        size_t set_start = (jcp.ow_block - 1 + jcp.kw / jcp.stride_w) * set_idx
                + nstl::min(set_idx, jcp.kw % jcp.stride_w);
        el_offset += (set_start + set_shift) * jcp.ic_block_int_np;
    }
    return jcp.typesize_in * el_offset;
}

// Code generation
void jit_avx512_core_amx_fwd_kernel_t::prepare_output(int tail) {
    for (int h = 0; h < jcp.nb_oh_blocking; h++)
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tilezero(Tmm(get_out_tensor(h, i, tail)));
}

void jit_avx512_core_amx_fwd_kernel_t::init_runtime_counters(
        bool start_with_last_tile_block) {
    prv_width_ = start_with_last_tile_block && jcp.tile_tail > 0
            ? jcp.tile_tail
            : jcp.tile_width;

    row_count_ = 0;
    is_store_done_ = false;
    is_buffer_empty_ = true;
}

bool jit_avx512_core_amx_fwd_kernel_t::maybe_eltwise(int position) {
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

Ymm jit_avx512_core_amx_fwd_kernel_t::ymm_mask(
        const Ymm ymm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

Zmm jit_avx512_core_amx_fwd_kernel_t::zmm_mask(
        const Zmm zmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

void jit_avx512_core_amx_fwd_kernel_t::cvt2ps(data_type_t type_in,
        const Zmm zmm_in, const Operand &op, bool mask_flag = false) {
    const Zmm zmm = zmm_mask(zmm_in, mask_flag);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector_bf16(
        Zmm zmm_out, int ocb, int h, int w) {
    const bool mask_flag = jcp.is_nspc && jcp.oc_without_padding != jcp.oc
            && ocb == (jcp.nb_oc_blocking - 1);

    auto addr = EVEX_compress_addr(out_ptr, get_out_row_offset(h, ocb, w));

    const auto &p = attr_.post_ops_;

    const int sum_idx = p.find(primitive_kind::sum);
    if (sum_idx != -1) {
        if (jcp.dst_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vpslld(zmm_prev_dst, zmm_prev_dst, 16);
            vaddps(zmm_out, zmm_prev_dst);
        } else {
            vmovups(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vaddps(zmm_out, zmm_prev_dst);
        }
    }
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        if (jcp.bia_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_bias, mask_flag), bias_addr);
            vpslld(zmm_bias, zmm_bias, 16);
            vaddps(zmm_out, zmm_bias);
        } else
            vaddps(zmm_mask(zmm_out, mask_flag), bias_addr);
    }

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    if (eltwise_ind != -1) eltwise_injector_->compute_vector(zmm_out.getIdx());

    if (jcp.dst_dt == data_type::bf16) {
        Ymm ymm_out = Ymm(zmm_out.getIdx());
        vcvtneps2bf16(ymm_out, zmm_out);
        vmovdqu16(addr, ymm_mask(ymm_out, mask_flag, true));
    } else {
        vmovups(addr, zmm_mask(zmm_out, mask_flag, true));
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector_int8(
        Zmm zmm_out, int ocb, int h, int w) {
    const int nb_oc_block = jcp.nb_oc_blocking;
    const int oc_block = jcp.oc_block;
    const bool mask_flag = true && jcp.oc_without_padding != jcp.oc
            && ocb == (nb_oc_block - 1);

    auto addr = EVEX_compress_addr(out_ptr, get_out_row_offset(h, ocb, w));

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    int scale_offset = jcp.is_oc_scale * (sizeof(float) * ocb * oc_block);
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
    }
    /* add bias to zmm_accum */
    vcvtdq2ps(zmm_out, zmm_out);
    if (jcp.with_bias) vaddps(zmm_out, zmm_out, zmm_bias);
    const Zmm zmm_out_msk = zmm_mask(zmm_out, mask_flag);
    vmulps(zmm_out_msk, zmm_out,
            EVEX_compress_addr(reg_ptr_scales, scale_offset));

    /* Do post-ops */
    if (maybe_eltwise(0)) eltwise_injector_->compute_vector(zmm_out.getIdx());
    if (p_sum_scale) { // post_op: sum
        cvt2ps(jcp.dst_dt, zmm_prev_dst, addr, mask_flag);
        if (*p_sum_scale == 1.f)
            vaddps(zmm_out, zmm_prev_dst);
        else
            vfmadd231ps(zmm_out, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
    }
    if (maybe_eltwise(1)) eltwise_injector_->compute_vector(zmm_out.getIdx());

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, u8, s8, s32)) {
        init_saturate_f32(
                zmm_zero, zmm_saturation, aux_reg_saturation, f32, jcp.dst_dt);
        saturate_f32(zmm_out, zmm_zero, zmm_saturation, jcp.dst_dt);
        vcvtps2dq(zmm_out, zmm_out);
    }

    const Zmm zmm_out_store = zmm_mask(zmm_out, mask_flag, true);

    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32: vmovups(addr, zmm_out_store); break;
        case data_type::s8: vpmovsdb(addr, zmm_out_store); break;
        case data_type::u8: vpmovusdb(addr, zmm_out_store); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector(
        Zmm zmm_out, int ocb, int h, int w) {
    /*
    Output:
              jcp.is_nspc              !jcp.is_nspc
              ---------------------    ---------------------
        INT8: [N][H][W][NBOC][16OC]
        BF16: [N][H][W][NBOC][16OC] or [N][NBOC][H][W][16OC]
    */
    if (jcp.src_dt == data_type::bf16) {
        store_output_vector_bf16(zmm_out, ocb, h, w);
    } else {
        store_output_vector_int8(zmm_out, ocb, h, w);
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output(
        int width, int tail, bool do_store) {
    auto store_output_block = [=](int width, int tail, bool do_store,
                                      bool is_last_h = false) {
        // Calculate the number of oh blocks; it may differ on last call
        const int last_h_blks
                = div_up(jcp.oh, jcp.oh_per_tile) % jcp.nb_oh_blocking;
        const int h_blks = is_last_h && last_h_blks != 0 ? last_h_blks
                                                         : jcp.nb_oh_blocking;
        // Calculate the number of oh rows per tile; it may differ on last call
        const int h_tail = is_last_h && jcp.oh % jcp.oh_per_tile != 0
                ? (h_blks - 1) * jcp.oh_per_tile + jcp.oh % jcp.oh_per_tile
                : h_blks * jcp.oh_per_tile;
        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ohb = 0; ohb < h_blks; ohb++) {
                /* Formats: Workspace: [NBOC][W][16OC] */
                tilestored(ptr[wsp_ptr + reg_wei_stride
                                   + get_wsp_ocb_offset(ohb, ocb)],
                        Tmm(get_out_tensor(ohb, ocb, tail)));
                is_buffer_empty_ = false;
                is_store_done_ = false;
                for (int tw = 0; tw < width && do_store; tw++) {
                    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
                    const int owp = gen_kw + jcp.ow - 1;
                    const int oh_index = ohb * jcp.oh_per_tile + tw / owp;
                    const int ow_index = tw % owp;
                    assert(IMPLICATION(jcp.oh_per_tile == 1,
                            ohb == oh_index && tw == ow_index));
                    if (oh_index < h_tail && ow_index < jcp.ow) {
                        Zmm zmm_out = Zmm(tw);
                        vmovups(zmm_out,
                                ptr[wsp_ptr
                                        + get_wsp_row_offset(ohb, ocb, tw)]);
                        store_output_vector(zmm_out, ocb, oh_index, ow_index);
                    }
                }
            }
        }
    };

    // adjustment in case interleave store is turned off
    do_store = do_store || jcp.per_one_pstore == 0;
    if (jcp.oh % (jcp.oh_per_tile * jcp.nb_oh_blocking) == 0) {
        store_output_block(width, tail, do_store);
    } else {
        Label label_oh_oc_store, label_done;
        cmp(reg_last_h, 0);
        jne(label_oh_oc_store, T_NEAR);
        store_output_block(width, tail, do_store, true); // last h
        jmp(label_done, T_NEAR);
        L(label_oh_oc_store);
        store_output_block(width, tail, do_store, false);
        L(label_done);
    }
    if (do_store) add(out_ptr, get_out_shift(width));
}

void jit_avx512_core_amx_fwd_kernel_t::interleave_store(int width) {
    for (int c = 0;
            c < jcp.per_one_pstore && !is_store_done_ && !is_buffer_empty_;
            c++) {
        // row_count = ohb * OCB * TW + ocb * TW + tw
        int tw = row_count_ % prv_width_;
        int ocb = (row_count_ / prv_width_) % jcp.nb_oc_blocking;
        int ohb = (row_count_ / prv_width_) / jcp.nb_oc_blocking;

        Zmm zmm_out = Zmm(tw);
        vmovups(zmm_out, ptr[wsp_ptr + get_wsp_row_offset(ohb, ocb, tw)]);
        store_output_vector(zmm_out, ocb, ohb, tw);
        row_count_++;

        if (row_count_
                == prv_width_ * jcp.nb_oc_blocking * jcp.nb_oh_blocking) {
            add(out_ptr, get_out_shift(prv_width_));
            row_count_ = 0;
            is_store_done_ = true;
            prv_width_ = width;
        }
    }
}

void jit_avx512_core_amx_fwd_kernel_t::compute_icb_loop(
        int width, bool do_store) {
    const bool tail = width == jcp.tile_tail;

    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (jcp.src_dt == data_type::bf16 && jcp.wei_dt == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::u8) {
            tdpbuud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::s8) {
            tdpbusd(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::u8) {
            tdpbsud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };

    prepare_output(tail);

    // reduced lowering path
    if (jcp.is_relo) {
        const int nreduce = jcp.nreduce;
        const int stride = jcp.ic_block_int; // ie 64 (32) for int8 (bf16)
        mov(aux_inp_ptr, inp_ptr);
        mov(aux_wei_ptr, wei_ptr);
        for (int ireduce = 0; ireduce < nreduce; ireduce += stride) {
            for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                tileloadd(Tmm(get_inp_tensor(ohb, tail)),
                        ptr[aux_inp_ptr + get_inp_offset(ohb, 0)
                                + reg_inp_stride]);
            }
            for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                tileloadd(Tmm(get_wei_tensor(ocb)),
                        ptr[aux_wei_ptr + get_wei_offset(ocb, 0)
                                + reg_wei_stride]);
                for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                    tdpbxxd(Tmm(get_out_tensor(ohb, ocb, tail)),
                            Tmm(get_inp_tensor(ohb, tail)),
                            Tmm(get_wei_tensor(ocb)));
                    interleave_store(width);
                }
            }
            if (ireduce + stride < nreduce) {
                add(aux_inp_ptr, stride * jcp.typesize_in);
                add(aux_wei_ptr, stride * jcp.oc_block * jcp.typesize_in);
            }
        }
        store_output(width, tail, do_store);

        add(inp_ptr, get_inp_shift());
        return;
    }

    // normal and k-remainders path
    for (int icb = 0; icb < jcp.nb_ic_int; icb++) {
        mov(aux_inp_ptr, inp_ptr);
        mov(aux_wei_ptr, wei_ptr);
        for (int kh = 0; kh < jcp.kh; kh++) {
            for (int set_idx = 0; set_idx < jcp.n_stride_sets;
                    set_idx++) { // used to optimize input memory reuse in L1$
                for (int kw = set_idx; kw < jcp.kw; kw += jcp.kw_step) {
                    for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                        tileloadd(Tmm(get_inp_tensor(ohb, tail)),
                                ptr[aux_inp_ptr + get_inp_offset(ohb, kw)
                                        + reg_inp_stride]);
                    }
                    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                        tileloadd(Tmm(get_wei_tensor(ocb)),
                                ptr[aux_wei_ptr + get_wei_offset(ocb, kw)
                                        + reg_wei_stride]);
                        for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                            tdpbxxd(Tmm(get_out_tensor(ohb, ocb, tail)),
                                    Tmm(get_inp_tensor(ohb, tail)),
                                    Tmm(get_wei_tensor(ocb)));
                            interleave_store(width);
                        }
                    }
                }
            }
            add(aux_inp_ptr, get_inp_h_step());
            add(aux_wei_ptr, get_wei_h_step());
        }
        add(inp_ptr, get_inp_icb_step());
        add(wei_ptr, get_wei_icb_step());
    }
    sub(inp_ptr, get_inp_icb_step() * jcp.nb_ic_int);
    sub(wei_ptr, get_wei_icb_step() * jcp.nb_ic_int);

    store_output(width, tail, do_store);

    add(inp_ptr, get_inp_shift());
}

void jit_avx512_core_amx_fwd_kernel_t::compute_ow_loop() {
    auto compute_ow_loop_body = [=](bool last_owb, int num_tile_blocks) {
        int gen_tile_tail = last_owb && jcp.tile_tail > 0 ? jcp.tile_tail
                                                          : jcp.tile_width;
        init_runtime_counters(last_owb && num_tile_blocks == 1);
        for (int owb = 0; owb < num_tile_blocks - 1; owb++)
            compute_icb_loop(jcp.tile_width, false);
        compute_icb_loop(gen_tile_tail, true);
    };

    if (jcp.nb_ow == 1) {
        compute_ow_loop_body(true, jcp.ow_blocks);
    } else {
        assert(jcp.oh_per_tile == 1);
        Label label_done;
        int ow_blocks_per_call = utils::div_up(jcp.ow_block, jcp.tile_width);
        int last_owb_tile_blocks = jcp.ow_blocks % ow_blocks_per_call;
        if (last_owb_tile_blocks == 0 && jcp.tile_tail > 0)
            last_owb_tile_blocks = ow_blocks_per_call;
        if (last_owb_tile_blocks > 0) {
            Label label_not_last_owb;
            mov(reg_tmp, ptr[param1 + GET_OFF(owb)]);
            cmp(reg_tmp, jcp.nb_ow - 1);
            jne(label_not_last_owb, T_NEAR);

            compute_ow_loop_body(true, last_owb_tile_blocks);

            jmp(label_done, T_NEAR);

            L(label_not_last_owb);
        }
        compute_ow_loop_body(false, ow_blocks_per_call);

        L(label_done);
    }
}

void jit_avx512_core_amx_fwd_kernel_t::generate() {
    preamble();

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(wei_ptr, ptr[param1 + GET_OFF(filt)]);
    mov(out_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(wsp_ptr, ptr[param1 + GET_OFF(acc_s32)]);

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    mov(reg_last_h, ptr[param1 + GET_OFF(last_h)]);

    const int fac = jcp.is_relo ? jcp.stride_w * jcp.kh
                                : jcp.is_pbuffer_strided ? 1 : jcp.stride_w;
    const int inp_stride = fac * jcp.ic_block_int_np * jcp.typesize_in;
    const int wei_stride = jcp.oc_block * jcp.typesize_acc;
    mov(reg_inp_stride, inp_stride);
    mov(reg_wei_stride, wei_stride);

    if (jcp.is_nspc && jcp.oc_without_padding != jcp.oc) {
        // Use mask 0xF by default for all output data and post-ops
        // loads / stores with block index
        // ocb = occ * jcp.nb_oc_blocking + (jcp.nb_oc_blocking - 1)
        // TODO: use masked loads / stores for the last occ only
        int current_block_size = jcp.oc_block;
        int mask = (1 << current_block_size) - 1;
        Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
        Xbyak::Label mask_is_set;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(mask_is_set, T_NEAR);
        // Reset the mask
        current_block_size = jcp.oc_without_padding % jcp.oc_block;
        mask = (1 << current_block_size) - 1;
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);

        L(mask_is_set);
    }
    compute_ow_loop();

    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx512_core_amx_fwd_kernel_t::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;
    const bool is_bf16 = jcp.src_dt == data_type::bf16;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    auto is_sum = [&](int idx) {
        if (is_bf16)
            return p.entry_[idx].is_sum();
        else
            return p.contain(sum, idx);
    };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || is_sum(0);
        case 2:
            return (is_sum(0) && is_eltwise(1))
                    || (!is_bf16 && is_sum(1) && is_eltwise(0));
        default: return false;
    }

    return false;
}

void jit_avx512_core_amx_fwd_kernel_t::tile_configure(char *tcfg_buff) {
    const int vnni_width = jcp.src_dt == data_type::bf16 ? 2 : 4;
    // Input tile dimensions
    const int a_col = jcp.is_relo ? jcp.ic_block_int
                                  : jcp.ic_block_int_np * jcp.kw_per_tile;
    // Weights tile dimensions
    const int b_col = jcp.oc_block * vnni_width;
    const int b_row = a_col / vnni_width;
    // Accumulator tile dimensions
    const int c_col = 16;

    for (size_t i = 0; i < 64; i++)
        tcfg_buff[i] = 0;

    // Weights (W_BASE) Tensor Tiles
    for (int i = 0; i < jcp.nb_oc_blocking; i++)
        tc_configure_tile((palette_config_t *)tcfg_buff, get_wei_tensor(i),
                b_row, b_col * jcp.typesize_in);

    // Input (I_BASE) and Accumulator (C_BASE) Tensor Tiles
    for (int h = 0; h < jcp.nb_oh_blocking; h++) {
        tc_configure_tile((palette_config_t *)tcfg_buff, get_inp_tensor(h),
                jcp.tile_width, a_col * jcp.typesize_in);
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_out_tensor(h, i), jcp.tile_width,
                    c_col * jcp.typesize_acc);
    }
    if (jcp.tile_tail != 0) {
        assert(jcp.nb_oh_blocking == 1);
        assert(jcp.oh_per_tile == 1);
        assert(jcp.ow > jcp.tile_width);
        tc_configure_tile((palette_config_t *)tcfg_buff,
                get_inp_tensor(0, true), jcp.tile_tail,
                a_col * jcp.typesize_in);
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_out_tensor(0, i, true), jcp.tile_tail,
                    c_col * jcp.typesize_acc);
    }

    ((palette_config_t *)tcfg_buff)->palette_id = amx::get_max_palette();
}

status_t jit_avx512_core_amx_fwd_kernel_t::init_conf(jit_conv_conf_t &jcp,
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
    bool is_1d = ndims == 3;
    bool is_3d = ndims == 5;

    if (is_3d) return status::unimplemented;

    const bool is_bf16_convolution
            = everyone_is(true, src_d.data_type() == data_type::bf16,
                    weights_d.data_type() == data_type::bf16,
                    one_of(dst_d.data_type(), data_type::bf16, data_type::f32));
    const bool is_int8_convolution = everyone_is(true,
            (src_d.data_type() == data_type::u8
                    || src_d.data_type() == data_type::s8),
            weights_d.data_type() == data_type::s8,
            one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8));

    bool supported = false
            || (is_bf16_convolution && mayiuse(avx512_core_bf16_amx_bf16))
            || (is_int8_convolution && mayiuse(avx512_core_bf16_amx_int8));
    if (!supported) return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.isa = is_bf16_convolution ? avx512_core_bf16_amx_bf16
                                  : avx512_core_bf16_amx_int8;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;

    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = !is_1d ? src_d.dims()[ndims - 2] : 1;
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = !is_1d ? dst_d.dims()[ndims - 2] : 1;
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kh = !is_1d ? weights_d.dims()[with_groups + ndims - 2] : 1;
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.t_pad = !is_1d ? cd.padding[0][ndims - 4] : 0;
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_h = !is_1d ? cd.strides[ndims - 4] : 1;
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.dilate_d = is_3d ? cd.dilates[ndims - 5] : 0;
    jcp.dilate_h = !is_1d ? cd.dilates[ndims - 4] : 0;
    jcp.dilate_w = cd.dilates[ndims - 3];

    if (jcp.dilate_d != 0) return status::unimplemented;
    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    if (jcp.l_pad >= gen_kw || jcp.r_pad >= gen_kw || jcp.t_pad >= gen_kh
            || jcp.b_pad >= gen_kh)
        return status::unimplemented;

    const int max_pad = 28; // akin to maximum jcp.ur_w value in other jits
    if (jcp.l_pad > max_pad || jcp.r_pad > max_pad)
        return status::unimplemented; // TODO: relax this restriction

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.src_dt = cd.src_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise)
        return status::unimplemented; // TODO: add support of DW convolution

    format_tag_t dat_tag_ncsp
            = utils::pick(ndims - 3, format_tag::nCw16c, format_tag::nChw16c);
    format_tag_t dat_tag_nspc
            = utils::pick(ndims - 3, format_tag::nwc, format_tag::nhwc);
    // To toggle the default data layout for BF16 between nChw16c and nhwc,
    // swap the following two variable definitions. Current choice: nhwc.

    // Clang-tidy change - if it was intentional please revert it and
    // put `NOLINTNEXTLINE` to suppress the warning.
    format_tag_t dat_tag_opt = dat_tag_nspc;
    format_tag_t dat_tag_alt
            = is_bf16_convolution ? dat_tag_ncsp : dat_tag_nspc;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag_opt));
        jcp.src_tag = dat_tag_opt;
    } else
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag_alt, dat_tag_opt);

    if (!one_of(jcp.src_tag, dat_tag_alt, dat_tag_opt))
        return status::unimplemented;

    jcp.is_nspc = jcp.src_tag == dat_tag_nspc;
    assert(IMPLICATION(is_int8_convolution, jcp.is_nspc));

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, jcp.src_tag));
        jcp.dst_tag = jcp.src_tag;
    } else
        jcp.dst_tag = dst_d.matches_one_of_tag(jcp.src_tag);

    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;

    if (jcp.with_bias && bias_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));

    jcp.nthr = nthreads;

    jcp.ic_block = 16;
    jcp.oc_block = 16;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0;
    if (!args_ok) return status::unimplemented;

    const int vnni_width = is_bf16_convolution ? 2 : 4;
    jcp.ic_block_int = jcp.ic_block * vnni_width; // 32 for bf16, 64 for int8

    // small-ic parameters
    jcp.ic_block_int_np = jcp.is_nspc
            ? nstl::min(jcp.ic_block_int, jcp.ic_without_padding)
            : jcp.ic_block_int;
    bool is_small_ic = jcp.ic_block_int_np < jcp.ic_block_int;

    // reduced lowering
    jcp.is_relo = is_small_ic
            // no trivial cases
            && 1 < jcp.kh * jcp.kw
            // required for use of VPERMB instruction in weights copy kernel
            && IMPLICATION(is_int8_convolution,
                    cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
            // no dilation or excessive stride along w-direction
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            // no dilation or excessive stride along h-direction
            && jcp.stride_h <= jcp.kh && jcp.stride_w <= jcp.kw;
    jcp.nreduce = jcp.kh * jcp.kw * jcp.ic_block_int_np;

    if (!jcp.is_relo) {
        jcp.ic_block_int_np = is_bf16_convolution
                ? jcp.ic_block_int
                : rnd_up(jcp.ic_block_int_np, vnni_width);
        is_small_ic = jcp.ic_block_int_np < jcp.ic_block_int;
    }

    // k-remainders
    jcp.kw_per_tile = is_small_ic && !jcp.is_relo && jcp.dilate_w == 0
                    && jcp.stride_w <= jcp.kw // TODO: relax this restriction
                    && jcp.kw * jcp.ic_block_int_np <= jcp.ic_block_int
            ? jcp.kw
            : 1;
    jcp.is_pbuffer_strided = (1 == jcp.kw_per_tile);
    jcp.n_stride_sets
            = jcp.is_pbuffer_strided ? nstl::min(jcp.stride_w, jcp.kw) : 1;
    jcp.kw_step = jcp.is_pbuffer_strided ? jcp.stride_w : jcp.kw_per_tile;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag;
        wei_tag = jcp.is_relo ? pick(with_groups + 2 * (ndims - 3), Owi16o,
                          gOwi16o, Owhi16o, gOwhi16o)
                              : is_bf16_convolution
                        ? pick(with_groups + 2 * (ndims - 3), OIw16i16o2i,
                                gOIw16i16o2i, OIhw16i16o2i, gOIhw16i16o2i)
                        : is_small_ic ? pick(with_groups + 2 * (ndims - 3),
                                  OwI16o4i, gOwI16o4i, OhwI16o4i, gOhwI16o4i)
                                      : pick(with_groups + 2 * (ndims - 3),
                                              OIw16i16o4i, gOIw16i16o4i,
                                              OIhw16i16o4i, gOIhw16i16o4i);

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }
        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_acc = sizeof(int32_t);

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_int = div_up(jcp.ic, jcp.ic_block_int);

    jcp.nb_oc_blocking_thr_chunk = 1;

    const int max_palette = amx::get_max_palette();
    jcp.max_tiles = amx::get_max_tiles(max_palette);
    jcp.full_tile_width = amx::get_max_rows(max_palette);
    if (jcp.max_tiles != 8 || jcp.full_tile_width != 16)
        return status::unimplemented;

    // Pack n rows per tile, such that:
    // ow + (ow + gen_kw - 1) * (n - 1) <= jcp.full_tile_width
    auto calculate_tile_width = [&](int n) {
        assert(n > 0);
        return jcp.ow + (gen_kw + jcp.ow - 1) * (n - 1);
    };
    const bool ok_to_pack_tile = !jcp.is_relo
            && (utils::everyone_is(1, jcp.kh, jcp.kw)
                    || utils::everyone_is(1, jcp.stride_h, jcp.stride_w));
    const int max_oh_per_tile
            = 1 + (jcp.full_tile_width - jcp.ow) / (jcp.ow + gen_kw - 1);
    jcp.oh_per_tile = ok_to_pack_tile
            ? nstl::min(jcp.oh, nstl::max(1, max_oh_per_tile))
            : 1;
    jcp.tile_width = nstl::min<int>(
            jcp.full_tile_width, calculate_tile_width(jcp.oh_per_tile));
    jcp.ow_blocks = utils::div_up(jcp.ow, jcp.tile_width);

    // Prefer to use a single tile width when possible
    // (eg ow28 => 2 tiles of 14 vs 1 of 16 and 1 of 12)
    if (jcp.oh_per_tile == 1 && jcp.ow % jcp.ow_blocks == 0)
        jcp.tile_width = jcp.ow / jcp.ow_blocks;
    jcp.tile_tail = jcp.oh_per_tile == 1 ? jcp.ow % jcp.tile_width : 0;

    jcp.nb_oc_blocking = (jcp.nb_oc % 2 == 0) ? 2 : 1;
    jcp.nb_ic_blocking = 1;
    jcp.nb_oh_blocking
            = utils::everyone_is(true, jcp.tile_tail == 0,
                      // requirement for interleave stores
                      IMPLICATION(jcp.ow_blocks > 1, jcp.oh % 2 == 0),
                      // requirement for small spatial
                      utils::div_up(jcp.oh, jcp.oh_per_tile) > 1,
                      // choose maximal pbuffer overlap for reduced lowering
                      !jcp.is_relo)
            ? 2
            : 1;

    // TODO: tune oh blocking
    const int oh_blk_size_param = jcp.is_relo ? 1 : 10;
    const int oh_step_size = jcp.nb_oh_blocking * jcp.oh_per_tile;
    const int oh_blk_size = rnd_up(oh_blk_size_param, oh_step_size);
    jcp.oh_blk_size = rnd_up(nstl::min(jcp.oh, oh_blk_size), oh_step_size);
    // ihp means here input buffer height including padding - the number
    // of input rows required for computation of jcp.oh_blk_size output rows;
    // if input row doesn't participate in computation of output any row it
    // isn't copied to buffer at all (jcp.stride_h > jcp.kh case)
    jcp.ihp = jcp.is_relo
            ? jcp.oh_blk_size
            : (jcp.oh_blk_size - 1) * nstl::min(jcp.stride_h, gen_kh) + gen_kh;

    // TODO: tune ow blocking
    const int ow_blocks_per_call = jcp.is_relo ? 10 : 2;
    jcp.ow_block = nstl::min(jcp.ow, jcp.tile_width * ow_blocks_per_call);
    jcp.nb_ow = utils::div_up(jcp.ow, jcp.ow_block);
    // iwp includes all width elements that are really used in calculation
    // including left and right zero padding
    const bool are_sets_interleaved
            = IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1);
    jcp.iwp = are_sets_interleaved
            ? (jcp.ow_block - 1) * nstl::min(jcp.stride_w, jcp.kw) + gen_kw
            : jcp.ow_block * jcp.kw;

    // Number of ops per tile store
    int ops_tile_store = jcp.tile_width;
    // Number of ops per accumulation tile
    int avaliable_ops = jcp.is_relo
            ? utils::div_up(jcp.nreduce, jcp.ic_block_int)
            : jcp.nb_ic_int * jcp.kh * (jcp.kw / jcp.kw_per_tile);
    // Number of vectors to store per tile operation
    // NOTE: set to zero to turn off interleave store (mostly for debugging)
    jcp.per_one_pstore = utils::div_up(ops_tile_store, avaliable_ops);

    const int kh_fac = jcp.is_relo ? jcp.kh : 1;
    jcp.inp_buffer_size
            = jcp.nb_ic_int * jcp.ihp * jcp.iwp * kh_fac * jcp.ic_block_int_np
            // pbuffer pointer shifts each oh step for reduced-lowering
            + jcp.is_relo * (jcp.oh - 1) * jcp.stride_h * jcp.ic_block_int_np
            // extra $line due to pbuffer writing full Zmm
            + jcp.ic_block_int;
    jcp.wei_buffer_size = jcp.ngroups * jcp.nb_oc
            * rnd_up(jcp.kh * jcp.kw * jcp.ic * jcp.oc_block, 1024);
    jcp.wsp_buffer_size = jcp.nb_oh_blocking * jcp.nb_oc_blocking
            * jcp.full_tile_width * jcp.oc_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    return status::success;
}

void jit_avx512_core_amx_fwd_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {

    size_t inp_buffer_size = jcp.nthr * jcp.inp_buffer_size;
    scratchpad.book(key_conv_amx_inp_buffer, inp_buffer_size, jcp.typesize_in);
    if (jcp.is_relo) {
        scratchpad.book(
                key_conv_amx_wei_buffer, jcp.wei_buffer_size, jcp.typesize_in);
    }

    size_t wsp_size = jcp.nthr * jcp.wsp_buffer_size;
    scratchpad.book(key_conv_amx_wsp_buffer, wsp_size, jcp.typesize_acc);
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_bia);
    }
    scratchpad.book(key_conv_amx_tilecfg, 1, 64); // 1 whole cacheline
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
