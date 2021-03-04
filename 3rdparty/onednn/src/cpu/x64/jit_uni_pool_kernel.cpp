/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
* Copyright 2018 YANDEX LLC
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
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/x64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::~jit_uni_pool_kernel() = default;

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::jit_uni_pool_kernel(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jpp(ajpp), bf16_emu_(nullptr) {
    if (jpp.is_bf16 && !isa_has_bf16(jpp.isa))
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                bf16_emu_reserv_4, bf16_emu_reserv_5);

    if (jpp.with_postops) {
        static constexpr bool use_per_oc_spatial_strategy = false;

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jpp.post_ops,
                        binary_injector::static_params_t(reg_param,
                                use_per_oc_spatial_strategy,
                                binary_injector::rhs_arg_static_params_t {
                                        static_cast<std::size_t>(
                                                this->xmm4.getIdx()),
                                        this->rax, this->rdx,
                                        true /*preserve gpr*/,
                                        true /*preserve vmm*/,
                                        GET_OFF(post_ops_binary_rhs_arg_vec),
                                        memory_desc_wrapper(*dst_md)}));
    }
}

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    const bool is_avx512 = utils::one_of(isa, avx512_common, avx512_core);
    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = is_avx512 ? 16 : 8;

    jpp.alg = pd.alg_kind;

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::one_of(isa, avx512_common, avx512_core)
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    auto ncsp_fmt_tag = format_tag::undef;

    const unsigned int L3_cache_size_per_core
            = platform::get_per_core_cache_size(3);
    const size_t block_size
            = ((size_t)jpp.id * jpp.ih * jpp.iw + jpp.od * jpp.oh * jpp.ow)
            * jpp.c_block * types::data_type_size(src_d.data_type());

    const bool forward_ncsp_allowed = !jpp.is_backward
            && jpp.c_without_padding > 3
            && ((jpp.ih > 1 && jpp.iw > 1
                        && block_size <= L3_cache_size_per_core)
                    || src_d.data_type() == data_type::bf16);

    const bool backward_ncsp_allowed = jpp.is_backward
            && ((jpp.ih > 1 && jpp.iw > 1 && jpp.c_without_padding > 1
                        && block_size <= L3_cache_size_per_core)
                    || (src_d.data_type() == data_type::bf16
                            && !(jpp.alg == pooling_max
                                    && block_size > L3_cache_size_per_core)));

    ncsp_fmt_tag = ((forward_ncsp_allowed || backward_ncsp_allowed)
                           && isa == avx512_core && ndims <= 5)
            ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jptg_ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag) ? jptg_nspc : jptg_blocked;
    }

    jpp.isa = (jpp.is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16
                                                         : isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && IMPLICATION(jpp.is_bf16, mayiuse(avx512_core))
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    jpp.c = jpp.tag_kind == jptg_blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jptg_blocked) assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jptg_blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;

        if ((isa == avx || isa == avx2) && jpp.c_tail > 0)
            // Additional register needed for tail mask
            jpp.ur -= 1;

        if (jpp.is_training)
            jpp.ur = is_avx512 ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur = is_avx512 ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur = is_avx512 ? 12 : 6;
        else
            jpp.ur = is_avx512 ? 24 : 12;
    }
    if (jpp.is_bf16) {
        jpp.ur = (!isa_has_bf16(jpp.isa))
                ? jpp.ur - 4 // Free registers for AVX512 emulation
                : jpp.ur - 1; // Free register for cvt from bf16 to f32
    }

    // select jpp.ur_bc
    if (jpp.tag_kind == jptg_nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, nthreads);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9) break; // Heuristic threshold
        }

        //take into account cache re-usage after zeroing on backward
        if (jpp.is_backward && ndims < 5) {
            const int L2 = platform::get_per_core_cache_size(2)
                    / sizeof(jpp.dt_size);
            int ur_bc = nstl::max(1, L2 / (jpp.kh * jpp.iw * jpp.c_block));
            jpp.ur_bc = nstl::min(jpp.ur_bc, ur_bc);
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }
    auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
    if (utils::div_up(jpp.l_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;
    if (utils::div_up(right_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jptg_ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    const auto attr = *ppd->attr();
    if (!post_ops_ok(jpp, attr, dst_d)) return status::unimplemented;

    jpp.post_ops = attr.post_ops_;

    return status::success;
}

static int reg_ind(int shift, int bc, int j, int ur_bc, int ur_w) noexcept {
    return shift * ur_bc * ur_w + bc * ur_w + j;
};

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::prepare_tail_mask() {
    if (isa >= avx512_common) {
        size_t c_tail_mask = (1ULL << jpp.c_tail) - 1ULL;
        mov(tmp_gpr.cvt32(), c_tail_mask);
        kmovw(k_c_tail_mask, tmp_gpr.cvt32());
    } else if (isa == avx || isa == avx2) {
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};
        mov(tmp_gpr, reinterpret_cast<size_t>(&mask[8 - jpp.c_tail]));
        vmovups(vmm_c_tail_mask, ptr[tmp_gpr]);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::put_one_in_vmm() {
    mov(tmp_gpr, 1);
    uni_broadcast_reg_val(tmp_gpr.getIdx(), vmm_one.getIdx());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::uni_broadcast_reg_val(
        const int reg_idx, const int vmm_idx) {
    movq(Xmm(vmm_idx), reg64_t(reg_idx));
    uni_vpbroadcastd(Vmm(vmm_idx), Xmm(vmm_idx));
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::push_vmm_val(const int idx) {
    Vmm val_to_store(idx);
    sub(rsp, val_to_store.getBit());
    uni_vmovups(ptr[rsp], val_to_store);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::pop_vmm_val(const int idx) {
    Vmm val_to_load(idx);
    uni_vmovups(val_to_load, ptr[rsp]);
    add(rsp, val_to_load.getBit());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx,
        const reg64_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        /*TODO: maybe use vpmovzxwd + vpslld,
             * in order to free up vmm_idx() register */
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            Vmm vmm_to_load = is_c_tail_proccessing
                    ? Vmm(idx) | k_c_tail_mask | T_z
                    : Vmm(idx);
            vpmovzxwd(vmm_to_load, ptr[reg_ptr + offset]);
            vpslld(vmm_to_load, vmm_to_load, 16);
        } else {
            vmovups(Ymm(idx), ptr[reg_ptr + offset]);
            vpermw(Vmm(idx) | k_mask_cvt | T_z, vmm_idx(), Vmm(idx));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    pinsrd(Xmm(idx), ptr[reg_ptr + offset + i * jpp.dt_size],
                            i);
                }
            } else if (isa == avx || isa == avx2) {
                vmaskmovps(Vmm(idx), vmm_c_tail_mask, ptr[reg_ptr + offset]);
            } else {
                vmovups(Zmm(idx) | k_c_tail_mask | T_z, ptr[reg_ptr + offset]);
            }
        } else {
            uni_vmovups(Vmm(idx), ptr[reg_ptr + offset]);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const reg64_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            vmovdqu16(ptr[reg_ptr + offset] | k_c_tail_mask, Ymm(idx));
        } else {
            vmovups(yword[reg_ptr + offset], Ymm(idx));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    pextrd(ptr[reg_ptr + offset + i * jpp.dt_size], Xmm(idx),
                            i);
                }
            } else if (isa == avx || isa == avx2) {
                vmaskmovps(ptr[reg_ptr + offset], vmm_c_tail_mask, Vmm(idx));
            } else {
                vmovups(ptr[reg_ptr + offset] | k_c_tail_mask, Zmm(idx));
            }
        } else {
            uni_vmovups(vmmword[reg_ptr + offset], Vmm(idx));
        }
    }
}
template <cpu_isa_t isa>
bool jit_uni_pool_kernel<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (!jpp.is_backward) {
        for (const auto &entry : entries) {
            if (entry.is_eltwise()) {
                jpp.with_eltwise = true;
            } else if (entry.is_binary()) {
                jpp.with_binary = true;
            } else
                return false;
        }

        jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    }

    if (jpp.with_eltwise && isa == avx) return false;

    return binary_injector::binary_args_broadcast_supported(post_ops, dst_d)
            && binary_injector::binary_args_tail_supported(
                    post_ops, dst_d, cpu_isa_traits<isa>::vlen);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::apply_postops(int ur_bc, int ur_w, int c_block) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    const int end_idx = vmm_idx_upper_bound() + 1;
    const int start_idx = end_idx - (ur_bc * ur_w);
    const bool sse41_postops_disabled
            = isa == sse41 && disable_postops_when_sse_high_half_processed_;

    if (jpp.with_binary && !sse41_postops_disabled) {
        static constexpr int sse41_simd_w
                = cpu_isa_traits<sse41>::vlen / sizeof(float);
        const int sse_elem_off = sse_high_half ? sse41_simd_w : 0;
        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto vmm_idx
                        = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).getIdx();
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr[reg_param + GET_OFF(c_elem_off)]);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, bci * c_block + sse_elem_off);
            }
        }
    }
    postops_injector_->compute_vector_range(start_idx, end_idx, rhs_arg_params);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r, bool with_c_tail_proccessing) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj * stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * stride_w);

        if (non_zero_kw != prev_kw) {
            mov(tmp_gpr, float2int((float)non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            if (with_c_tail_proccessing && (isa == avx || isa == avx2)) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                uni_broadcast_reg_val(
                        reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
            }
            uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            if (with_c_tail_proccessing && (isa == avx || isa == avx2)) {
                pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::avg_step(int ur_w, int ur_bc, int pad_l,
        int pad_r, bool with_c_tail_proccessing) {

    auto iw = jpp.iw;
    auto kw = jpp.kw;
    auto stride_w = jpp.stride_w;
    auto c_block = jpp.c_block;
    auto dt_size = jpp.dt_size;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        if (isa == sse41 && !jpp.is_c_padded) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward)
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
        for (int bci = 0; bci < ur_bc; bci++) {
            const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                load(accvr.getIdx(), reg_output, output_offset,
                        is_tail_processing(bci));
                uni_vdivps(accvr, accvr, vmm_tmp);
            } else {
                uni_vpxor(accvr, accvr, accvr);
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    auto inpyr = yreg(inpr_i);
                    load(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                    uni_vaddps(inpvr, inpvr, accvr);
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(inpyr, zreg(inpr_i));
                        else
                            vcvtneps2bf16(inpyr, inpvr);
                    }
                    store(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                } else {
                    if (jpp.is_bf16 || is_tail_processing(bci)
                            || (isa == sse41
                                    && c_off % (jpp.c_block / 2) != 0)) {
                        load(vmm_tmp_1.getIdx(), aux_reg_input, input_offset,
                                is_tail_processing(bci));

                        uni_vaddps(accvr, accvr, vmm_tmp_1);
                    } else {
                        uni_vaddps(accvr, accvr,
                                ptr[aux_reg_input + input_offset]);
                    }
                }
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto accvr = vreg(accr_i);
                uni_vdivps(accvr, accvr, vmm_tmp);
            }
        }

        if (jpp.with_postops) apply_postops(ur_bc, ur_w, c_block);

        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto accvr = vreg(accr_i);
                const auto output_offset
                        = dt_size * (jj * c_off + bci * c_block);
                if (jpp.is_bf16) {
                    const auto acczr = zreg(accr_i);
                    const auto accyr = yreg(accr_i);
                    if (!isa_has_bf16(jpp.isa))
                        bf16_emu_->vcvtneps2bf16(accyr, acczr);
                    else
                        vcvtneps2bf16(accyr, accvr);
                }
                store(reg_idx(accr_i), reg_output, output_offset,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_fwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto is_tail_processing = [&](int bc) {
        if (isa == sse41 && !jpp.is_c_padded) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    mov(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    movq(xmm_tmp, tmp_gpr);
    uni_vbroadcastss(vmm_tmp, xmm_tmp);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
        uni_vmovups(accvr, vmm_tmp);
        if (jpp.is_training) {
            const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
            uni_vpxor(indvr, indvr, indvr);
        }
    }
    if (jpp.is_training) {
        movq(xmm_tmp, reg_k_shift);
        uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
    }
    if (jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
                load(reg_idx(inpr_i), aux_reg_input, input_offset,
                        is_tail_processing(bci));
                if (isa == sse41) {
                    movups(vmm_mask, accvr);
                    cmpps(vmm_mask, inpvr, _cmp_lt_os);
                    blendvps(accvr, inpvr);
                    if (jpp.is_training) blendvps(indvr, vmm_k_offset);
                } else if (isa == avx || isa == avx2) {
                    vcmpps(cvtvr, accvr, inpvr, _cmp_lt_os);
                    vblendvps(accvr, accvr, inpvr, cvtvr);
                    if (jpp.is_training)
                        vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                } else {
                    vcmpps(k_store_mask, accvr, inpvr, _cmp_lt_os);
                    vblendmps(accvr | k_store_mask, accvr, inpvr);
                    if (jpp.is_training)
                        vblendmps(indvr | k_store_mask, indvr, vmm_k_offset);
                }
            }
            if (jpp.is_training) {
                if (with_c_tail_proccessing && (isa == avx || isa == avx2)) {
                    push_vmm_val(vmm_c_tail_mask.getIdx());
                    put_one_in_vmm();
                }

                if (isa == avx && !mayiuse(avx2))
                    avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
                else
                    uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);

                if (with_c_tail_proccessing && (isa == avx || isa == avx2))
                    pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        if (jpp.is_training) {
            mov(tmp_gpr, ptr[reg_param + GET_OFF(kd_padding_shift)]);
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            if (isa == avx && !mayiuse(avx2)) {
                Xmm t(vmm_mask.getIdx());
                avx_vpadd1(vmm_k_offset, xmm_tmp, t);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
            }
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    if (with_c_tail_proccessing && jpp.is_c_padded && isa == sse41)
        mov(tmp_gpr, 0); // needed zero to fill padded tail

    if (jpp.with_postops) apply_postops(ur_bc, ur_w, c_block);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        const auto accvr = vreg(accr_i);
        const auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        if (jpp.is_bf16) {
            auto acczr = zreg(accr_i);
            auto accyr = yreg(accr_i);
            if (!isa_has_bf16(jpp.isa))
                bf16_emu_->vcvtneps2bf16(accyr, acczr);
            else
                vcvtneps2bf16(accyr, accvr);
        }
        store(reg_idx(accr_i), reg_output, output_offset,
                is_tail_processing(bci));

        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            const auto indr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                auto xr = xreg(indr_i);
                if (isa == sse41) {
                    for (int i = 0; i < (jpp.c_block / 2); ++i) {
                        if (is_tail_processing(bci)
                                && i + (sse_high_half ? (jpp.c_block / 2) : 0)
                                        >= jpp.c_tail) {
                            if (jpp.is_c_padded)
                                mov(ptr[reg_index + step_index + i],
                                        tmp_gpr.cvt8()); // fill padded tail with zeros
                            else
                                break; // tail end
                        } else {
                            // bytes which should be stored are located in
                            // least significant bits(8 to be precise) of 32 bits parts
                            // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                            pextrb(ptr[reg_index + step_index + i], xr, 4 * i);
                        }
                    }
                } else if (isa == avx || isa == avx2) {
                    auto yr = yreg(indr_i);
                    if (is_tail_processing(bci) && !jpp.is_c_padded) {
                        const int max_nr_of_vals
                                = jpp.c_tail > (jpp.c_block / 2)
                                ? (jpp.c_block / 2)
                                : jpp.c_tail;
                        for (int i = 0; i < max_nr_of_vals; ++i) {
                            // bytes which should be stored are located in
                            // least significant bits(8 to be precise) of 32 bits parts
                            // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                            vpextrb(ptr[reg_index + step_index + i], xr, 4 * i);
                        }

                        if (jpp.c_tail > (jpp.c_block / 2)) {
                            Xmm higher_128bits(vmm_mask.getIdx());
                            vextractf128(higher_128bits, yr, 1);
                            for (int i = 0; i < jpp.c_tail - (jpp.c_block / 2);
                                    ++i) {
                                // bytes which should be stored are located in
                                // least significant bits(8 to be precise) of 32 bits parts
                                // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                                vpextrb(ptr[reg_index + step_index
                                                + (jpp.c_block / 2) + i],
                                        higher_128bits, 4 * i);
                            }
                        }
                    } else {
                        if (is_tail_processing(bci)) {
                            assert(jpp.is_c_padded);
                            vandps(yr, yr, vmm_c_tail_mask);
                        }
                        if (jj == 0) {
                            vmovd(xmm_tmp, reg_shuf_mask);
                            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
                        }
                        if (mayiuse(avx2)) {
                            vpshufb(yr, yr, vmm_tmp);
                            vmovd(ptr[reg_index + step_index], xr);
                            vperm2i128(yr, yr, yr, 0x1u);
                            vmovd(ptr[reg_index + step_index
                                          + (jpp.c_block / 2)],
                                    xr);
                        } else {
                            Xmm t(vmm_mask.getIdx());
                            vextractf128(t, yr, 0);
                            vpshufb(t, t, xmm_tmp);
                            vmovd(ptr[reg_index + step_index], t);
                            vextractf128(t, yr, 1);
                            vpshufb(t, t,
                                    xmm_tmp); // ymm_tmp[:128]==ymm_tmp[127:0]
                            vmovd(ptr[reg_index + step_index
                                          + (jpp.c_block / 2)],
                                    t);
                        }
                    }
                } else {
                    if (is_tail_processing(bci)) {
                        if (jpp.is_c_padded) {
                            knotw(k_c_tail_mask, k_c_tail_mask);
                            vpxord(vr | k_c_tail_mask, vr, vr);
                            knotw(k_c_tail_mask, k_c_tail_mask);
                            vpmovusdb(ptr[reg_index + step_index], vr);
                        } else
                            vpmovusdb(ptr[reg_index + step_index],
                                    vr | k_c_tail_mask);
                    } else {
                        vpmovusdb(ptr[reg_index + step_index], vr);
                    }
                }
            } else {
                store(vr.getIdx(), reg_index, step_index,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_bwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        if (isa == sse41) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half)
                            || (jpp.c_tail == (jpp.c_block / 2) && sse_high_half
                                    && jpp.is_c_padded));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto outr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        load(reg_idx(outr_i), reg_output, out_offset, is_tail_processing(bci));
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        const auto indr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            auto indxr = xreg(indr_i);
            if (isa == sse41) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++)
                        pinsrb(indxr, ptr[reg_index + step_index + i], i);
                } else {
                    movd(indxr, ptr[reg_index + step_index]);
                }
                pmovzxbd(indvr, indxr);
            } else if (isa == avx || isa == avx2) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    for (int i = 0; i < jpp.c_tail; i++)
                        vpinsrb(indxr, indxr, ptr[reg_index + step_index + i],
                                i);
                } else {
                    vmovq(indxr, ptr[reg_index + step_index]);
                }
                if (!mayiuse(avx2)) {
                    avx_pmovzxbd(indvr, indxr, xmm_tmp);
                } else {
                    vpmovzxbd(indvr, indxr);
                }
            } else {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    vpmovzxbd(indvr | k_c_tail_mask | T_z,
                            ptr[reg_index + step_index]);
                } else {
                    vpmovzxbd(indvr, ptr[reg_index + step_index]);
                }
            }
        } else {
            load(indvr.getIdx(), reg_index, step_index,
                    is_tail_processing(bci));
        }
    }
    movq(xmm_tmp, reg_k_shift);
    uni_vpbroadcastd(vmm_k_offset, xmm_tmp);

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            push(dst_ptr);
        }
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        mov(reg_kd_pad_shift, ptr[reg_param + GET_OFF(kd_padding_shift)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto outvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto indvr = vreg(reg_ind(1, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
                load(reg_idx(inpr_i), aux_reg_input, inp_offset,
                        is_tail_processing(bci));
                if (isa == sse41) {
                    mov(dst_ptr, aux_reg_input);
                    add(dst_ptr, inp_offset);

                    movups(cvtvr, indvr);
                    pcmpeqd(cvtvr, vmm_k_offset);
                    addps(inpvr, outvr);
                    if (is_tail_processing(bci)) {
                        Label end_cond_move[4];
                        for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2);
                                i++) {
                            pextrd(tmp_gpr.cvt32(), cvtvr, i);
                            cmp(tmp_gpr, 0);
                            je(end_cond_move[i], T_NEAR);
                            pextrd(ptr[dst_ptr + i * jpp.dt_size], inpvr, i);
                            L(end_cond_move[i]);
                        }
                    } else
                        maskmovdqu(inpvr, cvtvr);
                } else if (isa == avx || isa == avx2) {
                    if (mayiuse(avx2)) {
                        vpcmpeqd(cvtvr, indvr, vmm_k_offset);
                    } else {
                        avx_pcmpeqd(cvtvr, indvr, vmm_k_offset, xmm_tmp);
                    }
                    vaddps(inpvr, inpvr, outvr);
                    if (is_tail_processing(bci)) {
                        vandps(cvtvr, cvtvr, vmm_c_tail_mask);
                    }
                    vmaskmovps(
                            vmmword[aux_reg_input + inp_offset], cvtvr, inpvr);
                } else {
                    auto indzr = zreg(inpr_i);
                    auto indyr = yreg(inpr_i);
                    vpcmpeqd(k_store_mask, indvr, vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, outvr, outvr);
                    vaddps(inpvr, inpvr, vmm_tmp);
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(indyr, indzr);
                        else
                            vcvtneps2bf16(indyr, inpvr);
                    }
                    store(inpvr.getIdx(), aux_reg_input, inp_offset,
                            is_tail_processing(bci));
                }
            }

            if (with_c_tail_proccessing && (isa == avx || isa == avx2)) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                put_one_in_vmm();
            }

            if (isa == avx && !mayiuse(avx2)) {
                avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
            }

            if (with_c_tail_proccessing && (isa == avx || isa == avx2))
                pop_vmm_val(vmm_c_tail_mask.getIdx());
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);

        mov(tmp_gpr, reg_kd_pad_shift);
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        if (isa == avx && !mayiuse(avx2)) {
            Xmm t(vmm_mask.getIdx());
            avx_vpadd1(vmm_k_offset, vmm_tmp, t);
        } else {
            uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            pop(dst_ptr);
        }
        pop(reg_output);
        pop(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::zero_diff_src(
        int ur_bc, bool with_c_tail_proccessing) {
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : jpp.c_block;

    Label l_skip, l_ih_loop, l_id_loop;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    mov(reg_zero_id, ptr[reg_param + GET_OFF(zero_id)]);
    cmp(reg_zero_id, 0);
    jz(l_skip, T_NEAR);

    mov(reg_zero_ih, ptr[reg_param + GET_OFF(zero_ih)]);
    cmp(reg_zero_ih, 0);
    jz(l_skip, T_NEAR);

    mov(reg_zero_ptr, ptr[reg_param + GET_OFF(zero_ptr)]);

    Vmm vzero = vmm_tmp;
    uni_vpxor(vzero, vzero, vzero);

    const int width_size = jpp.iw * c_off * jpp.dt_size;

    auto aux_reg_zero_ptr = tmp_gpr;

    L(l_id_loop);
    {
        mov(aux_reg_zero_ptr, reg_zero_ptr);
        mov(aux_reg_zero_ih, reg_zero_ih);
        L(l_ih_loop);
        {
            const auto vlen = cpu_isa_traits<isa>::vlen;
            const int step = c_off * jpp.dt_size;

            // TODO: maybe a big code generated here
            for_(int i = 0; i < width_size; i += step)
            for (int bci = 0; bci < ur_bc; bci++) {
                const int offs = i + bci * jpp.c_block * jpp.dt_size;
                if (isa == sse41) {
                    bool is_needed_c_tail_processing = false;
                    if (is_tail_processing(bci)
                            && jpp.c_tail < (jpp.c_block / 2))
                        is_needed_c_tail_processing = true;
                    store(vzero.getIdx(), reg_zero_ptr, offs,
                            is_needed_c_tail_processing);
                    if (!is_tail_processing(bci)
                            || (is_tail_processing(bci)
                                    && (jpp.is_c_padded
                                            || jpp.c_tail
                                                    > (jpp.c_block / 2)))) {
                        store(vzero.getIdx(), reg_zero_ptr, offs + vlen,
                                is_tail_processing(bci));
                    }

                } else {
                    store(vzero.getIdx(), reg_zero_ptr, offs,
                            is_tail_processing(bci));
                }
            }
            add(reg_zero_ptr, width_size);
            dec(aux_reg_zero_ih);
            jnz(l_ih_loop, T_NEAR);
        }
        mov(reg_zero_ptr, aux_reg_zero_ptr);
        add(reg_zero_ptr, width_size * jpp.ih);
        dec(reg_zero_id);
        jnz(l_id_loop, T_NEAR);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::generate() {

    this->preamble();

    Label idx_table;

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;

    int vlen = cpu_isa_traits<isa>::vlen;

#if defined(_WIN32)
    // Always mimic the Unix ABI (see the note about maskmovdqu in the header
    // file).
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif
    if (!isa_has_bf16(jpp.isa) && jpp.is_bf16) bf16_emu_->init_vcvtneps2bf16();

    mov(reg_input, ptr[reg_param + GET_OFF(src)]);
    mov(reg_output, ptr[reg_param + GET_OFF(dst)]);
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[reg_param + GET_OFF(indices)]);
    mov(reg_kh, ptr[reg_param + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[reg_param + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[reg_param + GET_OFF(ker_area_h)]);
    mov(reg_nbc, ptr[reg_param + GET_OFF(ur_bc)]);

    if (jpp.is_bf16) {
        mov(tmp_gpr.cvt32(), 0xAAAAAAAA);
        kmovd(k_mask_cvt, tmp_gpr.cvt32());

        mov(tmp_gpr, idx_table);
        vmovups(vmm_idx(), ptr[tmp_gpr]);
    }

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool with_c_tail_proccessing,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);

        if (isa == sse41) {
            if (with_c_tail_proccessing && jpp.c_tail <= (jpp.c_block / 2)) {

                // In nspc format in case of c tail processing if c tail is
                // equal or lower than 4 we don't have to process
                // last high half block, because it doesn't exist
                if (!jpp.is_c_padded) ur_bc -= 1;
                /*
                 * In case of c_tail_processing if c_tail is equal or lower than 4
                 * applying postops never make sense. In case of blocked format it
                 * can cause overwriting zero padding or segfault because the element
                 * corresponding to the piece with padded zeros doesn't exist in binary
                 * postops arg1 tensor (nchw format) in per_oc bcast strategy.
                 */
                disable_postops_when_sse_high_half_processed_
                        = jpp.tag_kind == jptg_blocked;
            }
            sse_high_half = true;
            step_high_half(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);
            sse_high_half = false;
            disable_postops_when_sse_high_half_processed_ = false;
        }

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        auto shift = (isa == sse41) ? vlen : 0;
        add(reg_input, dt_size * (ur_w * stride_w - lpad) * c_off - shift);
        add(reg_output, dt_size * ur_w * c_off - shift);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ishift = (isa == sse41) ? jpp.c_block / 2 : 0;
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
            add(reg_index, (ur_w * c_off - ishift) * ind_dt_size);
        }
    };

    auto perform_ker = [&](int ur_bc, bool with_c_tail_processing) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg)
            zero_diff_src(ur_bc, with_c_tail_processing);

        if (jpp.alg == pooling_avg_exclude_padding
                && (!with_c_tail_processing || (isa != avx && isa != avx2))) {
            // vmm_ker_area_h and vmm_c_tail_mask are stored in one register
            // so when vmm_c_tail_mask is used we need to load vmm_ker_area_h
            // exactly where this information is needed with the
            // vmm_c_tail_mask information being saved first
            uni_broadcast_reg_val(
                    reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
        }

        if (jpp.alg == pooling_avg_include_padding) {
            mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            if (!with_c_tail_processing || (isa != avx && isa != avx2)) {
                // The same situation as above(vmm_ker_area_h).
                put_one_in_vmm();
            }

            if (isa == avx || isa == avx2) { mov(reg_shuf_mask, 0x0c080400); }
        }

        auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
        auto ur_w_tail = jpp.ow % ur_w;

        int n_oi = ow / ur_w;

        int r_pad1
                = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w, kw);
        if (r_pad1 > 0) n_oi--;

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                process_oi(ur_w, ur_bc, l_pad, r_pad1, with_c_tail_processing);
            else
                process_oi(ur_w, ur_bc, l_pad, 0, with_c_tail_processing);
        }

        xor_(oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0, with_c_tail_processing);

                inc(oi_iter);
                cmp(oi_iter, n_oi);
                jl(ow_loop, T_NEAR);
            }
        }

        if (r_pad1 > 0 && n_oi >= 0)
            process_oi(ur_w, ur_bc, 0, r_pad1, with_c_tail_processing);

        if (ur_w_tail != 0)
            process_oi(
                    ur_w_tail, ur_bc, 0, r_pad, with_c_tail_processing, false);
    };
    Label ur_bc_tail_label, c_tail_processing_label, finish_label;

    if (jpp.ur_bc_tail > 0) {
        cmp(reg_nbc, jpp.ur_bc);
        jne(ur_bc_tail_label, T_NEAR);
    } else if (jpp.c_tail != 0) {
        // ur_bc contains number of channel blocks to processing
        // b_c contains number of channel blocks already processed
        // If reg_nbc + tmp_gpr == jpp.nb_c then this is
        // information that probably channel tail processing will be needed.
        mov(tmp_gpr, ptr[reg_param + GET_OFF(b_c)]);
        add(tmp_gpr, reg_nbc);
        cmp(tmp_gpr, jpp.nb_c);
        je(c_tail_processing_label, T_NEAR);
    }

    perform_ker(jpp.ur_bc, false);

    if (jpp.ur_bc_tail > 0) {
        jmp(finish_label, T_NEAR);

        // If ur_bc_tail exists then we know that this is
        // last set of blocks to process and we need
        // care of c tail processing if number of channels
        // is not divided by number of channels in block
        L(ur_bc_tail_label);
        if (jpp.c_tail != 0) prepare_tail_mask();
        perform_ker(jpp.ur_bc_tail, jpp.c_tail != 0);

        L(finish_label);
    } else if (jpp.c_tail != 0) {
        jmp(finish_label, T_NEAR);

        L(c_tail_processing_label);
        prepare_tail_mask();
        perform_ker(jpp.ur_bc, true);

        L(finish_label);
    }

    this->postamble();

    if (jpp.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();

    if (jpp.is_bf16) {
        align(64);
        L(idx_table);
        const uint16_t _idx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dw(_idx[i]);
    }
}

template struct jit_uni_pool_kernel<sse41>;
template struct jit_uni_pool_kernel<avx>;
template struct jit_uni_pool_kernel<avx2>;
template struct jit_uni_pool_kernel<avx512_common>;
template struct jit_uni_pool_kernel<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
