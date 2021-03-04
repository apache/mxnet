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

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_common_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_resampling_args_t, field)
struct jit_resampling_args_t {
    const void *src; // fwd: src bwd: diff_dst
    const void *dst; // fwd: dst bwd: diff_src
    dim_t d; // fwd: od  bwd: id
    dim_t h; // fwd: oh  bwd: ih
    dim_t w; // fwd: ow  bwd: iw
};

struct jit_avx512_common_resampling_kernel : public c_compatible {
    jit_avx512_common_resampling_kernel(const resampling_pd_t *pd) : pd_(pd) {}
    virtual ~jit_avx512_common_resampling_kernel() = default;

    virtual status_t create_kernel() = 0;
    virtual void operator()(const jit_resampling_args_t *args) = 0;

protected:
    // Convert between vector register lengths.
    template <typename Vmm>
    static inline Vmm cvt_reg(const Xmm &xmm) {
        return Vmm(xmm.getIdx());
    }

    const resampling_pd_t *pd_;

    data_type_t data_type() const {
        if (pd_->is_fwd())
            return pd_->src_md()->data_type;
        else
            return pd_->diff_src_md()->data_type;
    }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

// jit kernels
namespace {

struct jit_avx512_common_resampling_t
    : public jit_avx512_common_resampling_kernel,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_resampling)

    jit_avx512_common_resampling_t(const resampling_pd_t *pd)
        : jit_avx512_common_resampling_kernel(pd) {

        if (pd_->is_fwd()) {
            const memory_desc_wrapper src_d(pd_->src_md());
            inner_stride_ = src_d.blocking_desc().strides[pd_->ndims() - 1];
            stride_d_ = pd_->IH() * pd_->IW() * inner_stride_;
            stride_h_ = pd_->IW() * inner_stride_;
            stride_w_ = inner_stride_;
        } else {
            const memory_desc_wrapper diff_src_d(pd_->diff_src_md());
            inner_stride_
                    = diff_src_d.blocking_desc().strides[pd_->ndims() - 1];
            stride_d_ = pd_->OH() * pd_->OW() * inner_stride_;
            stride_h_ = pd_->OW() * inner_stride_;
            stride_w_ = inner_stride_;
        }

        number_of_loops_ = (inner_stride_ / simd_w());
        tail_mask_ = (((size_t)1 << (inner_stride_ % simd_w())) - (size_t)1);
        if (tail_mask_ != 0) prepare_mask();
        stack_size_needed_ = 0;
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(const jit_resampling_args_t *args) override {
        return jit_generator::operator()(args);
    }

private:
    enum class rounding_mode { none, floor, ceil, rounding_mode_max };

    struct bwd_counting_range_t {
        RegExp loop_counter;
        struct start_t {
            RegExp linear[2];
            RegExp nearest;
        } start;
        struct end_t {
            RegExp linear[2];
            RegExp nearest;
        } end;
    };

    void round_to_near_away_from_zero(const Reg64 &dst_int_val,
            const Xmm &to_round, const Xmm &zero_point_five, const Xmm &tmp) {
        EvexModifierRounding rm_trunc(EvexModifierRounding::T_RZ_SAE);
        vaddss(tmp, to_round, zero_point_five);
        vcvtss2si(dst_int_val, tmp | rm_trunc);
    }

    void for_begin(Label &begin_loop, Label &end_loop,
            const RegExp &loop_counter, const RegExp &start, const RegExp &end,
            const Reg64 &tmp) {
        // for (initialization; check; incrementation)
        // initialization
        mov(tmp, ptr[start]);
        mov(ptr[loop_counter], tmp);
        L(begin_loop);
        // check
        mov(tmp, ptr[loop_counter]);
        cmp(tmp, ptr[end]);
        jge(end_loop, T_NEAR);
    }

    void for_end(Label &begin_loop, Label &end_loop, const RegExp &loop_counter,
            const Reg64 &tmp) {
        // incrementation
        mov(tmp, ptr[loop_counter]);
        inc(tmp);
        mov(ptr[loop_counter], tmp);
        jmp(begin_loop, T_NEAR);
        L(end_loop);
    }

    void max(const Reg64 &reg, const dim_t to_cmp) {
        mov(reg_tmp, to_cmp);
        cmp(reg, reg_tmp);
        cmovl(reg, reg_tmp);
    }

    void min(const Reg64 &reg, const dim_t to_cmp) {
        mov(reg_tmp, to_cmp);
        cmp(reg, reg_tmp);
        cmovg(reg, reg_tmp);
    }

    void move_imm_float_to_xmm(const Xmm &xmm, const Reg64 &tmp, float imm) {
        mov(tmp.cvt32(), float2int(imm));
        vmovd(xmm, tmp.cvt32());
    }

    void prepare_mask() {
        Reg64 reg_tail = reg_tmp;
        mov(reg_tail.cvt32(), tail_mask_);
        kmovw(k_tail_mask, reg_tail.cvt32());
    }

    void generate() override {
        preamble();

        if (is_bf16()) {
            use_bf16_emulation_ = !mayiuse(avx512_core_bf16);
            if (use_bf16_emulation_) {
                bf16_emulation_.reset(new bf16_emulation_t(this,
                        bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                        bf16_emu_scratch, bf16_emu_reserv_5));
                bf16_emulation_->init_vcvtneps2bf16();
            }
        }

        mov(reg_src, ptr[abi_param1 + GET_OFF(src)]);
        mov(reg_dst, ptr[abi_param1 + GET_OFF(dst)]);

        move_imm_float_to_xmm(xmm_zero_point_five, reg_tmp, 0.5f);

        if (pd_->is_fwd()) {
            // Count coeffs
            if (pd_->ndims() == 5) {
                mov(reg_curr_d, ptr[abi_param1 + GET_OFF(d)]);
                mov(reg_curr_h, ptr[abi_param1 + GET_OFF(h)]);
                mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                count_dim_coeff(xmm_d_coeff, reg_curr_d, pd_->OD(), pd_->ID());
                count_dim_coeff(xmm_h_coeff, reg_curr_h, pd_->OH(), pd_->IH());
                count_dim_coeff(xmm_w_coeff, reg_curr_w, pd_->OW(), pd_->IW());
            } else if (pd_->ndims() == 4) {
                mov(reg_curr_h, ptr[abi_param1 + GET_OFF(h)]);
                mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                count_dim_coeff(xmm_h_coeff, reg_curr_h, pd_->OH(), pd_->IH());
                count_dim_coeff(xmm_w_coeff, reg_curr_w, pd_->OW(), pd_->IW());
            } else {
                mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                count_dim_coeff(xmm_w_coeff, reg_curr_w, pd_->OW(), pd_->IW());
            }
        } else {
            if (pd_->desc()->alg_kind == alg_kind::resampling_linear) {
                // Stack:
                // ow_loop_counter:
                // ow_left_start  : 8
                // ow_left_end    : 16
                // ow_right_start : 24
                // ow_right_end   : 32
                // ----- 3 dims -----
                // oh_loop_counter: 40
                // oh_left_start  : 48
                // oh_left_end    : 56
                // oh_right_start : 64
                // oh_right_end   : 72
                // ----- 4 dims -----
                // od_loop_counter: 80
                // od_left_start  : 88
                // od_left_end    : 96
                // od_right_start : 104
                // od_right_end   : 112
                // ----- 5 dims -----

                // 5*size(int64)*nr_of_spatial_dims
                stack_size_needed_ = 5 * 8 * (pd_->ndims() - 2);
                sub(rsp, stack_size_needed_);

                if (pd_->ndims() == 5) {
                    mov(reg_curr_d, ptr[abi_param1 + GET_OFF(d)]);
                    mov(reg_curr_h, ptr[abi_param1 + GET_OFF(h)]);
                    mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                    count_bwd_counting_range(
                            rsp + 80, od, reg_curr_d, pd_->OD(), pd_->ID());
                    count_bwd_counting_range(
                            rsp + 40, oh, reg_curr_h, pd_->OH(), pd_->IH());
                    count_bwd_counting_range(
                            rsp, ow, reg_curr_w, pd_->OW(), pd_->IW());
                } else if (pd_->ndims() == 4) {
                    mov(reg_curr_h, ptr[abi_param1 + GET_OFF(h)]);
                    mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                    count_bwd_counting_range(
                            rsp + 40, oh, reg_curr_h, pd_->OH(), pd_->IH());
                    count_bwd_counting_range(
                            rsp, ow, reg_curr_w, pd_->OW(), pd_->IW());
                } else {
                    mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                    count_bwd_counting_range(
                            rsp, ow, reg_curr_w, pd_->OW(), pd_->IW());
                }
            } else {
                // Stack:
                // ow_loop_counter:
                // ow_start       : 8
                // ow_end         : 16
                // oh_loop_counter: 24
                // oh_start       : 32
                // oh_end         : 40
                // od_loop_counter: 48
                // od_start       : 56
                // od_end         : 64

                // 3*size(int64)*max_nr_of_spatial_dims
                stack_size_needed_ = 3 * 8 * 3;
                sub(rsp, stack_size_needed_);

                mov(reg_curr_d, ptr[abi_param1 + GET_OFF(d)]);
                mov(reg_curr_h, ptr[abi_param1 + GET_OFF(h)]);
                mov(reg_curr_w, ptr[abi_param1 + GET_OFF(w)]);
                count_bwd_counting_range(
                        rsp + 48, od, reg_curr_d, pd_->OD(), pd_->ID());
                count_bwd_counting_range(
                        rsp + 24, oh, reg_curr_h, pd_->OH(), pd_->IH());
                count_bwd_counting_range(
                        rsp, ow, reg_curr_w, pd_->OW(), pd_->IW());
            }
        }

        // Choose algorithm
        if (pd_->desc()->alg_kind == alg_kind::resampling_linear) {
            if (pd_->ndims() == 5) {
                trilinear();
            } else if (pd_->ndims() == 4) {
                bilinear();
            } else {
                linear();
            }
        } else {
            nearest();
        }

        if (!pd_->is_fwd()) add(rsp, stack_size_needed_);
        postamble();
    }

    void count_dim_coeff(const Xmm &xmm_coeff, const Reg64 &reg_dim,
            dim_t y_max, dim_t x_max) {
        // Formula = ((y + 0.5f) * x_max / y_max) - 0.5f
        vcvtsi2ss(xmm_coeff, xmm_coeff, reg_dim); // y
        vaddss(xmm_coeff, xmm_coeff, xmm_zero_point_five); // y + 0.5f

        move_imm_float_to_xmm(xmm_tmp_factor, reg_tmp, (float)x_max);
        vmulss(xmm_coeff, xmm_coeff,
                xmm_tmp_factor); // (y + 0.5f) * x_max
        move_imm_float_to_xmm(xmm_tmp_factor, reg_tmp, (float)y_max);
        vdivss(xmm_coeff, xmm_coeff,
                xmm_tmp_factor); // (y + 0.5f) * x_max / y_max

        vsubss(xmm_coeff, xmm_coeff,
                xmm_zero_point_five); // ((y + 0.5) * x_max / y_max) - 0.5
    }

    void count_bwd_counting_range(RegExp stack_position,
            bwd_counting_range_t &c_range, const Reg64 &curr_position,
            dim_t y_max, dim_t x_max) {
        c_range.loop_counter = stack_position;
        if (pd_->desc()->alg_kind == alg_kind::resampling_linear) {
            c_range.start.linear[0] = stack_position + 8;
            c_range.end.linear[0] = stack_position + 16;
            c_range.start.linear[1] = stack_position + 24;
            c_range.end.linear[1] = stack_position + 32;
        } else {
            c_range.start.nearest = stack_position + 8;
            c_range.end.nearest = stack_position + 16;
        }

        EvexModifierRounding rm_ceil(EvexModifierRounding::T_RU_SAE);
        EvexModifierRounding rm_floor(EvexModifierRounding::T_RD_SAE);

        if (pd_->desc()->alg_kind == alg_kind::resampling_linear) {
            // coeff = (pos + 0.5) * y_max / x_max - 0.5
            count_dim_coeff(xmm_coeff, curr_position, x_max, y_max);

            // l_start: x == 0 ? 0 : ceil(coeff)
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_ceil);
            mov(reg_tmp, 0);
            cmp(curr_position, reg_tmp);
            cmove(reg_tmp_idx, reg_tmp);
            mov(ptr[c_range.start.linear[0]], reg_tmp_idx);

            // r_end: x == x_max-1 ? y_max : min(max(0, floor(coeff) + 1), y_max)
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_floor);
            add(reg_tmp_idx, 1);
            max(reg_tmp_idx, 0);
            min(reg_tmp_idx, y_max);
            cmp(curr_position, x_max - 1);
            mov(reg_tmp, y_max);
            cmove(reg_tmp_idx, reg_tmp);
            mov(ptr[c_range.end.linear[1]], reg_tmp_idx);

            // coeff = ((pos-1) + 0.5) * y_max / x_max - 0.5
            sub(curr_position, 1);
            count_dim_coeff(xmm_coeff, curr_position, x_max, y_max);

            // r_start: max(0, floor(coeff) + 1)
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_floor);
            add(reg_tmp_idx, 1);
            max(reg_tmp_idx, 0);
            mov(ptr[c_range.start.linear[1]], reg_tmp_idx);

            // coeff = ((pos+1) + 0.5) * y_max / x_max - 0.5
            add(curr_position, 2);
            count_dim_coeff(xmm_coeff, curr_position, x_max, y_max);

            // l_end: min(ceil(coeff), y_max)
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_ceil);
            min(reg_tmp_idx, y_max);
            mov(ptr[c_range.end.linear[0]], reg_tmp_idx);
        } else {
            float factor = (float)y_max / x_max;

            // start: ceil(pos * factor - 0.5f)
            vcvtsi2ss(xmm_coeff, xmm_coeff, curr_position);
            move_imm_float_to_xmm(xmm_tmp_factor, reg_tmp, factor);
            vmulss(xmm_coeff, xmm_coeff, xmm_tmp_factor);
            vsubss(xmm_coeff, xmm_coeff, xmm_zero_point_five);
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_ceil);
            mov(ptr[c_range.start.nearest], reg_tmp_idx);

            // start: ceil((pos+1) * factor - 0.5f)
            add(curr_position, 1);
            vcvtsi2ss(xmm_coeff, xmm_coeff, curr_position);
            vmulss(xmm_coeff, xmm_coeff, xmm_tmp_factor);
            vsubss(xmm_coeff, xmm_coeff, xmm_zero_point_five);
            vcvtss2si(reg_tmp_idx, xmm_coeff | rm_ceil);
            mov(ptr[c_range.end.nearest], reg_tmp_idx);
        }
    }

    void load_data(const Zmm &zmm_to_load, const Reg64 &src_address,
            const Reg64 &offset, bool is_tail) {
        Reg64 reg_address = reg_tmp;

        mov(reg_address, src_address);
        add(reg_address, offset);

        if (is_bf16()) {
            Zmm zmm_loaded_data = is_tail
                    ? zmm_to_load | k_tail_mask | Xbyak::util::T_z
                    : zmm_to_load;
            vpmovzxwd(zmm_loaded_data, ptr[reg_address]);
            vpslld(zmm_loaded_data, zmm_loaded_data, 16);
        } else {
            Zmm zmm_loaded_data
                    = is_tail ? zmm_to_load | k_tail_mask : zmm_to_load;
            vmovups(zmm_loaded_data, ptr[reg_address]);
        }
    }

    void store_data(const Zmm &zmm_to_store, const Reg64 &dst_address,
            size_t offset, bool is_tail) {
        if (is_bf16()) {
            Ymm ymm_bf16_to_store = Ymm(zmm_to_store.getIdx());

            if (use_bf16_emulation_) {
                bf16_emulation_->vcvtneps2bf16(ymm_bf16_to_store, zmm_to_store);
            } else {
                vcvtneps2bf16(ymm_bf16_to_store, zmm_to_store);
            }

            if (!is_tail) {
                vmovups(ptr[reg_dst + offset], ymm_bf16_to_store);
            } else {
                vmovdqu16(
                        ptr[reg_dst + offset] | k_tail_mask, ymm_bf16_to_store);
            }
        } else {
            if (is_tail) {
                vmovups(ptr[reg_dst + offset] | k_tail_mask, zmm_to_store);
            } else {
                vmovups(ptr[reg_dst + offset], zmm_to_store);
            }
        }
    }

    void count_idx_and_weight_for_linear(const Xmm &coeff, const Zmm &weight,
            const Reg64 &idx, dim_t dim_max, rounding_mode rm) {
        Reg64 reg_idx_floor;
        if (pd_->is_fwd() && rm == rounding_mode::ceil) {
            EvexModifierRounding rm_ceil(EvexModifierRounding::T_RU_SAE);
            EvexModifierRounding rm_floor(EvexModifierRounding::T_RD_SAE);
            vcvtss2si(idx,
                    coeff | rm_ceil); // ceil(coeff)
            reg_idx_floor = reg_tmp;
            vcvtss2si(reg_idx_floor,
                    coeff | rm_floor); // floor(coeff)

        } else {
            EvexModifierRounding rm_floor(EvexModifierRounding::T_RD_SAE);
            vcvtss2si(idx,
                    coeff | rm_floor); // floor(coeff)
            reg_idx_floor = idx;
        }

        vcvtsi2ss(cvt_reg<Xmm>(zmm_tmp), cvt_reg<Xmm>(zmm_tmp), reg_idx_floor);
        vsubss(cvt_reg<Xmm>(weight), coeff,
                zmm_tmp); // W = coeff - idx
        if (rm == rounding_mode::floor) {
            move_imm_float_to_xmm(cvt_reg<Xmm>(zmm_tmp), reg_tmp, 1.0f);
            vsubss(cvt_reg<Xmm>(weight), cvt_reg<Xmm>(zmm_tmp),
                    cvt_reg<Xmm>(weight)); // W = 1 - (coeff - idx)
        }
        vbroadcastss(weight, cvt_reg<Xmm>(weight));

        if (pd_->is_fwd()) {
            if (rm == rounding_mode::ceil) {
                min(idx, dim_max - 1);
            } else if (rm == rounding_mode::floor) {
                max(idx, 0);
            }
        }
    }

    void linear_alg(int64_t channel_offset, rounding_mode rm_w,
            rounding_mode rm_h = rounding_mode::none,
            rounding_mode rm_d = rounding_mode::none, bool is_tail = false) {
        xor_(reg_offset, reg_offset); // reg_offset = 0

        if (rm_w != rounding_mode::none) {
            // out: Ww, curr_w
            count_idx_and_weight_for_linear(
                    xmm_w_coeff, zmm_weight, reg_curr_w, pd_->IW(), rm_w);
            // curr_w * stride_w_
            if (!pd_->is_fwd()) mov(reg_curr_w, ptr[ow.loop_counter]);
            imul(reg_offset, reg_curr_w, stride_w_);
        }
        if (rm_h != rounding_mode::none) {
            // out: Wh, curr_h
            count_idx_and_weight_for_linear(
                    xmm_h_coeff, zmm_tmp_weight, reg_curr_h, pd_->IH(), rm_h);
            // Ww * Wh
            vmulps(zmm_weight, zmm_weight, zmm_tmp_weight);
            // curr_w * stride_w_ + curr_h * stride_h_
            if (!pd_->is_fwd()) mov(reg_curr_h, ptr[oh.loop_counter]);
            imul(reg_tmp, reg_curr_h, stride_h_);
            add(reg_offset, reg_tmp);
        }
        if (rm_d != rounding_mode::none) {
            // out: Wd, curr_d
            count_idx_and_weight_for_linear(
                    xmm_d_coeff, zmm_tmp_weight, reg_curr_d, pd_->ID(), rm_d);
            // Ww * Wh * Wd
            vmulps(zmm_weight, zmm_weight, zmm_tmp_weight);
            // curr_w * stride_w_ + curr_h * stride_h_ + curr_d * stride_d_
            if (!pd_->is_fwd()) mov(reg_curr_d, ptr[od.loop_counter]);
            imul(reg_tmp, reg_curr_d, stride_d_);
            add(reg_offset, reg_tmp);
        }

        add(reg_offset, channel_offset);
        imul(reg_offset, reg_offset, dtype_size());

        // read src
        load_data(zmm_src, reg_src, reg_offset, is_tail);

        // mul src, weight
        vmulps(zmm_tmp, zmm_src, zmm_weight);
        vaddps(zmm_dst, zmm_dst, zmm_tmp);
    }

    void linear() {
        int64_t channel_offset = 0;

        auto resample_linear = ([&](bool is_tail) {
            auto call_linear = ([&](int i) {
                linear_alg(channel_offset,
                        i % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_w */,
                        rounding_mode::none /* rounding_mode_h */,
                        rounding_mode::none /* rounding_mode_d */, is_tail);
            });

            // zero dst
            vpxorq(zmm_dst, zmm_dst, zmm_dst);

            if (pd_->is_fwd()) {
                for (int i = 0; i < 2; i++) {
                    call_linear(i);
                }
            } else {
                Label label[2][2];

                for (int i = 0; i < 2; i++) {
                    // for (dim_t ow = w.start[i]; ow < w.end[i]; ow++)
                    for_begin(label[i][0], label[i][1], ow.loop_counter,
                            ow.start.linear[i], ow.end.linear[i], reg_tmp);
                    count_dim_coeff(xmm_w_coeff, reg_tmp, pd_->OW(), pd_->IW());

                    call_linear(i + 1);

                    for_end(label[i][0], label[i][1], ow.loop_counter, reg_tmp);
                }
            }

            // store dst
            store_data(
                    zmm_dst, reg_dst, channel_offset * dtype_size(), is_tail);
        });

        for (unsigned i = 0; i < number_of_loops_;
                i++, channel_offset += simd_w())
            resample_linear(false);

        if (tail_mask_ != 0) resample_linear(true);
    }

    void bilinear() {
        int64_t channel_offset = 0;

        auto resample_linear = ([&](bool is_tail) {
            auto call_linear = ([&](int i, int j) {
                linear_alg(channel_offset,
                        i % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_w */,
                        j % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_h */,
                        rounding_mode::none /* rounding_mode_d */, is_tail);
            });

            // zero dst
            vpxorq(zmm_dst, zmm_dst, zmm_dst);

            if (pd_->is_fwd()) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        call_linear(i, j);
                    }
                }
            } else {
                Label label[2][2][4];

                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        // for (dim_t ow = w.start[i]; ow < w.end[i]; ow++)
                        for_begin(label[i][j][0], label[i][j][1],
                                ow.loop_counter, ow.start.linear[i],
                                ow.end.linear[i], reg_tmp);
                        count_dim_coeff(
                                xmm_w_coeff, reg_tmp, pd_->OW(), pd_->IW());
                        // for (dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                        for_begin(label[i][j][2], label[i][j][3],
                                oh.loop_counter, oh.start.linear[j],
                                oh.end.linear[j], reg_tmp);
                        count_dim_coeff(
                                xmm_h_coeff, reg_tmp, pd_->OH(), pd_->IH());

                        call_linear(i + 1, j + 1);

                        for_end(label[i][j][2], label[i][j][3], oh.loop_counter,
                                reg_tmp);
                        for_end(label[i][j][0], label[i][j][1], ow.loop_counter,
                                reg_tmp);
                    }
                }
            }

            // store dst
            store_data(
                    zmm_dst, reg_dst, channel_offset * dtype_size(), is_tail);
        });

        for (unsigned i = 0; i < number_of_loops_;
                i++, channel_offset += simd_w())
            resample_linear(false);

        if (tail_mask_ != 0) resample_linear(true);
    }

    void trilinear() {
        int64_t channel_offset = 0;

        auto resample_linear = ([&](bool is_tail) {
            auto call_linear = ([&](int i, int j, int k) {
                linear_alg(channel_offset,
                        i % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_w */,
                        j % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_h */,
                        k % 2 ? rounding_mode::floor
                              : rounding_mode::ceil /* rounding_mode_d */,
                        is_tail);
            });

            // zero dst
            vpxorq(zmm_dst, zmm_dst, zmm_dst);

            if (pd_->is_fwd()) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        for (int k = 0; k < 2; k++) {
                            call_linear(i, j, k);
                        }
                    }
                }
            } else {
                Label label[2][2][2][6];

                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        for (int k = 0; k < 2; k++) {
                            // for (dim_t ow = w.start[i]; ow < w.end[i]; ow++)
                            for_begin(label[i][j][k][0], label[i][j][k][1],
                                    ow.loop_counter, ow.start.linear[i],
                                    ow.end.linear[i], reg_tmp);
                            count_dim_coeff(
                                    xmm_w_coeff, reg_tmp, pd_->OW(), pd_->IW());
                            // for (dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                            for_begin(label[i][j][k][2], label[i][j][k][3],
                                    oh.loop_counter, oh.start.linear[j],
                                    oh.end.linear[j], reg_tmp);
                            count_dim_coeff(
                                    xmm_h_coeff, reg_tmp, pd_->OH(), pd_->IH());
                            // for (dim_t od = d.start[k]; od < d.end[k]; od++)
                            for_begin(label[i][j][k][4], label[i][j][k][5],
                                    od.loop_counter, od.start.linear[k],
                                    od.end.linear[k], reg_tmp);
                            count_dim_coeff(
                                    xmm_d_coeff, reg_tmp, pd_->OD(), pd_->ID());

                            call_linear(i + 1, j + 1, k + 1);

                            for_end(label[i][j][k][4], label[i][j][k][5],
                                    od.loop_counter, reg_tmp);
                            for_end(label[i][j][k][2], label[i][j][k][3],
                                    oh.loop_counter, reg_tmp);
                            for_end(label[i][j][k][0], label[i][j][k][1],
                                    ow.loop_counter, reg_tmp);
                        }
                    }
                }
            }

            // store dst
            store_data(
                    zmm_dst, reg_dst, channel_offset * dtype_size(), is_tail);
        });

        for (unsigned i = 0; i < number_of_loops_;
                i++, channel_offset += simd_w())
            resample_linear(false);

        if (tail_mask_ != 0) resample_linear(true);
    }

    void nearest_alg(int64_t channel_offset, bool is_tail = false) {
        xor_(reg_offset, reg_offset); // reg_offset = 0

        auto get_idx = ([&](const Reg64 &idx, const Xmm &coeff,
                                dim_t dim_max_size) {
            round_to_near_away_from_zero(idx, coeff, xmm_zero_point_five,
                    cvt_reg<Xmm>(zmm_tmp)); // round_to_nearest(coeff)
            min(idx, dim_max_size - 1);
            max(idx, 0);
        });

        if (pd_->is_fwd()) {
            get_idx(reg_curr_w, xmm_w_coeff, pd_->IW());
            get_idx(reg_curr_h, xmm_h_coeff, pd_->IH());
            get_idx(reg_curr_d, xmm_d_coeff, pd_->ID());
        } else {
            mov(reg_curr_w, ptr[ow.loop_counter]);
            mov(reg_curr_h, ptr[oh.loop_counter]);
            mov(reg_curr_d, ptr[od.loop_counter]);
        }

        imul(reg_offset, reg_curr_w, stride_w_); // iw * stride_w_
        imul(reg_tmp, reg_curr_h, stride_h_);
        add(reg_offset, reg_tmp); // iw * stride_w_ + ih * stride_h_
        imul(reg_tmp, reg_curr_d, stride_d_);
        add(reg_offset,
                reg_tmp); // iw * stride_w_ + ih * stride_h_ + id * stride_d_

        add(reg_offset,
                channel_offset); // iw * stride_w_ + ih * stride_h_ + id * stride_d_ + channel_offset
        imul(reg_offset, reg_offset,
                dtype_size()); // (iw * stride_w_ + ih * stride_h_ + id * stride_d_ + channel_offset)*dt_size

        if (pd_->is_fwd()) {
            // read nearest to dst
            load_data(zmm_dst, reg_src, reg_offset, is_tail);
        } else {
            // add nearest to dst
            load_data(zmm_tmp, reg_src, reg_offset, is_tail);
            vaddps(zmm_dst, zmm_dst, zmm_tmp);
        }
    }

    void nearest() {
        int64_t channel_offset = 0;

        auto resample_nearest = ([&](bool is_tail) {
            // zero dst
            vpxorq(zmm_dst, zmm_dst, zmm_dst);

            if (pd_->is_fwd()) {
                nearest_alg(channel_offset, is_tail);
            } else {
                Label label[6];

                // for (dim_t ow = w.start[i]; ow < w.end[i]; ow++)
                for_begin(label[0], label[1], ow.loop_counter, ow.start.nearest,
                        ow.end.nearest, reg_tmp);
                // for (dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                for_begin(label[2], label[3], oh.loop_counter, oh.start.nearest,
                        oh.end.nearest, reg_tmp);
                // for (dim_t od = d.start[k]; od < d.end[k]; od++)
                for_begin(label[4], label[5], od.loop_counter, od.start.nearest,
                        od.end.nearest, reg_tmp);

                nearest_alg(channel_offset, is_tail);

                for_end(label[4], label[5], od.loop_counter, reg_tmp);
                for_end(label[2], label[3], oh.loop_counter, reg_tmp);
                for_end(label[0], label[1], ow.loop_counter, reg_tmp);
            }

            // store dst
            store_data(
                    zmm_dst, reg_dst, channel_offset * dtype_size(), is_tail);
        });

        for (unsigned i = 0; i < number_of_loops_;
                i++, channel_offset += simd_w())
            resample_nearest(false);

        if (tail_mask_ != 0) resample_nearest(true);
    }

    int vlen() {
        int vlen = cpu_isa_traits<avx512_common>::vlen;
        return is_bf16() ? vlen / 2 : vlen;
    }
    int simd_w() { return vlen() / dtype_size(); }

    /* bf16 emulator registers */
    Zmm bf16_emu_reserv_1 = Zmm(26);
    Zmm bf16_emu_reserv_2 = Zmm(27);
    Zmm bf16_emu_reserv_3 = Zmm(28);
    Reg64 bf16_emu_scratch = r14;
    Zmm bf16_emu_reserv_5 = Zmm(29);
    /* ----------------------- */

    /* registers used */
    Zmm zmm_src = Zmm(1);
    Ymm ymm_dst = Ymm(2);
    Zmm zmm_dst = Zmm(2);
    Zmm zmm_weight = Zmm(3);
    Xmm xmm_coeff = Xmm(4);
    Xmm xmm_d_coeff = Xmm(4);
    Xmm xmm_h_coeff = Xmm(5);
    Xmm xmm_w_coeff = Xmm(6);
    Xmm xmm_zero_point_five = Xmm(7);
    Zmm zmm_tmp = Zmm(8);
    Zmm zmm_tmp_weight = Zmm(9);
    Xmm xmm_tmp_factor = Xmm(9);

    Opmask k_tail_mask = k6;
    Reg64 reg_src = rax;
    Reg64 reg_dst = rbx;
    Reg64 reg_tmp = r8;
    Reg64 reg_curr_d = r9;
    Reg64 reg_curr_h = r10;
    Reg64 reg_curr_w = r11;
    Reg64 reg_offset = r12;
    Reg64 reg_tmp_idx = r12;
    /* -------------- */

    /* additional instruction builders */
    std::unique_ptr<bf16_emulation_t> bf16_emulation_;
    /* ------------------------------- */

    bwd_counting_range_t ow;
    bwd_counting_range_t oh;
    bwd_counting_range_t od;

    dim_t stride_d_;
    dim_t stride_h_;
    dim_t stride_w_;
    dim_t inner_stride_;
    unsigned number_of_loops_;
    size_t tail_mask_;
    bool use_bf16_emulation_;
    unsigned stack_size_needed_;
};

} // namespace

template <data_type_t d_type>
status_t jit_avx512_common_resampling_fwd_t<d_type>::pd_t::init(
        engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;
    bool ok = mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && is_fwd()
            && !has_zero_dim_memory()
            && utils::everyone_is(
                    d_type, src_md()->data_type, dst_md()->data_type)
            && set_default_params() == status::success
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    format_tag_t dat_tag = memory_desc_matches_one_of_tag(*src_md(), nCw8c,
            nChw8c, nCdhw8c, nCw16c, nChw16c, nCdhw16c, nwc, nhwc, ndhwc);
    if (!memory_desc_matches_tag(*dst_md(), dat_tag))
        return status::unimplemented;

    return status::success;
}

template <impl::data_type_t d_type>
inline jit_avx512_common_resampling_fwd_t<
        d_type>::jit_avx512_common_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <impl::data_type_t d_type>
status_t jit_avx512_common_resampling_fwd_t<d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_avx512_common_resampling_t(pd())));
    return kernel_->create_kernel();
}

template <impl::data_type_t d_type>
jit_avx512_common_resampling_fwd_t<
        d_type>::~jit_avx512_common_resampling_fwd_t()
        = default;

template <impl::data_type_t d_type>
status_t jit_avx512_common_resampling_fwd_t<d_type>::execute(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();

    const memory_desc_wrapper src_d(pd()->src_md());
    const dim_t inner_stride = src_d.blocking_desc().strides[pd()->ndims() - 1];
    const dim_t nsp_outer = src_d.nelems(true) / (ID * IH * IW * inner_stride);

    parallel_nd(nsp_outer, OD, OH, OW,
            [&](dim_t nsp, dim_t od, dim_t oh, dim_t ow) {
                dim_t src_off = nsp * ID * IH * IW * inner_stride;
                dim_t dst_off
                        = (nsp * OD * OH * OW + od * OH * OW + oh * OW + ow)
                        * inner_stride;
                jit_resampling_args_t args;
                args.src = src + src_off;
                args.dst = dst + dst_off;
                args.d = od;
                args.h = oh;
                args.w = ow;
                (*kernel_)(&args);
            });

    return status_t();
}

template struct jit_avx512_common_resampling_fwd_t<data_type::bf16>;
template struct jit_avx512_common_resampling_fwd_t<data_type::f32>;

template <data_type_t d_type>
status_t jit_avx512_common_resampling_bwd_t<d_type>::pd_t::init(
        engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;
    bool ok = mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && !is_fwd()
            && !has_zero_dim_memory()
            && utils::everyone_is(
                    d_type, diff_src_md()->data_type, diff_dst_md()->data_type)
            && set_default_params() == status::success
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    format_tag_t dat_tag = memory_desc_matches_one_of_tag(*diff_src_md(), nCw8c,
            nChw8c, nCdhw8c, nCw16c, nChw16c, nCdhw16c, nwc, nhwc, ndhwc);
    if (!memory_desc_matches_tag(*diff_dst_md(), dat_tag))
        return status::unimplemented;

    return status::success;
}

template <impl::data_type_t d_type>
inline jit_avx512_common_resampling_bwd_t<
        d_type>::jit_avx512_common_resampling_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <impl::data_type_t d_type>
status_t jit_avx512_common_resampling_bwd_t<d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_avx512_common_resampling_t(pd())));
    return kernel_->create_kernel();
}

template <impl::data_type_t d_type>
jit_avx512_common_resampling_bwd_t<
        d_type>::~jit_avx512_common_resampling_bwd_t()
        = default;

template <impl::data_type_t d_type>
status_t jit_avx512_common_resampling_bwd_t<d_type>::execute(
        const exec_ctx_t &ctx) const {

    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const dim_t inner_stride
            = diff_src_d.blocking_desc().strides[pd()->ndims() - 1];
    const dim_t nsp_outer
            = diff_src_d.nelems(true) / (ID * IH * IW * inner_stride);

    parallel_nd(nsp_outer, ID, IH, IW,
            [&](dim_t nsp, dim_t id, dim_t ih, dim_t iw) {
                dim_t diff_dst_off = nsp * OD * OH * OW * inner_stride;
                dim_t diff_src_off
                        = (nsp * ID * IH * IW + id * IH * IW + ih * IW + iw)
                        * inner_stride;
                jit_resampling_args_t args;
                args.src = diff_dst + diff_dst_off;
                args.dst = diff_src + diff_src_off;
                args.d = id;
                args.h = ih;
                args.w = iw;
                (*kernel_)(&args);
            });

    return status_t();
}

template struct jit_avx512_common_resampling_bwd_t<data_type::bf16>;
template struct jit_avx512_common_resampling_bwd_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl