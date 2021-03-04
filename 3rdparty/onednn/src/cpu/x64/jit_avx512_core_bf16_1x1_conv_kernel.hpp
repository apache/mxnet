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

#ifndef CPU_X64_JIT_AVX512_CORE_BF16_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_BF16_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_bf16_1x1_conv_kernel : public jit_generator {
    jit_avx512_core_bf16_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, ker_code_size)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr)
        , bf16_emu_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);

        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);
    }

    ~jit_avx512_core_bf16_1x1_conv_kernel() {
        delete eltwise_injector_;
        delete bf16_emu_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_1x1_conv_kernel)

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    const jit_1x1_conv_conf_t &jcp;
    const primitive_attr_t &attr_;

private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;
    enum {
        ker_code_size = 1024 * 1024,
    };

    reg64_t aux_reg_load_data = r15;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t reg_output_stride = rsi;
    reg64_t reg_bias_data = r12;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t reg_bcast_data = r8;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reg_bcast_loop_iter = rdx;
    reg64_t reg_load_loop_work = r13;
    reg64_t reduce_loop_iter = abi_param1;
    reg64_t reg_load_dim_tail_mask = aux_reg_load_data;

    reg64_t imm_addr64 = aux_reg_load_data;
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    reg64_t reg_trans_tmp = reg_reduce_pos_flag;
    reg64_t reg_store_buf
            = reg_output_stride; // reg_output_stride used only in BWD/WU
    reg64_t aux_reg_store_buf = reg_load_loop_work;

    mask_t vmask = k7;
    // Used for axb tail handling.
    // k_load_dim_mask is dynamically updated with k_load_mask_tail_mask
    // whenever tail is detected
    mask_t k_load_dim_mask = Xbyak::Opmask(2);
    mask_t k_load_dim_mask_extended = Xbyak::Opmask(3);
    mask_t k_load_dim_tail_mask = Xbyak::Opmask(4);
    mask_t k_load_dim_tail_mask_extended = Xbyak::Opmask(5);

    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(30);
    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm vreg_bcast = Xbyak::Zmm(31);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(25);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(27);
    reg64_t bf16_emu_reserv_4 = imm_addr64;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(29);

    Xbyak::Zmm zmm_tmp2 = Xbyak::Zmm(30);

    Xbyak::Opmask full_mask = Xbyak::Opmask(7);
    Xbyak::Opmask half_mask = Xbyak::Opmask(6);
    Xbyak::Opmask half_mask_hi = Xbyak::Opmask(5);
    Xbyak::Label dst_prm_table;

    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;

    int bcast_loop_work_offt = 0;
    int reg_load_loop_work_off = 8;
    int perm_reg_offset = 16;
    int broadcast_space = 32;
    int stack_space_needed = 360;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);
    void compute_diff_bias(int load_loop_blk);

    void generate() override;
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);
    inline bool is_bcast_layout_nxc() {
        switch (jcp.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            case prop_kind::backward_data:
                return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            case prop_kind::backward_weights:
                return jcp.uses_permw_transposition
                        && utils::one_of(jcp.src_tag, format_tag::ndhwc,
                                format_tag::nhwc, format_tag::nwc);
            default: assert(!"invalid prop_kind"); return false;
        }
    }
    inline bool is_load_layout_nxc() {
        return jcp.prop_kind == prop_kind::backward_weights
                && jcp.uses_permw_transposition
                && utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
    }
    inline bool is_out_layout_nxc() {
        switch (jcp.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            case prop_kind::backward_data:
                return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
            case prop_kind::backward_weights: return false;
            default: assert(!"invalid prop_kind"); return false;
        }
    }

    inline Xbyak::Zmm may_be_mask_zmm(Xbyak::Zmm zmm, bool mask_flag,
            bool zero_mask, bool use_extended_mask = false) {
        if (mask_flag) {
            zmm = zmm
                    | (use_extended_mask ? k_load_dim_mask_extended
                                         : k_load_dim_mask);
            if (zero_mask) zmm = zmm | T_z;
        }
        return zmm;
    }

    inline Xbyak::Ymm may_be_mask_ymm(Xbyak::Ymm ymm, bool mask_flag) {
        return mask_flag ? ymm | k_load_dim_mask : ymm;
    }

    bf16_emulation_t *bf16_emu_;
};
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
