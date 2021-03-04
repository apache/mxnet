/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Vmm>
struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_x8s8s32x_1x1_conv_fwd_ker_t)
    _jit_avx512_core_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);
    }

    ~_jit_avx512_core_x8s8s32x_1x1_conv_kernel() { delete eltwise_injector_; }

    bool maybe_eltwise(int position);
    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;

    /* register mapping */
    const Xbyak::Reg64 reg_last_load = r8;
    const Xbyak::Reg64 reg_bcast_data = r8;
    const Xbyak::Reg64 reg_ptr_scales = r8;
    const Xbyak::Reg64 reg_ptr_saturation_ubound = r8;
    const Xbyak::Reg64 reg_output_data = r9;
    const Xbyak::Reg64 reg_load_data = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r10;
    const Xbyak::Reg64 reg_reduce_loop_work = r11;
    const Xbyak::Reg64 reg_bias_data = r12;
    const Xbyak::Reg64 reg_comp_data = r12;
    const Xbyak::Reg64 reg_scratch = r13;
    const Xbyak::Reg64 aux_reg_bcast_data = r14;
    const Xbyak::Reg64 aux_reg_load_data = r15;
    const Xbyak::Reg64 imm_addr64 = r15;
    const Xbyak::Reg64 reg_reduce_pos_flag = rax;
    const Xbyak::Reg64 aux1_reg_bcast_data = rbx;
    const Xbyak::Reg64 reg_bcast_loop_work = rbx;
    const Xbyak::Reg64 bcast_loop_iter = rdx; // Note: Fix me
    const Xbyak::Reg64 reg_load_loop_work = rsi;
    const Xbyak::Reg64 aux_reg_output_data = abi_not_param1;
    const Xbyak::Reg64 reduce_loop_iter = abi_param1;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = aux_reg_load_data; // r15
    const Xbyak::Reg64 reg_src_zero_point = aux_reg_bcast_data; // r14
    const Xbyak::Reg64 reg_dst_zero_point = reg_src_zero_point;

    const Xbyak::Opmask ktail_mask = k6;
    const Xbyak::Opmask vmask = k7;

    const Vmm vmm_tmp = Vmm(28);
    const Vmm vmm_saturation = Vmm(28);
    const Vmm vmm_one = Vmm(29);
    const Vmm vmm_zero = Vmm(30);
    const Vmm vmm_prev_dst = Vmm(30);
    const Vmm vmm_shift = Vmm(30);
    const Vmm vmm_bcast = Vmm(31);
    const Vmm vmm_bias_alpha = Vmm(31);
    const Xbyak::Xmm xmm_bias_alpha = Xbyak::Xmm(31);
    /* zero-point */
    const Vmm vmm_zp = Vmm(30);
    const Vmm vmm_zp_tmp = vmm_zp;

    int bcast_loop_work_off = 0;
    int reg_bias_data_off = 8;
    int reg_bcast_data_off = 16;
    int reg_load_data_off = 24;
    int reg_ptr_sum_scale_off = 32;
    int reg_comp_data_off = 40;
    int reg_zp_compensation_off = 48;
    int reg_src_zero_point_off = 56;
    int reg_dst_zero_point_off = 64;
    int stack_space_needed = 72;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate() override;
    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag);
};

struct jit_avx512_core_x8s8s32x_1x1_conv_kernel {
    jit_avx512_core_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : kernel_(nullptr) {
        int ch_block = ajcp.ic_block;
        switch (ch_block) {
            case 16:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Zmm>(ajcp, attr);
                return;
            case 8:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Ymm>(ajcp, attr);
                return;
            case 4:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Xmm>(ajcp, attr);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() { return kernel_->create_kernel(); }

    ~jit_avx512_core_x8s8s32x_1x1_conv_kernel() { delete kernel_; }

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t *&src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads,
            bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    void operator()(const jit_1x1_conv_call_s *p) const { (*kernel_)(p); }
    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_x8s8s32x_1x1_conv_kernel);
    jit_generator *kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
