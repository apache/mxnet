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

#ifndef CPU_X64_JIT_UNI_X8S8S32X_CONV_KERNEL_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, typename Vmm>
struct _jit_uni_x8s8s32x_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_uni_x8s8s32x_conv_fwd_ker_t_)

    _jit_uni_x8s8s32x_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_
                    = new jit_uni_eltwise_injector_f32<isa>(this, jcp.eltwise);
    }
    ~_jit_uni_x8s8s32x_fwd_kernel() { delete eltwise_injector_; }

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;

    enum {
        typesize = sizeof(float),
        ic_sub_step = 4,
        ker_zp_reg_base_idx = 9,
        ker_reg_base_idx = 12,
        ker_dw_reg_base_idx = 14,
        ker_max_reg = 15,
    };
    enum ic_block_t {
        no_last_block,
        last_ic_block,
        last_sp_block,
    };

    /* data registers */
    const Xbyak::Reg64 reg_ptr_scales = rax;
    const Xbyak::Reg64 reg_ptr_saturation_ubound = rax;
    const Xbyak::Reg64 reg_inp = r8;
    const Xbyak::Reg64 reg_ker = r9;
    const Xbyak::Reg64 reg_out = r10;
    const Xbyak::Reg64 aux_reg_inp = r11;
    const Xbyak::Reg64 reg_ptr_sum_scale = r11;
    const Xbyak::Reg64 aux_reg_ker = r12;
    const Xbyak::Reg64 aux_reg_inp_d = r13;
    const Xbyak::Reg64 reg_compensation = r14;
    const Xbyak::Reg64 aux_reg_ker_d = r15;

    /* counter regs */
    const Xbyak::Reg64 reg_bias_alpha = abi_not_param1;
    const Xbyak::Reg64 reg_oi = rbx;
    const Xbyak::Reg64 reg_bias = rdx;
    const Xbyak::Reg64 reg_oc_blocks = rsi;
    const Xbyak::Reg64 reg_owb = aux_reg_ker;
    const Xbyak::Reg64 reg_scratch = reg_compensation;
    const Xbyak::Reg64 reg_ki = reg_compensation;
    const Xbyak::Reg64 reg_kj = reg_ptr_scales;
    const Xbyak::Reg64 reg_overflow = reg_ptr_scales;
    const Xbyak::Reg64 reg_icb = reg_bias;
    // Using 3d regs as depthwise3d is not yet supported
    const Xbyak::Reg64 reg_inp_buffer_ptr = aux_reg_inp_d;
    const Xbyak::Reg64 aux_reg_inp_buffer_ptr = aux_reg_ker_d;
    const Xbyak::Reg64 reg_jmp_tbl_base = reg_kj;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = aux_reg_inp;
    const Xbyak::Reg64 reg_src_zero_point = aux_reg_ker_d;
    const Xbyak::Reg64 reg_dst_zero_point = reg_src_zero_point;

    const Vmm vmm_wei = Vmm(0);
    /* used during bias section of store_output */
    const Vmm vmm_comp = Vmm(1); // only for signed input
    const Vmm vmm_bias = Vmm(0);
    /* used during post_op sum section of store_output */
    const Vmm vmm_prev_dst = Vmm(0);
    /* used during write-out section of store_output */
    const Vmm vmm_zero = Vmm(0);
    const Vmm vmm_saturation = Vmm(0);
    /* used for zero-point */
    const Vmm vmm_zp = Vmm(6);
    const Vmm vmm_zp_one = Vmm(5);
    const Vmm vmm_zp_comp = vmm_zp_one;
    const Vmm vmm_zp_dw_tmp = vmm_zp_one;

    /* used in compute_ker (but set during prepare_output) */
    const Vmm vmm_shift = vmm_comp; // only for signed input
    /* used in compute_ker */
    const Vmm vmm_tmp = Vmm(3); // not used for depthwise
    const Vmm vmm_one
            = Vmm(2); // set at start of kernel, not used for depthwise.
    /* used during scale section of store output */
    const Vmm vmm_scale = vmm_tmp;
    /* registers use only for depthwise
     * groups are always blocked by 8/4 (padded if needed),
     * hence use only Ymm registers for avx2
     * use Xmm registers for sse41*/
    const Vmm vmm_dw_wei = Vmm(0);
    Vmm vmm_dw_tmp;
    Vmm vmm_dw_src;
    Vmm vmm_dw_shifted_zero;

    Vmm vmm_out(int i_ur, int i_oc) {
        const int idx_limit = jcp.src_zero_point
                ? ker_zp_reg_base_idx
                : (jcp.is_depthwise ? ker_dw_reg_base_idx : ker_reg_base_idx);
        const int nb_x_blocking
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        const int idx = i_ur * nb_x_blocking + i_oc;
        assert(idx < idx_limit);
        MAYBE_UNUSED(idx_limit);
        /* remap register indices from 4 to 15
         * to avoid passing xmm0 to comp*/
        return Vmm(ker_max_reg - idx);
    }
    Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < ker_max_reg);
        return Vmm(ker_max_reg - idx);
    }
    Vmm vmm_bias_alpha() {
        int nb_c_block
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return Vmm(ker_max_reg - nb_c_block * jcp.ur_w);
    }
    Xbyak::Xmm xmm_bias_alpha() {
        int nb_c_block
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return Xbyak::Xmm(ker_max_reg - nb_c_block * jcp.ur_w);
    }
    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }
    int get_blocking_size() {
        return jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;
    }
    int get_tail_size() {
        return jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                                : jcp.oc_without_padding % jcp.oc_block;
    }

    bool maybe_eltwise(int position);
    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block_flag);
    void compute_ker_dw(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool h_padded);
    void compute_ker(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool h_padded = false);
    void compute_eltwise(int ur_w);
    void kh_loop(int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool is_last_spatial_block);
    void generate() override;

    void cvt2ps(data_type_t type_in, const Vmm &vmm_in, const Xbyak::Reg64 &reg,
            int offset, int load_size);
};

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_fwd_kernel {

    jit_uni_x8s8s32x_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : kernel_(nullptr) {
        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
        switch (ch_block) {
            case 8:
                if (utils::one_of(isa, avx2)) {
                    kernel_ = new _jit_avx2_x8s8s32x_fwd_kernel(ajcp, attr);
                    return;
                } else
                    assert(!"invalid channel blocking for current ISA");
            case 4:
                kernel_ = new _jit_uni_x8s8s32x_fwd_kernel<isa, Xbyak::Xmm>(
                        ajcp, attr);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() { return kernel_->create_kernel(); }

    ~jit_uni_x8s8s32x_fwd_kernel() { delete kernel_; }

    void operator()(const jit_conv_call_s *p) const { (*kernel_)(p); }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void (*jit_ker)(jit_conv_call_s *);
    using _jit_avx2_x8s8s32x_fwd_kernel
            = _jit_uni_x8s8s32x_fwd_kernel<avx2, Xbyak::Ymm>;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_x8s8s32x_fwd_kernel<isa>);
    jit_generator *kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
