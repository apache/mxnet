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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_1x1_fwd_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_1x1_fwd_kernel_t)

    jit_avx512_core_amx_1x1_fwd_kernel_t(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
    }
    ~jit_avx512_core_amx_1x1_fwd_kernel_t() { delete eltwise_injector_; }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    // Tile-registers decomposition
    enum { C_BASE = 0, W_BASE = 6, I_BASE = 4 };

    void tile_configure(char *tcgf_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    int row_count_;
    int buf_count_;
    bool is_store_done_;
    bool is_buffer_empty_;
    bool check_last_sb_;
    bool last_oc_block_flag_;

    /* data regs */
    Xbyak::Reg64 inp_ptr = r15;
    Xbyak::Reg64 wei_ptr = r14;
    Xbyak::Reg64 out_ptr = r13;
    Xbyak::Reg64 wsp_ptr = r12;

    Xbyak::Reg64 reg_bias = r11;
    Xbyak::Reg64 reg_ptr_scales = r10;
    Xbyak::Reg64 reg_ptr_sum_scale = r9;
    Xbyak::Reg64 aux_reg_saturation = reg_ptr_sum_scale;
    Xbyak::Reg64 reg_last_h = r8;
    Xbyak::Reg64 reg_tail = rax;

    Xbyak::Reg64 stride_seq = rbx;
    Xbyak::Reg64 stride_nhwc = rsi;
    Xbyak::Reg64 reg_tmp = abi_not_param1;

    Xbyak::Reg64 reg_oc_blocks = rdx;
    Xbyak::Reg64 reg_is_osb = rsi;
    Xbyak::Reg64 reg_postop = abi_not_param1;
    Xbyak::Reg64 reg_scratch = reg_bias;
    Xbyak::Reg64 reg_tilebuff = reg_ptr_scales;

    Xbyak::Zmm zmm_bias = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_saturation = zmm_bias;
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_prev_dst = Xbyak::Zmm(29);

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    bool is_bf16() const;

    void init_runtime_counters();

    int get_out_tensor(int h, int i) const;
    int get_inp_tensor(int h) const;
    int get_wei_tensor(int i) const;
    int get_ic_tail() const;

    size_t out_h_shift() const;
    size_t out_w_shift() const;
    size_t inp_offset(int ih, int iw, int icb) const;
    size_t out_row_offset(int h, int w, int ocb) const;

    void prepare_output();

    bool maybe_eltwise(int position);
    void cvt2ps(data_type_t type_in, Xbyak::Zmm ymm_in,
            const Xbyak::Operand &op, bool mask_flag);
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm zmm_in, bool mask_flag, bool store = false);
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm ymm_in, bool mask_flag, bool store = false);

    void update_buffer_pointers();
    void interleave_store();
    void store_output_vector_int8(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output_vector_bf16(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output_vector(const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output(bool do_store, bool is_tail);
    void icb_loop(bool do_store);
    void osb_loop(int nb_os = 1);

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
