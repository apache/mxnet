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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd)

    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_lstm_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd) {}

    ~jit_uni_lstm_cell_postgemm_fwd() {
        delete sigmoid_injector_;
        delete tanh_injector_;
    }

    status_t init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        // we use rax for both constant tables and load correspondent label
        // into it when calling correspondent injector.
        sigmoid_injector_ = new injector_t(
                this, alg_kind::eltwise_logistic, 0.0f, 0.0f, 1.0f, true, rax);
        tanh_injector_ = new injector_t(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    injector_t *sigmoid_injector_;
    injector_t *tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    size_t cstate_dt_size = sizeof(float);
    size_t hstate_dt_size = types::data_type_size(src_data_t);
    size_t gate_dt_size = types::data_type_size(src_data_t);
    size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    size_t qscale_dt_size = sizeof(float);
    size_t weights_peephole_dt_size = sizeof(float);
    size_t bias_dt_size = sizeof(float);

    void generate() override {
        using namespace Xbyak;

        auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;

        // Register map
        Reg64 loop_cnt(rbx); // loop counter
        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        Vmm G0(1), G1(2), G2(3), G3(4), tmp1_vmm(5), tmp2_vmm(6), zero_vmm(7);

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_scratch_gates_reg = abi_param2;
        auto addr_weights_peephole_reg = r11;
        auto addr_bias_reg = abi_param3;
        auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_states_t_l_copy_reg = r10;
        auto addr_c_states_tm1_l_reg = rdi;
        auto addr_c_states_t_l_reg = rsi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 16]);
        mov(addr_weights_peephole_reg, ptr[base_args + 24]);
#else
        auto addr_states_t_l_copy_reg = abi_param5;
        auto addr_c_states_tm1_l_reg = abi_param6;
        auto addr_c_states_t_l_reg = r10;
        auto base_args = get_stack_params_address();
        mov(addr_c_states_t_l_reg, ptr[base_args]);
        mov(addr_weights_peephole_reg, ptr[base_args + 8]);
#endif

        // helper lambda to address the gates and biases
        auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };

        auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        auto weights_peephole_addr = [&](int i) {
            return ptr[addr_weights_peephole_reg
                    + i * rnn_.dhc * weights_peephole_dt_size];
        };
        auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size];
        };

        // initialize registers with addresses and constants
        init_regs(vlen);

        sigmoid_injector_->load_table_addr();
        tanh_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // load G0 G1 G2 G3
            uni_vmovups(G0, sg_addr(0));
            uni_vmovups(G1, sg_addr(1));
            uni_vmovups(G2, sg_addr(2));
            uni_vmovups(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8) {
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, true);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, true);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, true);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, true);
            }

            // add biases
            uni_vmovups(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(2));
            uni_vaddps(G2, G2, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(3));
            uni_vaddps(G3, G3, tmp1_vmm);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                uni_vmovups(tmp1_vmm, weights_peephole_addr(0));
                uni_vmovups(tmp2_vmm, ptr[addr_c_states_tm1_l_reg]);
                uni_vfmadd231ps(G0, tmp1_vmm, tmp2_vmm);
                uni_vmovups(tmp1_vmm, weights_peephole_addr(1));
                uni_vfmadd231ps(G1, tmp1_vmm, tmp2_vmm);
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(0), G0, vlen);
                to_src<src_data_t>(wg_addr(1), G1, vlen);
                to_src<src_data_t>(wg_addr(2), G2, vlen);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmovups(tmp1_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1_vmm, tmp1_vmm, G1);
            uni_vfmadd231ps(tmp1_vmm, G0, G2);
            uni_vmovups(ptr[addr_c_states_t_l_reg], tmp1_vmm);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                uni_vmovups(tmp1_vmm, weights_peephole_addr(2));
                uni_vmovups(tmp2_vmm, ptr[addr_c_states_t_l_reg]);
                uni_vfmadd231ps(G3, tmp1_vmm, tmp2_vmm);
            }

            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (is_training) { to_src<src_data_t>(wg_addr(3), G3, vlen); }

            // states_t_l = G3 * tanh(c_states_t_l)
            uni_vmovups(tmp1_vmm, ptr[addr_c_states_t_l_reg]);
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp1_vmm.getIdx());
            uni_vmulps(tmp1_vmm, tmp1_vmm, G3);

            // downconvert and write back the state
            to_src<src_data_t>(ptr[addr_states_t_l_reg], tmp1_vmm, vlen);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(vector_loop_inc_regs);
            to_src<src_data_t>(
                    ptr[addr_states_t_l_copy_reg], tmp1_vmm, vlen, true);

            // increment address pointers
            L(vector_loop_inc_regs);
            add(addr_scratch_gates_reg, vlen);
            if (rnn_.is_lstm_peephole) add(addr_weights_peephole_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_states_t_l_copy_reg, vlen_dst);
            add(addr_c_states_tm1_l_reg, vlen);
            add(addr_c_states_t_l_reg, vlen);
            if (is_training) add(addr_ws_gates_reg, vlen_dst);
            inc_regs(vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        L(rem_loop_start_label);
        {
            // load G0 G1 G2 G3
            uni_vmovss(G0, sg_addr(0));
            uni_vmovss(G1, sg_addr(1));
            uni_vmovss(G2, sg_addr(2));
            uni_vmovss(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8) {
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, false);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, false);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, false);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, false);
            }

            // add biases
            uni_vmovss(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(2));
            uni_vaddps(G2, G2, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(3));
            uni_vaddps(G3, G3, tmp1_vmm);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                uni_vmovss(tmp1_vmm, weights_peephole_addr(0));
                uni_vmovss(tmp2_vmm, ptr[addr_c_states_tm1_l_reg]);
                uni_vfmadd231ss(G0, tmp1_vmm, tmp2_vmm);
                uni_vmovss(tmp1_vmm, weights_peephole_addr(1));
                uni_vfmadd231ss(G1, tmp1_vmm, tmp2_vmm);
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(0), G0, scratch_dt_size);
                to_src<src_data_t>(wg_addr(1), G1, scratch_dt_size);
                to_src<src_data_t>(wg_addr(2), G2, scratch_dt_size);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmovss(tmp1_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1_vmm, tmp1_vmm, G1);
            uni_vfmadd231ps(tmp1_vmm, G0, G2);
            uni_vmovss(ptr[addr_c_states_t_l_reg], tmp1_vmm);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                uni_vmovss(tmp1_vmm, weights_peephole_addr(2));
                uni_vmovss(tmp2_vmm, ptr[addr_c_states_t_l_reg]);
                uni_vfmadd231ss(G3, tmp1_vmm, tmp2_vmm);
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(3), G3, scratch_dt_size);
            }

            // states_t_l = G3 * tanh(c_states_t_l)
            uni_vmovss(tmp1_vmm, ptr[addr_c_states_t_l_reg]);
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp1_vmm.getIdx());
            uni_vmulps(tmp1_vmm, tmp1_vmm, G3);

            // downconcvert/quantize and write back the state
            to_src<src_data_t>(
                    ptr[addr_states_t_l_reg], tmp1_vmm, scratch_dt_size);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(rem_loop_inc_regs);
            to_src<src_data_t>(ptr[addr_states_t_l_copy_reg], tmp1_vmm,
                    scratch_dt_size, true);

            // increment address pointers
            L(rem_loop_inc_regs);
            add(addr_scratch_gates_reg, scratch_dt_size);
            if (rnn_.is_lstm_peephole)
                add(addr_weights_peephole_reg, weights_peephole_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_states_t_l_copy_reg, hstate_dt_size);
            add(addr_c_states_tm1_l_reg, cstate_dt_size);
            add(addr_c_states_t_l_reg, cstate_dt_size);
            if (is_training) add(addr_ws_gates_reg, gate_dt_size);
            inc_regs(qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);

        init_table(vlen);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
