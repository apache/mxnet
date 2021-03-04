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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_amx.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

struct jit_brgemm_kernel_base_t : public jit_generator {
    jit_brgemm_kernel_base_t(const brgemm_t &abrg)
        : brg(abrg), eltwise_injector_(nullptr) {
        if (brg.with_eltwise) {
            const auto &p = brg.attr->post_ops_;
            const int eltwise_ind = p.find(primitive_kind::eltwise);

            post_ops_t::entry_t::eltwise_t eltwise;
            eltwise = p.entry_[eltwise_ind].eltwise;
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, eltwise, true, rax, Xbyak::Opmask(1));
        }
    }

    ~jit_brgemm_kernel_base_t() override { delete eltwise_injector_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_base_t)

    brgemm_t brg;

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t param1 = abi_param1;

    const reg64_t reg_C = r15;
    const reg64_t reg_aux_C = r14;

    const reg64_t reg_A = r13;
    const reg64_t reg_B = r12;

    const reg64_t reg_aux_A = r11;
    const reg64_t reg_aux_B = r10;

    const reg64_t reg_bdb_loop = r9;
    const reg64_t reg_ldb_loop = r8;

    const reg64_t reg_stride_lda = reg_bdb_loop;
    const reg64_t reg_stride_ldb = reg_ldb_loop;
    const reg64_t reg_stride_ld_block = reg_ldb_loop;

    const reg64_t reg_BS_loop = rax;
    const reg64_t reg_rdb_loop = rbx;
    const reg64_t reg_BS = abi_not_param1;

    const reg64_t reg_a_offset = rdx;
    const reg64_t reg_b_offset = rsi;

    const reg64_t reg_aux1_A = rbp;
    const reg64_t reg_aux1_B = abi_param1;

    const reg64_t reg_offset_A = reg_aux1_A;
    const reg64_t reg_offset_B = reg_aux1_B;

    const reg64_t reg_bias = reg_rdb_loop;
    const reg64_t reg_scales = reg_rdb_loop;
    const reg64_t reg_aux_bias = reg_rdb_loop;
    const reg64_t reg_aux_scales = reg_rdb_loop;
    const reg64_t reg_do_post_ops = reg_rdb_loop;
    const reg64_t reg_tmp_gpr = reg_rdb_loop;
    const reg64_t reg_ptr_sum_scale = reg_rdb_loop;

    const reg64_t reg_buf = reg_rdb_loop;

    const reg64_t reg_D = reg_aux_A;
    const reg64_t reg_aux_D = reg_BS_loop;

    constexpr static int origin_offset_A_offs_ = 0;
    constexpr static int origin_offset_B_offs_ = 8;
    constexpr static int reg_bias_offs_ = 16;
    constexpr static int reg_aux_bias_offs_ = 24;
    constexpr static int reg_do_post_ops_offs_ = 32;
    constexpr static int reg_D_offs_ = 40;
    constexpr static int reg_aux_D_offs_ = 48;
    constexpr static int reg_scales_offs_ = 56;
    constexpr static int reg_aux_scales_offs_ = 64;
    constexpr static int reg_bdb_loop_offs_ = 72;
    constexpr static int reg_ldb_loop_offs_ = 80;
    constexpr static int reg_buf_offs_ = 88;
    constexpr static int stack_space_needed_ = 128;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);

    Xbyak::Zmm accm(int ld_block, int bd, int ld) {
        return Xbyak::Zmm(31 - (bd * ld_block + ld));
    }
#if defined(_N_BCST_1_LOAD)
    Xbyak::Zmm bcst(int bd) {
        assert(brg.ld_block2 * brg.bd_block < 31);
        return Xbyak::Zmm(31 - (brg.ld_block2 * brg.bd_block) - bd);
    }
    Xbyak::Zmm load() { return Xbyak::Zmm(0); }
#else
    Xbyak::Zmm load(int ld) {
        assert(brg.ld_block2 * brg.bd_block < 31);
        return Xbyak::Zmm(31 - (brg.ld_block2 * brg.bd_block) - ld);
    }
    Xbyak::Zmm bcst() { return Xbyak::Zmm(0); }
#endif

    Xbyak::Zmm zmm_tmp_1() { return Xbyak::Zmm(0); }
    Xbyak::Zmm zmm_tmp_2() { return Xbyak::Zmm(1); }
    Xbyak::Zmm zmm_tmp_3() { return Xbyak::Zmm(2); }

    Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);
    Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(int bd_block2, bool is_bdb_tail, int ld_block);

    void store_accumulators(
            int bd_block2, bool is_bdb_tail, int ld_block, bool is_ld_tail);
    void store_accumulators_without_post_ops(
            int bd_block, int ld_block, bool is_ld_tail);
    void store_accumulators_apply_post_ops(
            int bd_block, int ld_block, bool is_ld_tail);
    void apply_beta(int bd_block, int ld_block, bool is_ld_tail);

    void restore_A_B_matrices();
    void restore_offsets();
    void set_A_B_matrices();

    void gemm_microkernel_avx512(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail);
    void gemm_microkernel_amx(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ldb_tail);
    void gemm_microkernel(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ldb_tail);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            int ldb_loop_length, bool is_reg_tail, bool is_ld_tail);
    void bdb_loop();

    void generate() override;

    int A_offset(int bd, int rd, bool is_amx = false);
    int B_offset(int ld, int rd, bool is_amx = false);
    int C_offset(int bd, int ld);
    int D_offset(int bd, int ld);

    int rdb_A_offset();
    int rdb_B_offset();

    int ldb_B_offset(int ld_block2, bool is_tail = false);
    int ldb_C_offset(int ld_block2, bool is_tail = false);
    int ldb_D_offset(int ld_block2, bool is_tail = false);

    int bdb_A_offset(int bd_block2);
    int bdb_C_offset(int bd_block2);
    int bdb_D_offset(int bd_block2);

    int bias_offset(int ld, bool is_tail = false);
    int scales_offset(int ld, bool is_tail = false);
};

int jit_brgemm_kernel_base_t::A_offset(int bd, int rd, bool is_amx) {
    return (is_amx) ? brg.typesize_A * (bd * brg.bd_block * brg.LDA)
                    : brg.typesize_A * (bd * brg.LDA + rd);
}
int jit_brgemm_kernel_base_t::B_offset(int ld, int rd, bool is_amx) {
    return (is_amx)
            ? brg.typesize_B * (brg.rd_step * ld * brg.ld_block)
            : brg.typesize_B * (rd * brg.LDB + brg.rd_step * ld * brg.ld_block);
};
int jit_brgemm_kernel_base_t::C_offset(int bd, int ld) {
    return brg.typesize_C * (bd * brg.LDC + ld * brg.ld_block);
}
int jit_brgemm_kernel_base_t::D_offset(int bd, int ld) {
    return brg.typesize_D * (bd * brg.LDD + ld * brg.ld_block);
}

int jit_brgemm_kernel_base_t::rdb_A_offset() {
    return brg.typesize_A * brg.rd_block;
}
int jit_brgemm_kernel_base_t::rdb_B_offset() {
    return brg.typesize_B * brg.rd_block * brg.LDB;
}

int jit_brgemm_kernel_base_t::ldb_B_offset(int ld_block2, bool is_tail) {
    return (is_tail) ? brg.typesize_B * brg.ldb_tail * brg.ld_step
                     : brg.typesize_B * ld_block2 * brg.ld_block * brg.ld_step;
}
int jit_brgemm_kernel_base_t::ldb_C_offset(int ld_block2, bool is_tail) {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_base_t::ldb_D_offset(int ld_block2, bool is_tail) {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}

int jit_brgemm_kernel_base_t::bdb_A_offset(int bd_block2) {
    return brg.typesize_A * bd_block2 * brg.bd_block * brg.LDA;
}
int jit_brgemm_kernel_base_t::bdb_C_offset(int bd_block2) {
    return brg.typesize_C * bd_block2 * brg.bd_block * brg.LDC;
}
int jit_brgemm_kernel_base_t::bdb_D_offset(int bd_block2) {
    return brg.typesize_D * bd_block2 * brg.bd_block * brg.LDD;
}

int jit_brgemm_kernel_base_t::bias_offset(int ld, bool is_tail) {
    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * ld * brg.ld_block;
}
int jit_brgemm_kernel_base_t::scales_offset(int ld, bool is_tail) {
    return (is_tail) ? brg.is_oc_scale * sizeof(float) * brg.ldb_tail
                     : brg.is_oc_scale * sizeof(float) * ld * brg.ld_block;
}
Xbyak::Zmm jit_brgemm_kernel_base_t::zmm_mask(const Xbyak::Zmm zmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

Xbyak::Ymm jit_brgemm_kernel_base_t::ymm_mask(const Xbyak::Ymm ymm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

void jit_brgemm_kernel_base_t::cvt2ps(data_type_t type_in,
        const Xbyak::Zmm zmm_in, const Xbyak::Operand &op, bool mask_flag,
        bool store, Xbyak::Opmask ktail_mask) {
    const Xbyak::Zmm zmm = zmm_mask(zmm_in, mask_flag, store, ktail_mask);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::bf16:
            vpmovzxwd(zmm, op);
            vpslld(zmm, zmm, 16);
            break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (!one_of(type_in, data_type::f32, data_type::bf16))
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_brgemm_kernel_base_t::read_params() {
    Label label_done;

    if (brg.layout == brgemm_row_major)
        mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
    else
        mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);

    if (brg.layout == brgemm_row_major)
        mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
    else
        mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);

    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    if (brg.is_int8_amx || brg.is_bf16_amx) {
        mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);
        mov(ptr[rsp + reg_buf_offs_], reg_buf);
    }

    if (brg.type == brgemm_offs) {
        mov(reg_offset_A, ptr[param1 + GET_OFF(offset_A)]);
        mov(reg_offset_B, ptr[param1 + GET_OFF(offset_B)]);

        mov(ptr[rsp + origin_offset_A_offs_], reg_offset_A);
        mov(ptr[rsp + origin_offset_B_offs_], reg_offset_B);
    }
    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        mov(ptr[rsp + reg_bias_offs_], reg_bias);
    }
    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(ptr[rsp + reg_scales_offs_], reg_scales);
    }

    mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
    mov(ptr[rsp + reg_do_post_ops_offs_], reg_do_post_ops);
}

void jit_brgemm_kernel_base_t::load_accumulators(
        int bd_block2, bool is_bdb_tail, int ld_block2) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        for_(int bdb = 0; bdb < bd_block2; bdb++)
        for (int ldb = 0; ldb < ld_block2; ldb++)
            tilezero(Tmm(brgemm_amx::get_C_tensor(bdb, ldb)));
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            vxorps(zmm, zmm, zmm);
        }
    }
}

void jit_brgemm_kernel_base_t::apply_beta(
        int bd_block, int ld_block2, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm_beta = zmm_tmp_1();
    auto zmm_alpha = zmm_tmp_2();
    auto zmm_prev_dst = zmm_tmp_3();

    const bool apply_alpha = (brg.alpha != 1.f && brg.alpha != 0.f);

    if (brg.beta != 1.f) {
        mov(reg_tmp_gpr, float2int((float)brg.beta));
        movq(Xmm(zmm_beta.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_beta, Xmm(zmm_beta.getIdx()));
    }
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int((float)brg.alpha));
        movq(Xmm(zmm_alpha.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_alpha, Xmm(zmm_alpha.getIdx()));
    }
    for (int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (brg.is_int8 && (apply_alpha || brg.beta != 1.f))
                vcvtdq2ps(zmm, zmm);
            if (apply_alpha) vmulps(zmm, zmm, zmm_alpha);
            if (brg.beta != 1.f) {
                cvt2ps(brg.dt_c, zmm_prev_dst,
                        ptr[reg_aux_C + C_offset(bd, ld)], true, false, k_mask);
                vfmadd231ps(zmm, zmm_prev_dst, zmm_beta);
            } else {
                if (brg.is_int8)
                    vpaddd(zmm | k_mask | T_z, zmm,
                            ptr[reg_aux_C + C_offset(bd, ld)]);
                else
                    vaddps(zmm | k_mask | T_z, zmm,
                            ptr[reg_aux_C + C_offset(bd, ld)]);
            }
        }
}

void jit_brgemm_kernel_base_t::store_accumulators_apply_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (brg.with_bias) { mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]); }
    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (!brg.is_f32 && !brg.is_bf16
                    && ((brg.beta != 0.f) || (brg.beta != 1.f)))
                vcvtdq2ps(zmm, zmm);
            if (brg.with_bias) {
                auto zmm_bias = zmm_tmp_1();
                cvt2ps(brg.dt_bias, zmm_bias,
                        ptr[reg_aux_bias + bias_offset(ld)], true, false,
                        k_mask);
                vaddps(zmm, zmm, zmm_bias);
            }
        }
    }
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                const Xbyak::Zmm zmm = zmm_mask(
                        accm(ld_block2, bd, ld), true, false, k_mask);
                vmulps(zmm, zmm, ptr[reg_aux_scales + scales_offset(ld)]);
            }
        }
    }

    bool sum_before_eltwise = false;
    if (brg.with_sum) {
        const auto &p = brg.attr->post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        sum_before_eltwise
                = (sum_idx == 0) && p.contain(primitive_kind::eltwise, 1);
    }

    if (brg.with_eltwise && !sum_before_eltwise)
        eltwise_injector_->compute_vector_range(32 - bd_block * ld_block2, 32);

    if (brg.with_sum) {
        const float *p_sum_scale = &brg.sum_scale;
        if (*p_sum_scale != 1.f) mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm = accm(ld_block2, bd, ld);
                auto addr = ptr[reg_aux_D + D_offset(bd, ld)];

                auto zmm_prev_dst = Xbyak::Zmm(0);
                cvt2ps(brg.dt_d, zmm_prev_dst, addr, true, false, k_mask);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    }

    if (brg.with_eltwise && sum_before_eltwise)
        eltwise_injector_->compute_vector_range(32 - bd_block * ld_block2, 32);

    auto zmm_zero = zmm_tmp_1();
    if (brg.dt_d == data_type::u8) vpxord(zmm_zero, zmm_zero, zmm_zero);

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (brg.dt_d == data_type::u8) vmaxps(zmm, zmm_zero, zmm);
            if (!one_of(brg.dt_d, data_type::f32, data_type::bf16))
                vcvtps2dq(zmm, zmm);
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
            auto zmm = accm(ld_block2, bd, ld);
            auto ymm = Xbyak::Ymm(zmm.getIdx());
            const Xbyak::Zmm r_zmm = zmm_mask(zmm, true, true, k_mask);
            const Xbyak::Ymm r_ymm = ymm_mask(ymm, true, true, k_mask);
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32: vmovups(addr, r_zmm); break;
                case data_type::bf16:
                    vcvtneps2bf16(ymm, zmm);
                    vmovdqu16(addr, r_ymm);
                    break;
                case data_type::s8: vpmovsdb(addr, r_zmm); break;
                case data_type::u8: vpmovusdb(addr, r_zmm); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brgemm_kernel_base_t::store_accumulators_without_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {
    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (!one_of(brg.beta, 1.f, 0.f) && (!brg.is_f32 && !brg.is_bf16))
                vcvtps2dq(zmm, zmm);
            if (is_ld_tail)
                vmovups(ptr[reg_aux_C + C_offset(bd, ld)] | ld_tail_mask | T_z,
                        zmm);
            else
                vmovups(ptr[reg_aux_C + C_offset(bd, ld)], zmm);
        }
    }
}

void jit_brgemm_kernel_base_t::store_accumulators(
        int bd_block2, bool is_bdb_tail, int ld_block2, bool is_ld_tail) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
        if (brg.beta != 0.f && brg.alpha != 0)
            mov(reg_stride_ld_block, brg.ld_block * brg.typesize_C);
        else
            mov(reg_stride_ld_block, brg.LDC * brg.typesize_C);

        mov(reg_buf, ptr[rsp + reg_buf_offs_]);
        for (int bdb = 0; bdb < bd_block2; bdb++) {
            for (int ldb = 0; ldb < ld_block2; ldb++) {
                if (brg.beta != 0.f && brg.alpha != 0) {
                    tilestored(ptr[reg_buf + reg_stride_ld_block],
                            Tmm(brgemm_amx::get_C_tensor(bdb, ldb)));
                    for (int bd = 0; bd < brg.bd_block; bd++) {
                        size_t buf_offset
                                = (bd * brg.ld_block) * brg.typesize_C;
                        if (is_ld_tail)
                            vmovups(accm(1, bd, 0) | ld_tail_mask | T_z,
                                    ptr[reg_buf + buf_offset]);
                        else
                            vmovups(accm(1, bd, 0), ptr[reg_buf + buf_offset]);
                    }
                    apply_beta(brg.bd_block, 1, is_ld_tail);
                    store_accumulators_without_post_ops(
                            brg.bd_block, 1, is_ld_tail);
                } else {
                    tilestored(ptr[reg_aux_C + reg_stride_ld_block],
                            Tmm(brgemm_amx::get_C_tensor(bdb, ldb)));
                }
                add(reg_aux_C, ldb_C_offset(1));
            }
            sub(reg_aux_C, ldb_C_offset(ld_block2));
            add(reg_aux_C, bdb_C_offset(1));
        }
        sub(reg_aux_C, bdb_C_offset(bd_block2));
        mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        if (brg.beta != 0.f && brg.alpha != 0) {
            apply_beta(bd_block, ld_block2, is_ld_tail);
        }
        if (one_of(true, brg.with_eltwise, brg.with_scales, brg.with_bias,
                    brg.with_sum, brg.dt_d != brg.dt_c)) {
            Label label_done, label_store_without_post_ops;

            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_store_without_post_ops, T_NEAR);

            store_accumulators_apply_post_ops(bd_block, ld_block2, is_ld_tail);
            jmp(label_done, T_NEAR);

            L(label_store_without_post_ops);
            store_accumulators_without_post_ops(
                    bd_block, ld_block2, is_ld_tail);

            L(label_done);
        } else {
            store_accumulators_without_post_ops(
                    bd_block, ld_block2, is_ld_tail);
        }
    }
}

void jit_brgemm_kernel_base_t::restore_A_B_matrices() {
    if (brg.type != brgemm_offs) {
        mov(reg_aux1_A, reg_A);
        mov(reg_aux1_B, reg_B);
    }
}
void jit_brgemm_kernel_base_t::restore_offsets() {
    if (brg.type == brgemm_offs) {
        mov(reg_offset_A, ptr[rsp + origin_offset_A_offs_]);
        mov(reg_offset_B, ptr[rsp + origin_offset_B_offs_]);
    }
}
void jit_brgemm_kernel_base_t::set_A_B_matrices() {
    if (brg.type == brgemm_addr) {
        mov(reg_aux_A, ptr[reg_aux1_A]);
        mov(reg_aux_B, ptr[reg_aux1_B]);

        add(reg_aux1_A, 8);
        add(reg_aux1_B, 8);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        add(reg_aux1_A, brg.stride_a);
        add(reg_aux1_B, brg.stride_b);
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);

        add(reg_aux_A, ptr[reg_offset_A]);
        add(reg_aux_B, ptr[reg_offset_B]);
        add(reg_offset_A, 8);
        add(reg_offset_B, 8);
    }
    add(reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_b_offset);
}

void jit_brgemm_kernel_base_t::gemm_microkernel_amx(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    MAYBE_UNUSED(is_rd_tail);
    MAYBE_UNUSED(is_ld_tail);
    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8) {
            tdpbuud(x1, x2, x3);
        } else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8) {
            tdpbusd(x1, x2, x3);
        } else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::u8) {
            tdpbsud(x1, x2, x3);
        } else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };

    mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
    mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);

    mov(reg_stride_lda, brg.typesize_A * brg.LDA);
    mov(reg_stride_ldb, brg.rd_step * brg.typesize_B * brg.LDB);

    for (int ldb = 0; ldb < ld_block2; ldb++) {
        tileloadd(Tmm(brgemm_amx::get_B_tensor(ldb)),
                ptr[reg_aux_B + B_offset(ldb, 0, true) + reg_stride_ldb]);
    }
    for (int bdb = 0; bdb < bd_block2; bdb++) {
        tileloadd(Tmm(brgemm_amx::get_A_tensor(bdb)),
                ptr[reg_aux_A + A_offset(bdb, 0, true) + reg_stride_lda]);
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            tdpbxxd(Tmm(brgemm_amx::get_C_tensor(bdb, ldb)),
                    Tmm(brgemm_amx::get_A_tensor(bdb)),
                    Tmm(brgemm_amx::get_B_tensor(ldb)));
        }
    }
    mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
    mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
}

void jit_brgemm_kernel_base_t::gemm_microkernel_avx512(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    MAYBE_UNUSED(bd_block2);
    auto dot_product = [=](Zmm z1, Zmm z2, Zmm z3) {
        if (brg.is_f32)
            vfmadd231ps(z1, z2, z3);
        else if (brg.is_bf16)
            vdpbf16ps(z1, z2, z3);
        else if (brg.is_int8)
            vpdpbusd(z1, z3, z2);
    };
    auto broadcast = [=](Zmm z1, size_t offset) {
        if (brg.is_f32)
            vbroadcastss(z1, ptr[reg_aux_A + offset]);
        else if (brg.is_bf16 || brg.is_int8)
            vpbroadcastd(z1, ptr[reg_aux_A + offset]);
    };

    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    bool is_emdbd = brg.embd_bcst;

    int rd_loop = 0, rd_tail_size = 0;
    if (is_rd_tail) {
        if (brg.is_bf16 || brg.is_int8) {
            rd_tail_size = brg.rdb_tail % brg.rd_step;
            rd_loop = (rd_tail_size != 0)
                    ? ((brg.rdb_tail / brg.rd_step) + 1) * brg.rd_step
                    : brg.rdb_tail;
        } else
            rd_loop = brg.rdb_tail;
    } else
        rd_loop = brg.rd_block;

#if defined(_N_BCST_1_LOAD)
    for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
        for (int bd = 0; bd < bd_block && !is_emdbd; bd++) {
            if (is_rd_tail && rd_tail_size != 0 && (rd == rd_loop - brg.rd_step)
                    && (brg.is_bf16 || brg.is_int8)) {
                vpxord(bcst(bd), bcst(bd), bcst(bd));
                Xmm xmm_tmp = Xmm(bcst(bd).getIdx());
                load_bytes(xmm_tmp, reg_aux_A, A_offset(bd, rd),
                        rd_tail_size * brg.typesize_A);
                vpbroadcastd(bcst(bd), xmm_tmp);
            } else
                broadcast(bcst(bd), A_offset(bd, rd));
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            if (is_ld_tail) {
                vmovups(load() | ld_tail_mask | T_z,
                        ptr[reg_aux_B + B_offset(ld, rd)]);
            } else {
                vmovups(load(), ptr[reg_aux_B + B_offset(ld, rd)]);
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto zmm = accm(ld_block2, bd, ld);
                if (is_emdbd)
                    vfmadd231ps(
                            zmm, load(), zword_b[reg_aux_A + A_offset(bd, rd)]);
                else
                    dot_product(zmm, load(), bcst(bd));
            }
        }
    }
#else
    for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
        int prefetch_count_B = 0;
        for (int ld = 0; ld < ld_block2; ld++) {
            if (is_ld_tail) {
                vmovups(load(ld) | ld_tail_mask | T_z,
                        ptr[reg_aux_B + B_offset(ld, rd)]);
            } else {
                vmovups(load(ld), ptr[reg_aux_B + B_offset(ld, rd)]);
            }
        }
        for (int bd = 0; bd < bd_block; bd++) {
            if (!is_emdbd) {
                if (is_rd_tail && rd_tail_size != 0
                        && (rd == rd_loop - brg.rd_step)
                        && (brg.is_bf16 || brg.is_int8)) {
                    vpxord(bcst(), bcst(), bcst());
                    Xmm xmm_tmp = Xmm(bcst().getIdx());
                    load_bytes(xmm_tmp, reg_aux_A, A_offset(bd, rd),
                            rd_tail_size * brg.typesize_A);
                    vpbroadcastd(bcst(), xmm_tmp);
                } else
                    broadcast(bcst(), A_offset(bd, rd));
            }
            if (prefetch_count_B < ld_block2) {
                prefetcht0(ptr[reg_aux_B + B_offset(prefetch_count_B++, rd)
                        + brg.LDB * brg.rd_block * brg.typesize_B]);
            }
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm = accm(ld_block2, bd, ld);
                if (is_emdbd)
                    vfmadd231ps(zmm, load(ld),
                            zword_b[reg_aux_A + A_offset(bd, rd)]);
                else
                    dot_product(zmm, load(ld), bcst());
            }
        }
    }
#endif
}

void jit_brgemm_kernel_base_t::gemm_microkernel(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        gemm_microkernel_amx(
                bd_block2, is_bdb_tail, ld_block2, is_rd_tail, is_ld_tail);
    } else {
        gemm_microkernel_avx512(
                bd_block2, is_bdb_tail, ld_block2, is_rd_tail, is_ld_tail);
    }
}

void jit_brgemm_kernel_base_t::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, int ldb_loop_length, bool is_reg_tail, bool is_ld_tail) {

    auto ldb_shift = [&](int ld_block2, bool is_tail = false) {
        int C_offset
                = (is_tail) ? ldb_C_offset(1, true) : ldb_C_offset(ld_block2);
        int D_offset
                = (is_tail) ? ldb_D_offset(1, true) : ldb_D_offset(ld_block2);
        add(reg_aux_C, C_offset);
        add(reg_aux_D, D_offset);

        add(reg_b_offset,
                (is_tail) ? ldb_B_offset(1, true) : ldb_B_offset(ld_block2));

        if (brg.with_bias) {
            mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
            add(reg_aux_bias,
                    (is_tail) ? bias_offset(1, true) : bias_offset(ld_block2));
            mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
        }
        if (brg.with_scales) {
            mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
            add(reg_aux_scales,
                    (is_tail) ? scales_offset(1, true)
                              : scales_offset(ld_block2));
            mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
        }
    };

    Label ldb_loop_label;
    Label rdb_loop_label;
    Label N_loop_label;

    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        xor_(reg_b_offset, reg_b_offset);
        if (brg.with_bias) {
            mov(reg_bias, ptr[rsp + reg_bias_offs_]);
            mov(ptr[rsp + reg_aux_bias_offs_], reg_bias);
        }
        if (brg.with_scales) {
            mov(reg_scales, ptr[rsp + reg_scales_offs_]);
            mov(ptr[rsp + reg_aux_scales_offs_], reg_scales);
        }
    }

    mov(reg_ldb_loop, ldb_loop_length);
    L(ldb_loop_label);
    {
        load_accumulators(bd_block2, is_bdb_tail, ld_block2);

        mov(ptr[rsp + reg_D_offs_], reg_D);
        mov(ptr[rsp + reg_aux_D_offs_], reg_aux_D);

        restore_offsets();
        restore_A_B_matrices();

        if (brg.alpha != 0.f) {
            mov(reg_BS_loop, reg_BS);
            L(N_loop_label);
            {
                set_A_B_matrices();

                if (brg.rdb > 0) {
                    mov(reg_rdb_loop, brg.rdb);
                    L(rdb_loop_label);
                    {
                        const bool is_rd_tail = false;
                        gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                                is_rd_tail, is_ld_tail);

                        add(reg_aux_A, rdb_A_offset());
                        add(reg_aux_B, rdb_B_offset());

                        dec(reg_rdb_loop);
                        cmp(reg_rdb_loop, 0);
                    }
                    jg(rdb_loop_label, T_NEAR);
                }
                if (brg.rdb_tail != 0) {
                    const bool is_rd_tail = true;
                    gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail);
                }

                dec(reg_BS_loop);
                cmp(reg_BS_loop, 0);
            }
            jg(N_loop_label, T_NEAR);
        }
        mov(reg_D, ptr[rsp + reg_D_offs_]);
        mov(reg_aux_D, ptr[rsp + reg_aux_D_offs_]);

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail);

        if (!is_ld_tail)
            ldb_shift(ld_block2);
        else
            ldb_shift(1, true);

        dec(reg_ldb_loop);
        cmp(reg_ldb_loop, 0);
    }
    jg(ldb_loop_label, T_NEAR);
}

void jit_brgemm_kernel_base_t::bdb_loop() {
    auto do_ldb_loop = [=](int bd_block2, bool is_bdb_tail) {
        if (brg.ldb2 > 0) {
            const bool is_ld_reg_tail = false;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ld_block2, brg.ldb2,
                    is_ld_reg_tail, is_ld_tail);
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = (brg.ldb2 == 0) ? false : true;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ldb2_tail, 1, is_ld_reg_tail,
                    is_ld_tail);
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail
                    = (brg.ldb2 == 0 && brg.ldb2_tail == 0) ? false : true;
            const bool is_ld_tail = true;
            ldb_loop(bd_block2, is_bdb_tail, 1, 1, is_ld_reg_tail, is_ld_tail);
        }
    };

    auto bdb_loop_body = [=](int bd_block2, bool is_bdb_tail) {
        do_ldb_loop(bd_block2, is_bdb_tail);

        add(reg_C, bdb_C_offset(bd_block2));
        add(reg_D, bdb_D_offset(bd_block2));
        add(reg_a_offset, bdb_A_offset(bd_block2));
    };
    auto bdb_loop_avx512 = [=]() {
        Label bdb_loop_label;
        mov(reg_bdb_loop, brg.bdb);
        L(bdb_loop_label);
        {
            bdb_loop_body(1, false);

            dec(reg_bdb_loop);
            cmp(reg_bdb_loop, 0);
        }
        jg(bdb_loop_label, T_NEAR);
    };
    auto bdb_loop_amx = [=]() {
        Label bdb_loop_label;
        if (brg.bd_block2 > 1) {
            mov(reg_bdb_loop, brg.bdb2);
            L(bdb_loop_label);
            {
                bdb_loop_body(brg.bd_block2, false);

                dec(reg_bdb_loop);
                cmp(reg_bdb_loop, 0);
            }
            jg(bdb_loop_label, T_NEAR);
        }
        if (brg.bdb2_tail > 0) bdb_loop_body(brg.bdb2_tail, false);
    };

    xor_(reg_a_offset, reg_a_offset);
    if (brg.is_int8_amx || brg.is_bf16_amx)
        bdb_loop_amx();
    else
        bdb_loop_avx512();
    if (brg.bdb_tail > 0) do_ldb_loop(1, true);
}

void jit_brgemm_kernel_base_t::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const auto full_mask = size_t {0xffffffffffffffff};
    const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);

    reg64_t reg_mask = rax;

    mov(reg_mask, full_mask);
    kmovq(ld_full_mask, reg_mask);
    mov(reg_mask, tail_mask);
    kmovq(ld_tail_mask, reg_mask);

    read_params();

#if !defined(_N_BCST_1_LOAD)
    if (!brg.embd_bcst && (brg.is_bf16 || brg.is_int8)) {
        Xmm xmm_tmp = Xmm(bcst().getIdx());
        vpxor(xmm_tmp, xmm_tmp, xmm_tmp);
    }
#endif

    bdb_loop();

    add(rsp, stack_space_needed_);

    postamble();

    if (brg.with_eltwise) eltwise_injector_->prepare_table();
}

brgemm_kernel_t::brgemm_kernel_t(const brgemm_t abrd) {
    brgemm_kernel_ = new jit_brgemm_kernel_base_t(abrd);
}

status_t brgemm_kernel_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_kernel_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

brgemm_kernel_t::~brgemm_kernel_t() {
    delete brgemm_kernel_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
