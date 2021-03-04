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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_binary.hpp"
#include "cpu/x64/jit_uni_i8i8_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::data_type;
using namespace Xbyak;

template <data_type_t src0_type, data_type_t src1_type>
bool jit_uni_i8i8_binary_t<src0_type, src1_type>::post_ops_ok(
        const primitive_attr_t *attr, const memory_desc_wrapper &dst_d) {
    using namespace primitive_kind;

    const auto &p = attr->post_ops_;
    const auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    const auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };

    for (int i = 0; i < p.len(); i++) {
        if (p.contain(primitive_kind::sum, i)) {
            if (i > 0) return false;
        } else if (!(is_eltwise(i) || is_binary(i)))
            return false;
    }

    const int vlen = mayiuse(avx512_common)
            ? cpu_isa_traits<avx512_common>::vlen
            : cpu_isa_traits<avx2>::vlen;
    const int blksize = vlen / sizeof(float);
    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(p, dst_d);

    if (postops_per_oc_broadcast_exists && !dst_d.is_plain()
            && dst_d.is_blocking_desc()) {
        const auto blocking_desc = dst_d.blocking_desc();
        if (blocking_desc.inner_nblks != 1
                || blocking_desc.inner_blks[0] != blksize
                || blocking_desc.inner_idxs[0] != 1)
            return false;
    }

    return binary_injector::binary_args_broadcast_supported(p, dst_d)
            && binary_injector::binary_args_tail_supported(p, dst_d, vlen)
            && IMPLICATION(postops_per_oc_broadcast_exists,
                    binary_injector::all_binary_postop_rhs_per_oc_broadcast(p,
                            dst_d,
                            [&dst_d](const memory_desc_wrapper &rhs_arg_md) {
                                return IMPLICATION(!mayiuse(avx2),
                                        dst_d.consistent_with(rhs_arg_md)
                                                || dst_d.is_plain());
                            }));
}

enum class op_t : unsigned {
    none,
    tensor,
    bcast_c_blocked,
    bcast_n_spatial_c,
    bcast_n_c_spatial
};

static op_t get_bcast_per_c(const memory_desc_wrapper &src0_d) {
    const auto &strides = src0_d.blocking_desc().strides;
    const auto ndims = src0_d.ndims();

    if (!src0_d.is_plain())
        return op_t::bcast_c_blocked;
    else if (strides[1] == 1)
        return op_t::bcast_n_spatial_c;
    else if (strides[0] >= strides[1]
            && IMPLICATION(ndims >= 3, strides[1] >= strides[2]))
        return op_t::bcast_n_c_spatial;
    return op_t::none;
}
struct i8i8_binary_kernel_t {
    struct call_params_t {
        const float *scales_src0, *scales_src1;
        const char *src0;
        const char *src1;
        const char *dst;
        size_t spat_offt_count;
        const void *post_ops_binary_rhs_arg_vec;
        size_t oc_l_off;
    };

    i8i8_binary_kernel_t(int vlen) : vlen_(vlen) {}
    virtual ~i8i8_binary_kernel_t() = default;

    virtual void operator()(call_params_t *p) = 0;
    virtual status_t create_kernel() = 0;
    int vlen() const { return vlen_; }

protected:
    int vlen_ = 0;
    op_t postops_per_oc_bcast_ = op_t::none;
    /* load/store loop should start from vmm1, in case of sse4.1
     * eltwise injector use xmm(0) implicitly as tmp_xmm causing
     * assertion when xmm(0) passed to compute_vector_range
     */
    constexpr static int vmm_start_idx_ = 1;
};

#define PARAM_OFF(x) offsetof(call_params_t, x)

template <cpu_isa_t isa>
struct jit_uni_i8i8_binary_kernel_t : public i8i8_binary_kernel_t,
                                      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;

    const binary_pd_t *pd_;

    const Reg64 reg_param = abi_param1;
    const Reg64 reg_scales_src0 = rbx;
    const Reg64 reg_scales_src1 = rbp;
    const Reg64 reg_src0 = r8;
    const Reg64 reg_src1 = r9;
    const Reg64 reg_dst = r10;
    const Reg64 reg_offt_src0 = r11;
    const Reg64 reg_offt_src0_count = r12;
    const Reg64 reg_offt_src1 = rax;
    const Reg64 reg_reverse_spat_offt = r13;
    const Reg64 reg_tmp = r14;
    const Reg64 reg_elt_inj_table = r15;

    static constexpr size_t unroll_regs_ = isa == avx512_common ? 8 : 4;
    const size_t simd_w_ = vlen() / sizeof(float);
    size_t tail_size_ = 0;
    bool do_scale_src0_ = false;
    bool do_scale_src1_ = false;
    bool do_sum_ = false;
    float sum_scale_ = 0.f;
    bool broadcast_src1_value_ = false;

    const Vmm vreg_scales_src0 = Vmm(isa == avx512_common ? 17 : 9);
    const Vmm vreg_scales_src1 = Vmm(isa == avx512_common ? 18 : 10);
    const Vmm vreg_sum_scale = Vmm(isa == avx512_common ? 19 : 11);
    const Xmm xreg_sum_scale = Xmm(11);
    const Vmm vreg_zero = Vmm(isa == avx512_common ? 20 : 12);
    const Vmm vreg_saturation_ubound = Vmm(isa == avx512_common ? 21 : 13);
    const Vmm vreg_bcast_src1 = Vmm(isa == avx512_common ? 22 : 14);
    const Xmm xreg_bcast_src1 = Xmm(14);
    const Xmm xreg_tmp = Xmm(0);

    enum { nargs = 2 };
    // 0:src0 1:src1
    scales_t scales[nargs];

    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
    Opmask elt_inj_opmask = Opmask(1);

    void init() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const memory_desc_wrapper src1_d(pd_->src_md(1));
        const auto &dims = src0_d.dims();
        const auto ndims = src0_d.ndims();

        broadcast_src1_value_ = src1_d.nelems() == 1;
        dim_t nelems = 0;

        const bool postops_per_oc_broadcast_exists
                = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                        pd_->attr()->post_ops_, src0_d);
        const auto bcast_per_oc = get_bcast_per_c(src0_d);

        if (pd_->is_tensor_op() && !postops_per_oc_broadcast_exists)
            nelems = src0_d.nelems(true);
        else {
            if (bcast_per_oc == op_t::bcast_n_spatial_c)
                nelems = dims[1];
            else if (bcast_per_oc == op_t::bcast_n_c_spatial && ndims >= 3)
                nelems = utils::array_product(dims + 2, ndims - 2);
        }

        tail_size_ = nelems % simd_w_;

        scales[0].copy_from(pd_->attr()->scales_.get(DNNL_ARG_SRC_0));
        scales[1].copy_from(pd_->attr()->scales_.get(DNNL_ARG_SRC_1));

        do_scale_src0_ = !scales[0].has_default_values();
        do_scale_src1_ = !scales[1].has_default_values();

        const auto &po = pd_->attr()->post_ops_;
        do_sum_ = po.contain(primitive_kind::sum, 0)
                && po.entry_[0].sum.scale != 0.f;
        sum_scale_ = do_sum_ ? po.entry_[0].sum.scale : 0.f;
        postops_per_oc_bcast_ = get_bcast_per_c(src0_d);
        const bool with_eltwise = po.find(primitive_kind::eltwise) != -1;
        const bool with_binary = po.find(primitive_kind::binary) != -1;
        const bool with_postops = with_binary || with_eltwise;

        if (with_postops) init_post_ops_injector();
    }

    void init_post_ops_injector() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const auto &po = pd_->attr()->post_ops_;
        const eltwise_injector::static_params_t esp(true /*save_state*/,
                reg_elt_inj_table, elt_inj_opmask, true /*is_fwd*/,
                false /*use_dst*/);
        const binary_injector::rhs_arg_static_params_t rhs_arg_bsp {10, reg_tmp,
                reg_elt_inj_table, true /*preserve gpr*/, true /*preserve vmm*/,
                PARAM_OFF(post_ops_binary_rhs_arg_vec), src0_d};
        const binary_injector::static_params_t bsp(this->param1, rhs_arg_bsp);

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, po, bsp, esp);
    }

    void load_kernel_params() {
        mov(reg_tmp, float2int(sum_scale_));
        uni_vmovq(xreg_sum_scale, reg_tmp);
        uni_vbroadcastss(vreg_sum_scale, xreg_sum_scale);
        mov(reg_offt_src0_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src0, ptr[reg_param + PARAM_OFF(src0)]);
        mov(reg_src1, ptr[reg_param + PARAM_OFF(src1)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        if (do_scale_src0_)
            mov(reg_scales_src0, ptr[reg_param + PARAM_OFF(scales_src0)]);
        if (do_scale_src1_)
            mov(reg_scales_src1, ptr[reg_param + PARAM_OFF(scales_src1)]);
    }

    Address src0_ptr(size_t offt = 0) {
        return vmmword[reg_src0 + reg_offt_src0 + offt];
    }

    Address src1_ptr(size_t offt = 0) {
        return vmmword[reg_src1 + reg_offt_src1 + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_offt_src0 + offt];
    }

    void perform_op(const Vmm &v0, const Vmm &v1, const Vmm &s_src0,
            const Vmm &s_src1) {
        using namespace alg_kind;
        const auto alg = pd_->desc()->alg_kind;
        if (do_scale_src0_) uni_vmulps(v0, v0, s_src0);
        if (do_scale_src1_) uni_vmulps(v1, v1, s_src1);

        if (alg == binary_add)
            uni_vaddps(v0, v0, v1);
        else if (alg == binary_mul)
            uni_vmulps(v0, v0, v1);
        else if (alg == binary_max)
            uni_vmaxps(v0, v0, v1);
        else if (alg == binary_min)
            uni_vminps(v0, v0, v1);
        else if (alg == binary_div)
            uni_vdivps(v0, v0, v1);
        else
            assert(!"not supported operation!");
    }

    void load_and_convert(const Vmm &vmm, const Operand &op, data_type_t idt) {
        switch (idt) {
            case data_type::u8: uni_vpmovzxbd(vmm, op); break;
            case data_type::s8: uni_vpmovsxbd(vmm, op); break;
            default: assert(!"unreachable");
        }
        uni_vcvtdq2ps(vmm, vmm);
    }

    void accumulate_tail(const Xmm &xmm, int arg_num) {
        for (size_t i = 0; i < tail_size_; i++) {
            switch (arg_num) {
                case DNNL_ARG_SRC_0:
                    uni_vpinsrb(xmm, xmm, src0_ptr(i), i);
                    break;
                case DNNL_ARG_SRC_1:
                    uni_vpinsrb(xmm, xmm, src1_ptr(i), i);
                    break;
                case DNNL_ARG_DST: uni_vpinsrb(xmm, xmm, dst_ptr(i), i); break;
                default: assert(!"unsupported arg_num"); break;
            }
        }
    }

    void load(const Vmm &vmm, const Address &addr, int arg_num, data_type_t idt,
            bool tail) {
        // i8 -> f32
        if (!tail) {
            UNUSED(arg_num);
            load_and_convert(vmm, addr, idt);
        } else {
            UNUSED(addr);
            Xbyak::Xmm xreg_tmp = Xbyak::Xmm(vmm.getIdx());
            accumulate_tail(xreg_tmp, arg_num);
            load_and_convert(vmm, xreg_tmp, idt);
        }
    }

    void store_tail(const Xmm &xmm) {
        for (size_t i = 0; i < tail_size_; i++)
            uni_vpextrb(dst_ptr(i), xmm, i);
    }

    virtual void compute_dst(int unroll, bool tail) = 0;

    void apply_postops(int unroll) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        for (int vmm_idx = vmm_start_idx_; vmm_idx < unroll + vmm_start_idx_;
                vmm_idx++) {
            if (postops_per_oc_bcast_ == op_t::bcast_c_blocked
                    || postops_per_oc_bcast_ == op_t::bcast_n_c_spatial) {
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr[param1 + PARAM_OFF(oc_l_off)]);
            } else if (postops_per_oc_bcast_ == op_t::bcast_n_spatial_c) {
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        vmm_idx, reg_offt_src0);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(vmm_idx,
                        (vmm_idx - vmm_start_idx_) * static_cast<int>(simd_w_));
            }
        }
        postops_injector_->compute_vector_range(
                vmm_start_idx_, unroll + vmm_start_idx_, rhs_arg_params);
    }

    void forward() {
        auto dst_type = pd_->dst_md(0)->data_type;
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);
        init_saturate_f32(
                vreg_zero, vreg_saturation_ubound, reg_tmp, f32, dst_type);

        // Only mask 0 is supported at this point
        if (do_scale_src0_)
            uni_vbroadcastss(vreg_scales_src0, dword[reg_scales_src0]);
        if (do_scale_src1_)
            uni_vbroadcastss(vreg_scales_src1, dword[reg_scales_src1]);

        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        // reverse spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt, reg_offt_src0_count);
        xor_(reg_offt_src0, reg_offt_src0); // offt_src0 to get addr of src0/dst
        xor_(reg_offt_src1, reg_offt_src1); // offt_src1 to get addr of src1

        // bcast vreg just one time per kernel call
        if (broadcast_src1_value_) {
            uni_vpxor(xreg_bcast_src1, xreg_bcast_src1, xreg_bcast_src1);
            uni_vpinsrb(xreg_bcast_src1, xreg_bcast_src1, src1_ptr(0), 0);
            uni_vcvtdq2ps(xreg_bcast_src1, xreg_bcast_src1);
            uni_vbroadcastss(vreg_bcast_src1, xreg_bcast_src1);
        }

        L(unroll_loop);
        {
            size_t offt = unroll_regs_ * simd_w_;
            cmp(reg_reverse_spat_offt, offt);
            jl(unroll_loop_tail, T_NEAR);

            compute_dst(unroll_regs_, false);
            sub(reg_reverse_spat_offt, offt);
            add(reg_offt_src0, offt);
            if (!broadcast_src1_value_) add(reg_offt_src1, offt);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_spat_offt, simd_w_);
            jl(nelems_tail, T_NEAR);

            compute_dst(1, false);
            sub(reg_reverse_spat_offt, simd_w_);
            add(reg_offt_src0, simd_w_);
            if (!broadcast_src1_value_) add(reg_offt_src1, simd_w_);
            jmp(unroll_loop_tail);
        }

        L(nelems_tail);
        {
            cmp(reg_reverse_spat_offt, 1);
            jl(end, T_NEAR);

            compute_dst(1, true);
        }

        L(end);
    }

    void generate() override {
        preamble();
        load_kernel_params();
        forward();
        postamble();
        if (postops_injector_) postops_injector_->prepare_table();
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(i8i8_binary_kernel_t::call_params_t *p) override {
        return jit_generator::operator()(p);
    }

    jit_uni_i8i8_binary_kernel_t(const binary_pd_t *pd)
        : i8i8_binary_kernel_t(cpu_isa_traits<isa>::vlen), pd_(pd) {
        init();
    }

    ~jit_uni_i8i8_binary_kernel_t() override = default;
};

template <cpu_isa_t isa, data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t;

template <data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t<avx512_common, src0_type, src1_type>
    : public jit_uni_i8i8_binary_kernel_t<avx512_common> {

    void cvt2odt(const Operand &dst, const Vmm &src, data_type_t odt) {
        assert(utils::one_of(
                odt, data_type::u8, data_type::s8, data_type::s32));

        // properly saturate in f32
        saturate_f32(src, vreg_zero, vreg_saturation_ubound, odt);
        vcvtps2dq(src, src);

        switch (odt) {
            case data_type::s8: vpmovsdb(dst, src); break;
            case data_type::u8: vpmovusdb(dst, src); break;
            default: assert(!"unreachable");
        }
    }

    void store(const Operand &dst, const Vmm &src, data_type_t odt, bool tail) {
        // f32 -> i8 and store
        if (!tail) {
            cvt2odt(dst, src, odt);
        } else {
            UNUSED(dst);
            cvt2odt(xreg_tmp, src, odt);
            store_tail(xreg_tmp);
        }
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp = Vmm(unroll + i + vmm_start_idx_);
            const Vmm vreg_tmp_src1
                    = !broadcast_src1_value_ ? vreg_tmp : vreg_bcast_src1;
            const int offt = simd_w_ * i;
            load(vreg_tmp_src0, src0_ptr(offt), DNNL_ARG_SRC_0, src0_type,
                    tail);
            if (!broadcast_src1_value_) {
                load(vreg_tmp_src1, src1_ptr(offt), DNNL_ARG_SRC_1, src1_type,
                        tail);
            }

            // avoid multiple multiplication on input scale for broadcasted vreg
            uni_vmovups(vreg_tmp, vreg_tmp_src1);
            perform_op(vreg_tmp_src0, vreg_tmp, vreg_scales_src0,
                    vreg_scales_src1);
            if (do_sum_) {
                load(vreg_tmp, dst_ptr(offt), DNNL_ARG_DST, src0_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vreg_sum_scale);
            }
        }

        if (postops_injector_) apply_postops(unroll);

        for (int i = 0; i < unroll; i++) {
            const int offt = simd_w_ * i;
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            store(dst_ptr(offt), vreg_tmp_src0, src0_type, tail);
        }
    }

    jit_i8i8_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_i8i8_binary_kernel_t(pd) {}
};

template <data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t<avx2, src0_type, src1_type>
    : public jit_uni_i8i8_binary_kernel_t<avx2> {

    void cvt2odt(const Vmm &v, data_type_t odt) {
        // f32 -> s32
        // properly saturate in f32
        saturate_f32(v, vreg_zero, vreg_saturation_ubound, odt);
        vcvtps2dq(v, v);
        // v = { 8x32 }
        vpackssdw(v, v, vreg_zero);
        // v = { 4x16, 0, 4x16, 0 }
        vpermq(v, v, 0x58);
        // v =  { 8x16, 0 }

        switch (odt) {
            case data_type::u8: vpackuswb(v, v, vreg_zero); break;
            case data_type::s8: vpacksswb(v, v, vreg_zero); break;
            default: assert(!"unreachable");
        }
        // v = { 8x8, 0 }
    }

    void store(const Address &dst, const Vmm &src, data_type_t odt, bool tail) {
        // f32 -> i8 and store
        cvt2odt(src, odt);
        if (!tail) {
            uni_vmovq(dst, Xmm(src.getIdx())); // store 64 bits
        } else {
            UNUSED(dst);
            store_tail(Xmm(src.getIdx()));
        }
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp = Vmm(unroll + i + vmm_start_idx_);
            const Vmm vreg_tmp_src1
                    = !broadcast_src1_value_ ? vreg_tmp : vreg_bcast_src1;
            const int offt = simd_w_ * i;
            load(vreg_tmp_src0, src0_ptr(offt), DNNL_ARG_SRC_0, src0_type,
                    tail);
            if (!broadcast_src1_value_) {
                load(vreg_tmp_src1, src1_ptr(offt), DNNL_ARG_SRC_1, src1_type,
                        tail);
            }

            // avoid multiple multiplication on input scale for broadcasted vreg
            uni_vmovups(vreg_tmp, vreg_tmp_src1);
            perform_op(vreg_tmp_src0, vreg_tmp, vreg_scales_src0,
                    vreg_scales_src1);

            if (do_sum_) {
                load(vreg_tmp, dst_ptr(offt), DNNL_ARG_DST, src0_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vreg_sum_scale);
            }
        }

        if (postops_injector_) apply_postops(unroll);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const int offt = simd_w_ * i;
            store(dst_ptr(offt), vreg_tmp_src0, src0_type, tail);
        }
    }

    jit_i8i8_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_i8i8_binary_kernel_t(pd) {}
};

template <data_type_t src0_type, data_type_t src1_type>
struct jit_i8i8_binary_subkernel_t<sse41, src0_type, src1_type>
    : public jit_uni_i8i8_binary_kernel_t<sse41> {

    void cvt2odt(const Vmm &v, const Vmm &v_tmp, data_type_t odt) {
        // f32 -> s32
        // properly saturate in f32
        saturate_f32(v, vreg_zero, vreg_saturation_ubound, v_tmp, odt);
        cvtps2dq(v, v);
        // v = { 8x32 }
        packssdw(v, vreg_zero);
        // v = { 4x16, 0}

        switch (odt) {
            case data_type::u8: packuswb(v, vreg_zero); break;
            case data_type::s8: packsswb(v, vreg_zero); break;
            default: assert(!"unreachable");
        }
        // v = { 4x8, 0 }
    }

    void store(const Address &dst, const Vmm &v_src, const Vmm &v_tmp,
            data_type_t odt, bool tail) {
        // f32 -> i8 and store
        cvt2odt(v_src, v_tmp, odt);
        if (!tail) {
            movd(dst, Xmm(v_src.getIdx())); // store 32 bits
        } else {
            UNUSED(dst);
            store_tail(Xmm(v_src.getIdx()));
        }
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp = Vmm(unroll + i + vmm_start_idx_);
            const Vmm vreg_tmp_src1
                    = !broadcast_src1_value_ ? vreg_tmp : vreg_bcast_src1;
            const int offt = simd_w_ * i;
            load(vreg_tmp_src0, src0_ptr(offt), DNNL_ARG_SRC_0, src0_type,
                    tail);
            if (!broadcast_src1_value_) {
                load(vreg_tmp_src1, src1_ptr(offt), DNNL_ARG_SRC_1, src1_type,
                        tail);
            }

            // avoid multiple multiplication on input scale for broadcasted vreg
            movups(vreg_tmp, vreg_tmp_src1);
            perform_op(vreg_tmp_src0, vreg_tmp, vreg_scales_src0,
                    vreg_scales_src1);
            if (do_sum_) {
                load(vreg_tmp, dst_ptr(offt), DNNL_ARG_DST, src0_type, tail);
                mulps(vreg_tmp, vreg_sum_scale);
                addps(vreg_tmp_src0, vreg_tmp);
            }
        }

        if (postops_injector_) apply_postops(unroll);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            const Vmm vreg_tmp = Vmm(unroll + i + vmm_start_idx_);
            const int offt = simd_w_ * i;
            store(dst_ptr(offt), vreg_tmp_src0, vreg_tmp, src0_type, tail);
        }
    }

    jit_i8i8_binary_subkernel_t(const binary_pd_t *pd)
        : jit_uni_i8i8_binary_kernel_t(pd) {}
};

#undef PARAM_OFF

template <data_type_t src0_type, data_type_t src1_type>
std::unique_ptr<i8i8_binary_kernel_t> create_i8i8_binary_kernel(
        const binary_pd_t *pd) {
    if (mayiuse(avx512_common)) {
        using subkernel_t = jit_i8i8_binary_subkernel_t<avx512_common,
                src0_type, src1_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    } else if (mayiuse(avx2)) {
        using subkernel_t
                = jit_i8i8_binary_subkernel_t<avx2, src0_type, src1_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    } else {
        using subkernel_t
                = jit_i8i8_binary_subkernel_t<sse41, src0_type, src1_type>;
        return std::unique_ptr<subkernel_t> {new subkernel_t(pd)};
    }
}

template <data_type_t src0_type, data_type_t src1_type>
jit_uni_i8i8_binary_t<src0_type, src1_type>::jit_uni_i8i8_binary_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <data_type_t src0_type, data_type_t src1_type>
status_t jit_uni_i8i8_binary_t<src0_type, src1_type>::init(engine_t *engine) {
    kernel_ = create_i8i8_binary_kernel<src0_type, src1_type>(pd());
    return kernel_->create_kernel();
}

template <data_type_t src0_type, data_type_t src1_type>
jit_uni_i8i8_binary_t<src0_type, src1_type>::~jit_uni_i8i8_binary_t() = default;

template <data_type_t src0_type, data_type_t src1_type>
status_t jit_uni_i8i8_binary_t<src0_type, src1_type>::execute(
        const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto &post_ops = pd()->attr()->post_ops_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(post_ops, ctx);
    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));

    static constexpr int nargs = 2;
    scales_t scales[nargs];
    CHECK(scales[0].copy_from(pd()->attr()->scales_.get(DNNL_ARG_SRC_0)));
    CHECK(scales[1].copy_from(pd()->attr()->scales_.get(DNNL_ARG_SRC_1)));

    const int ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const dim_t MB = dims[0];
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
    const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
    const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
    const dim_t SP = D * H * W;
    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    post_ops, src0_d);
    const bool no_broadcast = pd()->is_tensor_op();

    if (no_broadcast && !postops_per_oc_broadcast_exists) {
        const int simd_w = (*kernel_).vlen(); // 1-byte elements
        const dim_t nelems0 = src0_d.nelems(true);
        const dim_t nelems0_simd = nelems0 / simd_w;
        const dim_t nelems0_tail = nelems0 % simd_w;
        bool has_tail = nelems0_tail > 0;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems0_simd + has_tail, nthr, ithr, start, end);
            if (start >= end) return;

            const bool ithr_does_tail
                    = has_tail && end == nelems0_simd + has_tail;
            const dim_t n_simd_to_do = (end - start - ithr_does_tail) * simd_w;
            const dim_t tail_to_do = ithr_does_tail * nelems0_tail;

            i8i8_binary_kernel_t::call_params_t p;
            p.spat_offt_count = (n_simd_to_do + tail_to_do) * sizeof(int8_t);
            p.src0 = src0 + start * simd_w;
            p.src1 = src1 + start * simd_w;
            p.dst = dst + start * simd_w;
            p.scales_src0 = scales[0].scales_;
            p.scales_src1 = scales[1].scales_;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel_)(&p);
        });
    } else {
        const auto postops_per_oc_bcast = get_bcast_per_c(src0_d);
        const auto &bcast_dims = pd()->broadcast_dims();
        const dim_t nelems_slice_src0
                = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);
        const dim_t nelems_slice_src1 = no_broadcast
                ? nelems_slice_src0
                : ((bcast_dims[0] == 0) ? utils::array_product(
                           src1_d.padded_dims() + 1, ndims - 1)
                                        : 0);
        const int simd_w = (*kernel_).vlen() / sizeof(float);

        if (no_broadcast && postops_per_oc_broadcast_exists
                && postops_per_oc_bcast == op_t::bcast_c_blocked) {
            const dim_t C_blocks = src0_d.padded_dims()[1] / simd_w;
            parallel_nd(MB, C_blocks, [&](dim_t mb, dim_t C_blk) {
                i8i8_binary_kernel_t::call_params_t p;
                p.spat_offt_count = SP * simd_w * sizeof(int8_t);
                const auto off = mb * nelems_slice_src0 + C_blk * SP * simd_w;
                p.dst = dst + off;
                p.src0 = src0 + off;
                p.src1 = src1 + off;
                p.oc_l_off = C_blk * simd_w;
                p.scales_src0 = scales[0].scales_;
                p.scales_src1 = scales[1].scales_;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*kernel_)(&p);
            });
        } else if (no_broadcast && postops_per_oc_broadcast_exists
                && postops_per_oc_bcast == op_t::bcast_n_c_spatial) {
            parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
                i8i8_binary_kernel_t::call_params_t p;
                p.spat_offt_count = SP * sizeof(int8_t);
                const auto off = mb * nelems_slice_src0 + c * SP;
                p.dst = dst + off;
                p.src0 = src0 + off;
                p.src1 = src1 + off;
                p.scales_src0 = scales[0].scales_;
                p.scales_src1 = scales[1].scales_;
                p.oc_l_off = c;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*kernel_)(&p);
            });
        } else {
            // Compute strategy:
            // Each line of channels is individual, parallel over MB and spatial
            parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
                i8i8_binary_kernel_t::call_params_t p;
                p.spat_offt_count = C * sizeof(int8_t);
                const auto offset = mb * nelems_slice_src0 + sp * C;
                p.dst = dst + offset;
                p.src0 = src0 + offset;
                const auto offset_src1
                        = no_broadcast ? offset : mb * nelems_slice_src1;
                p.src1 = src1 + offset_src1;
                p.scales_src0 = scales[0].scales_;
                p.scales_src1 = scales[1].scales_;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*kernel_)(&p);
            });
        }
    }

    return status::success;
}

using namespace data_type;

template struct jit_uni_i8i8_binary_t<u8, u8>;
template struct jit_uni_i8i8_binary_t<u8, s8>;
template struct jit_uni_i8i8_binary_t<s8, s8>;
template struct jit_uni_i8i8_binary_t<s8, u8>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
