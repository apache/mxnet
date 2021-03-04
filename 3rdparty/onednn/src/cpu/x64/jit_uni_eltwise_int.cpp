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
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_uni_eltwise_int.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

struct jit_args_t {
    const void *from;
    const void *for_comparison;
    const void *to;
    size_t work_amount;
};

struct jit_uni_eltwise_int_kernel : public c_compatible {
    jit_uni_eltwise_int_kernel(const eltwise_desc_t &desc) : desc_(desc) {}
    virtual ~jit_uni_eltwise_int_kernel() = default;

    virtual void operator()(jit_args_t *p) = 0;
    virtual status_t create_kernel() = 0;

protected:
    data_type_t data_type() const { return desc_.data_desc.data_type; }
    int dtype_size() const { return types::data_type_size(data_type()); }

    const eltwise_desc_t &desc() const { return desc_; }

private:
    const eltwise_desc_t &desc_;
};

/* jit kernels */
namespace {
using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_subkernel_int_t : public jit_uni_eltwise_int_kernel,
                                 public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_subkernel_int)

    jit_uni_subkernel_int_t(const eltwise_desc_t &desc)
        : jit_uni_eltwise_int_kernel(desc), jit_generator() {
        using namespace data_type;

        // Relu and linear for int types: s32, s8, u8; Only forward direction
        assert(utils::one_of(desc.alg_kind, alg_kind::eltwise_relu,
                alg_kind::eltwise_linear));
        assert(utils::one_of(data_type(), s32, s8, u8));
        assert(utils::one_of(isa, sse41, avx2, avx512_common));
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(jit_args_t *p) override {
        return jit_generator::operator()(p);
    }

    void generate() override {
        Reg64 param = abi_param1;

        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t simd_w = vlen / sizeof(float);
        const size_t loop_dec[] = {simd_w, 1};
        const size_t uf[] = {1, 1};
        const size_t shift[] = {dtype_size() * simd_w, (size_t)dtype_size()};
        const bool loop_vectorize[] = {true, false};

        preamble();

#define GET_OFF(field) offsetof(jit_args_t, field)
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
#undef GET_OFF

        mov(imm_addr64, float2int(desc().alpha));
        uni_vmovq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        mov(imm_addr64, float2int(desc().beta));
        uni_vmovq(xmm_beta, imm_addr64);
        uni_vbroadcastss(vmm_beta, xmm_beta);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        xor_(reg_int8, reg_int8);
        if (isa == avx512_common) {
            mov(reg_int8.cvt8(), 0x01);
            kmovw(k_mask_int8, reg_int8.cvt32());
        }

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(
                    loop_vectorize[id], uf[id], shift[id], desc().alg_kind);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        postamble();
    }

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using opmask_t = const Xbyak::Opmask;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 reg_int8 = r9;

    Vmm vmm_tmp = Vmm(isa == avx512_common ? 25 : 10);
    Vmm vmm_saturation_ubound = Vmm(isa == avx512_common ? 26 : 11);
    Vmm vmm_alpha = Vmm(isa == avx512_common ? 27 : 12);
    Xmm xmm_alpha = Xmm(12);
    Vmm vmm_beta = Vmm(isa == avx512_common ? 28 : 13);
    Xmm xmm_beta = Xmm(13);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 29 : 14);
    Vmm vmm_mask = Vmm(isa == avx512_common ? 30 : 15);

    opmask_t k_mask = k1;
    opmask_t k_mask_int8 = k2; // Mask for store 1 byte in case of AVX512

    bool is32bit() const { return data_type() == data_type::s32; }

    // Load 32bit data type (s32)
    void load_32bit(
            const bool vectorize, const Vmm &vr_from, const Address &mem_from) {

        if (vectorize) {
            // load full Vmm size
            uni_vmovups(vr_from, mem_from);
        } else {
            // load exactly one data item
            movss(Xmm(vr_from.getIdx()), mem_from);
        }
    }

    // Load 8bit data type (u8/s8)
    void load_8bit(const bool vectorize, const Vmm &vr_from,
            const Address &mem_from, bool is_signed) {

        // data type u8/s8 load as s32
        if (vectorize) {
            // load full Vmm size
            if (is_signed)
                uni_vpmovsxbd(vr_from, mem_from);
            else
                uni_vpmovzxbd(vr_from, mem_from);
        } else {
            // load exactly one data item
            mov(reg_int8.cvt8(), mem_from);
            if (is_signed)
                movsx(reg_int8.cvt32(), reg_int8.cvt8());
            else
                movzx(reg_int8.cvt32(), reg_int8.cvt8());
            uni_vmovq(Xmm(vr_from.getIdx()), reg_int8);
        }
    }

    // Load vregs with data from mem
    void load(
            const bool vectorize, const Vmm &vr_from, const Address &mem_from) {

        // Branching on data size
        if (is32bit())
            load_32bit(vectorize, vr_from, mem_from);
        else
            load_8bit(
                    vectorize, vr_from, mem_from, data_type() == data_type::s8);
    }

    // Processing
    void process_linear(const Vmm &vr_to, const Vmm &vr_from);
    void process_relu(const Vmm &vr_to, const Vmm &vr_from);

    // Store s32 for any isa
    void store_32bit(
            const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
        if (vectorize) {
            // store full Vmm size
            uni_vmovups(mem_to, vr_to);
        } else {
            // store exactly one data item
            movss(mem_to, Xmm(vr_to.getIdx()));
        }
    }

    // Store 8 bit int - isa-dependent
    void store_8bit(const bool vectorize, const Address &mem_to,
            const Vmm &vr_to, bool is_signed);

    // Store results from vregs to mem
    void store(const bool vectorize, const Address &mem_to, const Vmm &vr_to) {
        // Branching on data size
        if (is32bit())
            store_32bit(vectorize, mem_to, vr_to);
        else
            store_8bit(vectorize, mem_to, vr_to, data_type() == data_type::s8);
    }

    void compute_step(bool vectorize, const size_t uf, const size_t shift,
            const alg_kind_t alg) {

        auto vreg_from = [&](const size_t i) -> Vmm { return Vmm(i + 1); };
        auto vreg_to = [&](const size_t i) -> Vmm { return Vmm(uf + i + 1); };

        // 1. Load (vregs <- mem)
        for (size_t i = 0; i < uf; i++)
            load(vectorize, vreg_from(i), ptr[reg_from + i * shift]);

        // 2. Process (vregs <- vergs)
        switch (alg) {
            case alg_kind::eltwise_linear:
                for (size_t i = 0; i < uf; i++)
                    process_linear(vreg_to(i), vreg_from(i));
                break;
            case alg_kind::eltwise_relu:
                for (size_t i = 0; i < uf; i++)
                    process_relu(vreg_to(i), vreg_from(i));
                break;
            default: assert(!"unsupported alg");
        }

        // 3. Store (mem <- vregs)
        for (size_t i = 0; i < uf; i++)
            store(vectorize, ptr[reg_to + i * shift], vreg_to(i));
    }
};

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::process_linear(
        const Vmm &vr_to, const Vmm &vr_from) {
    uni_vcvtdq2ps(vr_to, vr_from);
    uni_vfmadd213ps(vr_to, vmm_alpha, vmm_beta);

    // Saturate before converting from f32 to s32
    Reg64 reg_tmp = r10;
    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    init_saturate_f32(vmm_zero, vmm_saturation_ubound, reg_tmp, data_type::f32,
            data_type());
    saturate_f32(vr_to, vmm_zero, vmm_saturation_ubound, vmm_tmp, data_type());

    uni_vcvtps2dq(vr_to, vr_to);
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_subkernel_int_t<sse41>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    cvtdq2ps(vr_from, vr_from);
    movups(vr_to, vr_from);
    mulps(vr_to, vmm_alpha);

    Vmm mask = Vmm(0);
    movups(mask, vr_from);
    cmpps(mask, vmm_zero, _cmp_nle_us);
    blendvps(vr_to, vr_from);
    cvtps2dq(vr_to, vr_to);
}

template <>
void jit_uni_subkernel_int_t<avx2>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    vcvtdq2ps(vr_from, vr_from);
    vmulps(vr_to, vr_from, vmm_alpha);
    vcmpgtps(vmm_mask, vr_from, vmm_zero);
    vblendvps(vr_to, vr_to, vr_from, vmm_mask);
    vcvtps2dq(vr_to, vr_to);
}

template <>
void jit_uni_subkernel_int_t<avx512_common>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    vcvtdq2ps(vr_from, vr_from);
    vmulps(vr_to, vr_from, vmm_alpha);
    vcmpps(k_mask, vr_from, vmm_zero, _cmp_nle_us);
    vblendmps(vr_to | k_mask, vr_to, vr_from);
    vcvtps2dq(vr_to, vr_to);
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int_t<isa>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    assert(!"unsupported isa");
}

template <>
void jit_uni_subkernel_int_t<sse41>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16
        packssdw(vr_to, vmm_zero);
        // s16 -> s8/u8
        if (is_signed)
            packsswb(vr_to, vmm_zero);
        else
            packuswb(vr_to, vmm_zero);

        movd(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        packssdw(vr_to, vmm_zero);
        if (is_signed)
            packsswb(vr_to, vmm_zero);
        else
            packuswb(vr_to, vmm_zero);
        movd(reg_int8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_int8.cvt8());
    }
}

template <>
void jit_uni_subkernel_int_t<avx2>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16 = {qw0, 0, qw1, 0}
        vpackssdw(vr_to, vr_to, vmm_zero);
        // permute to restore order{qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
        vpermq(Ymm(vr_to.getIdx()), Ymm(vr_to.getIdx()), 0x58);

        // s16 -> s8/u8 : {16 x s16}{16 x 0} -> {32 x s8/u8}
        if (is_signed)
            vpacksswb(vr_to, vr_to, vmm_zero);
        else
            vpackuswb(vr_to, vr_to, vmm_zero);
        uni_vmovq(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        vpackssdw(vr_to, vr_to, vmm_zero);
        if (is_signed)
            vpacksswb(vr_to, vr_to, vmm_zero);
        else
            vpackuswb(vr_to, vr_to, vmm_zero);
        vmovd(reg_int8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_int8.cvt8());
    }
}

template <>
void jit_uni_subkernel_int_t<avx512_common>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        if (is_signed)
            vpmovsdb(mem_to, vr_to);
        else
            vpmovusdb(mem_to, vr_to);
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        if (is_signed)
            vpmovsdb(mem_to, vr_to | k_mask_int8);
        else
            vpmovusdb(mem_to, vr_to | k_mask_int8);
    }
}

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    bool ok = mayiuse(isa)
            && desc()->data_desc.data_type == d_type
            // only relu and linear so far
            && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                    alg_kind::eltwise_linear)
            && !has_zero_dim_memory()
            && memory_desc_wrapper(src_md()).is_dense(true)
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::jit_uni_eltwise_int_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::init(engine_t *engine) {
    const auto &desc = *pd()->desc();
    CHECK(safe_ptr_assign(kernel_, new jit_uni_subkernel_int_t<isa>(desc)));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::~jit_uni_eltwise_int_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
void jit_uni_eltwise_int_fwd_t<isa, d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const size_t nelems = data_d.nelems(true);

    src += data_d.offset0();
    dst += data_d.offset0();

    const int cache_line = 64 / data_d.data_type_size();
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args_t();
        arg.from = (const void *)&src[start];
        arg.for_comparison = (const void *)&src[start];
        arg.to = (const void *)&dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount) (*kernel_)(&arg);
    });
}

using namespace data_type;

template struct jit_uni_eltwise_int_fwd_t<sse41, s32>;
template struct jit_uni_eltwise_int_fwd_t<avx2, s32>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s32>;

template struct jit_uni_eltwise_int_fwd_t<sse41, s8>;
template struct jit_uni_eltwise_int_fwd_t<avx2, s8>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s8>;

template struct jit_uni_eltwise_int_fwd_t<sse41, u8>;
template struct jit_uni_eltwise_int_fwd_t<avx2, u8>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, u8>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
