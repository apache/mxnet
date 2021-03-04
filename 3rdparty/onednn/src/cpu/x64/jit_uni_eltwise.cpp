/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel : public c_compatible {
    jit_uni_eltwise_kernel(const eltwise_pd_t *pd) : pd_(pd) {}
    virtual ~jit_uni_eltwise_kernel() = default;

    virtual void operator()(jit_args_t *args) = 0;
    virtual status_t create_kernel() = 0;

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const { return pd_->src_md()->data_type; }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

// jit kernels
namespace {

struct jit_bf16_injector_t {
    jit_bf16_injector_t(
            jit_generator *host, Opmask k_tail_mask, bf16_emulation_t *emu)
        : h(host), k_tail_mask_(k_tail_mask), emu_(emu) {}

    void prepare_mask() {
        Reg64 reg_tmp = h->r14;
        h->sub(h->rsp, 8); // sizeof(Reg64)
        h->mov(h->ptr[h->rsp], reg_tmp);
        h->mov(reg_tmp.cvt32(), 0x1);
        h->kmovd(k_tail_mask_, reg_tmp.cvt32());
        h->mov(reg_tmp, h->ptr[h->rsp]);
        h->add(h->rsp, 8);
    }

    void load_bf16_cvt_to_f32(size_t idx, Reg64 reg_src, bool is_tail = false,
            size_t offset = 0) {
        Zmm zmm_f32 = Zmm(idx);
        zmm_f32 = is_tail ? zmm_f32 | k_tail_mask_ | Xbyak::util::T_z : zmm_f32;
        h->vpmovzxwd(zmm_f32, h->ptr[reg_src + offset]);
        h->vpslld(zmm_f32, zmm_f32, 16);
    }

    void cvt_f32_to_bf16_store(int step, size_t idx, Reg64 reg_dst,
            bool is_tail = false, size_t offset = 0) {
        assert(step >= 1 && step <= 2
                && IMPLICATION(step == 2, is_tail == false));
        if (step == 2 && !is_tail) {
            Ymm ymm_bf16_0 = Ymm(idx);
            Ymm ymm_bf16_1 = Ymm(idx + 1);
            Zmm zmm_f32_0 = Zmm(idx);
            Zmm zmm_f32_1 = Zmm(idx + 1);
            if (emu_) {
                emu_->vcvtneps2bf16(ymm_bf16_0, zmm_f32_0);
                emu_->vcvtneps2bf16(ymm_bf16_1, zmm_f32_1);
                h->vinserti64x4(zmm_f32_0, zmm_f32_0, ymm_bf16_1, 1);
                h->vmovups(h->ptr[reg_dst + offset], zmm_f32_0);
            } else {
                h->vcvtne2ps2bf16(zmm_f32_1, zmm_f32_1, zmm_f32_0);
                h->vmovups(h->ptr[reg_dst + offset], zmm_f32_1);
            }
        } else {
            Ymm ymm_bf16 = Ymm(idx);
            Zmm zmm_f32 = Zmm(idx);
            if (emu_)
                emu_->vcvtneps2bf16(ymm_bf16, zmm_f32);
            else
                h->vcvtneps2bf16(ymm_bf16, zmm_f32);
            if (!is_tail)
                h->vmovdqu16(h->ptr[reg_dst + offset], ymm_bf16);
            else
                h->vmovdqu16(h->ptr[reg_dst + offset] | k_tail_mask_, ymm_bf16);
        }
    }

private:
    jit_generator *const h;
    Xbyak::Opmask k_tail_mask_;
    bf16_emulation_t *const emu_;
};

template <cpu_isa_t isa>
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd)
        : jit_uni_eltwise_kernel(pd), jit_generator() {
        if (is_bf16()) {
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_reserv_1,
                        bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                        bf16_emu_reserv_5));
            bf16_injector_.reset(new jit_bf16_injector_t(
                    this, k_tail_mask, bf16_emu_.get()));
        }

        const auto &desc = *pd_->desc();
        // there's no auxiliary vregs on fwd path
        const bool is_fwd = pd_->is_fwd();
        const bool save_state = is_fwd ? false : true;
        eltwise_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, 1.f, save_state,
                reg_injector_table, injector_mask, is_fwd, pd_->use_dst()));
    }

    void generate() override {
        const bool is_fwd = pd_->is_fwd();
        preamble();

        if (is_bf16()) {
            bf16_injector_->prepare_mask();
            if (!mayiuse(avx512_core_bf16)) bf16_emu_->init_vcvtneps2bf16();
        }

        Reg64 param = abi_param1;
        mov(reg_src, ptr[param + GET_OFF(src)]);
        mov(reg_dst, ptr[param + GET_OFF(dst)]);
        if (!is_fwd) mov(reg_diff_dst, ptr[param + GET_OFF(diff_dst)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        eltwise_injector_->load_table_addr();

        Label reminder_loop_start, reminder_loop_end;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w());
        jl(reminder_loop_start, T_NEAR);

        L(vectorized_loop_start);

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.
        if (is_bf16()) {
            bf16_injector_->load_bf16_cvt_to_f32(vmm_src.getIdx(), reg_src);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            if (!is_fwd) {
                bf16_injector_->load_bf16_cvt_to_f32(
                        vmm_diff_dst.getIdx(), reg_diff_dst);
                uni_vmulps(vmm_src, vmm_src, vmm_diff_dst);
            }
            bf16_injector_->cvt_f32_to_bf16_store(1, vmm_src.getIdx(), reg_dst);
        } else {
            uni_vmovups(vmm_src, ptr[reg_src]);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            if (!is_fwd) {
                uni_vmovups(vmm_diff_dst, ptr[reg_diff_dst]);
                uni_vmulps(vmm_src, vmm_src, vmm_diff_dst);
            }
            uni_vmovups(ptr[reg_dst], vmm_src);
        }

        const auto shift = vlen();
        add(reg_src, shift);
        add(reg_dst, shift);
        if (!is_fwd) add(reg_diff_dst, shift);

        sub(reg_work_amount, simd_w());
        cmp(reg_work_amount, simd_w());
        jge(vectorized_loop_start, T_NEAR);

        L(vectorized_loop_end);

        L(reminder_loop_start);

        cmp(reg_work_amount, 0);
        jle(reminder_loop_end, T_NEAR);
        if (is_bf16()) {
            bf16_injector_->load_bf16_cvt_to_f32(
                    vmm_src.getIdx(), reg_src, true);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            if (!is_fwd) {
                bf16_injector_->load_bf16_cvt_to_f32(
                        vmm_diff_dst.getIdx(), reg_diff_dst, true);
                uni_vmulps(vmm_src, vmm_src, vmm_diff_dst);
            }
            bf16_injector_->cvt_f32_to_bf16_store(
                    1, vmm_src.getIdx(), reg_dst, true);
        } else {
            uni_vmovss(xmm_src, ptr[reg_src]);
            eltwise_injector_->compute_vector(xmm_src.getIdx());
            if (!is_fwd) {
                uni_vmovss(xmm_diff_dst, ptr[reg_diff_dst]);
                uni_vmulps(xmm_src, xmm_src, xmm_diff_dst);
            }
            uni_vmovss(ptr[reg_dst], xmm_src);
        }
        add(reg_src, dtype_size());
        add(reg_dst, dtype_size());
        if (!is_fwd) add(reg_diff_dst, dtype_size());

        dec(reg_work_amount);
        jmp(reminder_loop_start, T_NEAR);

        L(reminder_loop_end);

        postamble();

        eltwise_injector_->prepare_table();
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(jit_args_t *p) override { jit_generator::operator()(p); }

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    int vlen() {
        int vlen = cpu_isa_traits<isa>::vlen;
        return is_bf16() ? vlen / 2 : vlen;
    }
    int simd_w() { return vlen() / dtype_size(); }

    Reg64 reg_src = rax;
    Reg64 reg_dst = r8;
    Reg64 reg_injector_table = r9;
    Reg64 reg_diff_dst = r10;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Opmask injector_mask = Opmask(1);

    Xmm xmm_src = Xmm(1);
    Vmm vmm_src = Vmm(1);
    Xmm xmm_diff_dst = Xmm(2);
    Vmm vmm_diff_dst = Vmm(2);
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> eltwise_injector_;

    /* bf16 support */
    Zmm bf16_emu_reserv_1 = Zmm(26);
    Zmm bf16_emu_reserv_2 = Zmm(27);
    Zmm bf16_emu_reserv_3 = Zmm(28);
    Reg64 bf16_emu_scratch = r14;
    Zmm bf16_emu_reserv_5 = Zmm(29);

    Opmask k_tail_mask = k6;

    std::unique_ptr<jit_bf16_injector_t> bf16_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

} // namespace

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());

    bool ok = mayiuse(isa) && is_fwd() && src_md()->data_type == d_type
            && IMPLICATION(src_md()->data_type == data_type::bf16,
                    mayiuse(avx512_core))
            && !has_zero_dim_memory()
            && data_d.is_dense(true)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!data_d.is_dense(), is_zero_preserved())
            && attr()->has_default_values();
    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    dst += data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = dst + start;
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());

    bool ok = mayiuse(isa) && !is_fwd()
            && utils::everyone_is(
                    d_type, src_md()->data_type, diff_src_md()->data_type)
            && IMPLICATION(src_md()->data_type == data_type::bf16,
                    mayiuse(avx512_core))
            && !has_zero_dim_memory() && set_default_formats_common()
            && data_d.is_dense(true)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!data_d.is_dense(), is_zero_preserved())
            && data_d == memory_desc_wrapper(diff_dst_md())
            && attr()->has_default_values();
    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    auto src = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = diff_src + start;
        args.diff_dst = diff_dst + start;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_eltwise_fwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_core, data_type::bf16>;

template struct jit_uni_eltwise_bwd_t<sse41, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_core, data_type::bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
