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

#ifndef CPU_X64_JIT_UNI_SOFTMAX_HPP
#define CPU_X64_JIT_UNI_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_softmax_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace softmax_impl {
template <cpu_isa_t isa>
struct driver_t;
}

template <cpu_isa_t isa>
struct jit_uni_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_softmax_fwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            auto data_type = src_d.data_type();
            auto is_dense = [&]() {
                const auto &bd = src_d.blocking_desc();

                if (!src_d.is_dense(true) || !src_d.only_padded_dim(axis()))
                    return false;

                // It is fine to use float here as the kernel uses halfs of
                // vector registers.
                const auto blk_size = cpu_isa_traits<isa>::vlen / sizeof(float);
                if (src_d.is_plain())
                    return bd.strides[axis()] == 1;
                else {
                    // 31 is a general limit, 2 is for unroll_regs_ = 4;
                    const size_t max_stride = (1LL << (31 - 2)) - 1;
                    const int last_blk = bd.inner_nblks - 1;
                    return true && bd.inner_blks[last_blk] == blk_size
                            && bd.inner_idxs[last_blk] == axis()
                            && sizeof(float) * bd.strides[axis()] < max_stride;
                }
            };

            using namespace data_type;
            bool ok = src_d == dst_d && mayiuse(isa) && is_fwd()
                    && !has_zero_dim_memory()
                    && utils::one_of(data_type, f32, bf16)
                    && IMPLICATION(data_type == bf16,
                            // extra check for isa is required because
                            // the avx512_common version may reject a
                            // problem because it is blocked by 8
                            // instead of 16.
                            isa >= avx512_common && mayiuse(avx512_core))
                    && is_dense() // not dense impl can be easily done
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        };
    };

    jit_uni_softmax_fwd_t(const pd_t *apd);
    ~jit_uni_softmax_fwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    softmax_impl::driver_t<isa> *softmax_driver_;
};

template <cpu_isa_t isa>
struct jit_uni_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_softmax_bwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            auto data_type = dst_d.data_type();
            auto is_dense = [&]() {
                const auto &bd = dst_d.blocking_desc();

                if (!dst_d.is_dense(true) || !dst_d.only_padded_dim(axis()))
                    return false;

                // It is fine to use float here as the kernel uses halfs of
                // vector registers.
                const auto blk_size = cpu_isa_traits<isa>::vlen / sizeof(float);
                if (dst_d.is_plain())
                    return bd.strides[axis()] == 1;
                else {
                    // 31 is a general limit, 2 is for unroll_regs_ = 4;
                    const size_t max_stride = (1LL << (31 - 2)) - 1;
                    const int last_blk = bd.inner_nblks - 1;
                    return true && bd.inner_blks[last_blk] == blk_size
                            && bd.inner_idxs[last_blk] == axis()
                            && sizeof(float) * bd.strides[axis()] < max_stride;
                }
            };

            using namespace data_type;
            bool ok = dst_d == diff_dst_d && dst_d == diff_src_d && mayiuse(isa)
                    && !is_fwd() && !has_zero_dim_memory()
                    && utils::one_of(data_type, f32, bf16)
                    && IMPLICATION(data_type == bf16,
                            // extra check for isa is required because
                            // the avx512_common version may reject a
                            // problem because it is blocked by 8
                            // instead of 16.
                            isa >= avx512_common && mayiuse(avx512_core))
                    && set_default_formats_common()
                    && is_dense() // not dense impl can be easily done
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        };
    };

    jit_uni_softmax_bwd_t(const pd_t *apd);
    ~jit_uni_softmax_bwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    softmax_impl::driver_t<isa> *softmax_driver_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
