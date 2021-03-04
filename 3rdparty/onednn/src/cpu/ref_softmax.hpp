/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_REF_SOFTMAX_HPP
#define CPU_REF_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd() && src_md()->data_type == data_type
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            const dim_t in_s = inner_size();
            const dim_t ou_s = outer_size();

            if (in_s > 1) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(
                        memory_tracking::names::key_softmax_reduction,
                        2 * in_s * ou_s);
            }
        }
    };

    ref_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        outer_size_ = pd()->outer_size();
        channels_ = pd()->axis_size();
        inner_size_ = pd()->inner_size();

        const memory_desc_wrapper data_d(pd()->src_md());
        const auto &bd = data_d.blocking_desc();

        auto axis = pd()->axis();
        dim_t axis_blk_size = 1;
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            if (bd.inner_idxs[iblk] == axis)
                axis_blk_size *= bd.inner_blks[iblk];

        use_dense_ = true && inner_size_ == 1 && data_d.is_dense(true)
                && data_d.only_padded_dim(axis)
                && bd.strides[axis] == axis_blk_size;
        return status::success;
    }

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            execute_forward_dense(ctx);
        else
            execute_forward_generic(ctx);
        return status::success;
    }

private:
    void execute_forward_dense(const exec_ctx_t &ctx) const;
    void execute_forward_generic(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    bool use_dense_;
    int outer_size_, channels_, inner_size_;
};

template <impl::data_type_t data_type>
struct ref_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && !is_fwd()
                    && utils::everyone_is(data_type, dst_md()->data_type,
                            diff_src_md()->data_type)
                    && set_default_formats_common()
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_softmax_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        outer_size_ = pd()->outer_size();
        channels_ = pd()->axis_size();
        inner_size_ = pd()->inner_size();

        const memory_desc_wrapper data_d(pd()->dst_md());
        const memory_desc_wrapper diff_d(pd()->diff_dst_md());
        const auto &bd = diff_d.blocking_desc();

        auto axis = pd()->axis();
        dim_t axis_blk_size = 1;
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            if (bd.inner_idxs[iblk] == axis)
                axis_blk_size *= bd.inner_blks[iblk];

        use_dense_ = true && inner_size_ == 1 && diff_d == data_d
                && diff_d.is_dense() && bd.strides[axis] == axis_blk_size;
        return status::success;
    }

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            execute_backward_dense(ctx);
        else
            execute_backward_generic(ctx);
        return status::success;
    }

private:
    void execute_backward_dense(const exec_ctx_t &ctx) const;
    void execute_backward_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    bool use_dense_;
    int outer_size_, channels_, inner_size_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
