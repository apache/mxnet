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

#ifndef CPU_REF_RESAMPLING_HPP
#define CPU_REF_RESAMPLING_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_resampling_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_resampling_fwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = is_fwd()
                    && utils::everyone_is(
                            data_type, src_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type)
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_resampling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_resampling_fwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void execute_forward(const exec_ctx_t &ctx) const;
};

template <impl::data_type_t data_type>
struct ref_resampling_bwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = !is_fwd()
                    && utils::everyone_is(data_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type)
                    && platform::has_data_type_support(data_type)
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_resampling_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_resampling_bwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void execute_backward(const exec_ctx_t &ctx) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
