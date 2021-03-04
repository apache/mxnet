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

#ifndef CPU_REF_BINARY_HPP
#define CPU_REF_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src0_type, data_type_t src1_type = src0_type,
        data_type_t dst_type = src0_type>
struct ref_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const bool ok = src0_type == src_md(0)->data_type
                    && src1_type == src_md(1)->data_type
                    && dst_type == dst_md()->data_type
                    && platform::has_data_type_support(src0_type)
                    && platform::has_data_type_support(src1_type)
                    && platform::has_data_type_support(dst_type)
                    && set_default_params() == status::success
                    && IMPLICATION(utils::one_of(src0_type, f32, bf16),
                            attr()->has_default_values(sm::post_ops))
                    && IMPLICATION(utils::one_of(src0_type, s8, u8),
                            attr()->has_default_values(
                                    sm::post_ops | sm::scales))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());
            if (!ok) return status::unimplemented;

            return status::success;
        }

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }
    };

    ref_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        return status::success;
    }

    using src0_data_t = typename prec_traits<src0_type>::type;
    using src1_data_t = typename prec_traits<src1_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
