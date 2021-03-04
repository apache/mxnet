/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_NSPC_BATCH_NORMALIZATION_HPP
#define CPU_NSPC_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_batch_normalization_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct nspc_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nspc_bnorm:any", nspc_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;

            bool ok = is_fwd() && !has_zero_dim_memory()
                    && src_md()->data_type == d_type
                    && platform::has_data_type_support(d_type)
                    && check_scale_shift_data_type()
                    && memory_desc_matches_tag(*src_md(), format_tag::nhwc)
                    && (attr()->has_default_values()
                            || this->with_relu_post_op());
            if (!ok) return status::unimplemented;

            if (is_training() && fuse_norm_relu()) init_default_ws(8);

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            using namespace data_type;

            auto scratchpad = scratchpad_registry().registrar();
            if (!stats_is_src()) {
                const size_t stats_buf_sz
                        = nstl::max(C(), dim_t(16)) * dnnl_get_max_threads();
                scratchpad.template book<acc_data_t>(
                        key_bnorm_reduction, stats_buf_sz);
                scratchpad.template book<acc_data_t>(
                        key_bnorm_tmp_mean, stats_buf_sz);
                scratchpad.template book<acc_data_t>(
                        key_bnorm_tmp_var, stats_buf_sz);
            }
            if (d_type == bf16) {
                const int simd_w = 16;
                const int nbufs = 2;
                const size_t bf16cvt_buf_sz = nbufs * dnnl_get_max_threads()
                        * utils::rnd_up(C(), simd_w);
                scratchpad.template book<acc_data_t>(
                        key_bnorm_bf16cvt, bf16cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<d_type>::type data_t;
    typedef float acc_data_t;

    nspc_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~nspc_batch_normalization_fwd_t() {}

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <data_type_t d_type>
struct nspc_batch_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nspc_bnorm:any", nspc_batch_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;

            bool ok = is_bwd() && !has_zero_dim_memory()
                    && set_default_formats_common()
                    && utils::everyone_is(d_type, src_md()->data_type,
                            diff_src_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && check_scale_shift_data_type()
                    && memory_desc_matches_tag(*src_md(), format_tag::nhwc)
                    && memory_desc_matches_tag(*diff_src_md(), format_tag::nhwc)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (fuse_norm_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            using namespace data_type;

            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<acc_data_t>(
                    key_bnorm_reduction, 2 * C() * dnnl_get_max_threads());
            scratchpad.template book<acc_data_t>(key_bnorm_tmp_diff_ss,
                    2 * C() * (dnnl_get_max_threads() + 1));
            if (d_type == bf16) {
                const int simd_w = 16;
                const int nbufs = 2 + !use_global_stats();
                const size_t bf16cvt_buf_sz = nbufs * dnnl_get_max_threads()
                        * utils::rnd_up(C(), simd_w);
                scratchpad.template book<acc_data_t>(
                        key_bnorm_bf16cvt, bf16cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<d_type>::type data_t;
    typedef float acc_data_t;

    nspc_batch_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~nspc_batch_normalization_bwd_t() {}

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
