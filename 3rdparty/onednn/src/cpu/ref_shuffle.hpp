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

#ifndef CPU_REF_SHUFFLE_HPP
#define CPU_REF_SHUFFLE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_shuffle_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <int data_type_size>
struct ref_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_shuffle_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;

            const data_type_t data_type = data_md()->data_type;
            bool ok = data_type_size == types::data_type_size(data_type)
                    && platform::has_data_type_support(data_type)
                    && attr()->has_default_values()
                    && IMPLICATION(!is_fwd(), set_default_formats_common());
            if (!ok) return status::unimplemented;

            if (ndims() == 5) {
                dat_tag_ = memory_desc_matches_one_of_tag(
                        *data_md(), nCdhw16c, nCdhw8c, nCdhw4c, ncdhw, ndhwc);
            } else if (ndims() == 4) {
                dat_tag_ = memory_desc_matches_one_of_tag(
                        *data_md(), nChw16c, nChw8c, nChw4c, nchw, nhwc);
            } else
                dat_tag_ = any;

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_shuffle_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const int axis_size = pd()->axis_size();
        const int group_size = pd()->group_size();
        const int transpose_row
                = pd()->is_fwd() ? group_size : axis_size / group_size;
        const int transpose_col
                = pd()->is_fwd() ? axis_size / group_size : group_size;
        rev_transposed_ = (int *)malloc(
                axis_size * sizeof(int), platform::get_cache_line_size());
        if (rev_transposed_ == nullptr) return dnnl_out_of_memory;
        parallel_nd(transpose_col, transpose_row, [&](int i, int j) {
            rev_transposed_[j * transpose_col + i] = i * transpose_row + j;
        });
        return dnnl_success;
    }

    ~ref_shuffle_t() { free(rev_transposed_); }

    typedef typename typesize_traits<data_type_size>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        switch (pd()->dat_tag_) {
            case nCdhw16c: execute_<nCdhw16c>(ctx); break;
            case nChw16c: execute_<nChw16c>(ctx); break;
            case nCdhw8c: execute_<nCdhw8c>(ctx); break;
            case nChw8c: execute_<nChw8c>(ctx); break;
            case nCdhw4c: execute_<nCdhw4c>(ctx); break;
            case nChw4c: execute_<nChw4c>(ctx); break;
            case ncdhw: execute_<ncdhw>(ctx); break;
            case nchw: execute_<nchw>(ctx); break;
            case ndhwc: execute_<ndhwc>(ctx); break;
            case nhwc: execute_<nhwc>(ctx); break;
            default: execute_<any>(ctx); break;
        }
        return status::success;
    }

private:
    template <format_tag_t tag>
    void execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    int *rev_transposed_ = nullptr;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
