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

#ifndef COMMON_SUM_PD_HPP
#define COMMON_SUM_PD_HPP

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

#include "utils.hpp"

namespace dnnl {
namespace impl {

struct sum_pd_t : public primitive_desc_t {
    sum_pd_t(const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
            const float *scales, const memory_desc_t *src_mds)
        : primitive_desc_t(attr, primitive_kind::sum), n_(n), dst_md_(*dst_md) {
        scales_.reserve(n_);
        for (int i = 0; i < n_; ++i)
            scales_.push_back(scales[i]);
        src_mds_.reserve(n_);
        for (int i = 0; i < n_; ++i)
            src_mds_.push_back(src_mds[i]);

        // Fill a desc that is intended for internal use only
        desc_ = sum_desc_t();
        desc_.primitive_kind = primitive_kind::sum;
        desc_.dst_md = dst_md_;
        desc_.n = n_;
        desc_.scales = scales_;
        desc_.src_mds = src_mds_;
    }

    const sum_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg >= DNNL_ARG_MULTIPLE_SRC
                && arg < DNNL_ARG_MULTIPLE_SRC + n_inputs())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        int src_index = arg - DNNL_ARG_MULTIPLE_SRC;
        if (src_index >= 0 && src_index < n_inputs()) return src_md(src_index);
        if (arg == DNNL_ARG_DST) return dst_md(0);
        return primitive_desc_t::arg_md(arg);
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index < n_inputs() ? &src_mds_[index] : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_acc_md() const {
        return need_output_reorder() ? &dst_acc_md_ : &dst_md_;
    }

    int n_inputs() const override { return n_; }
    int n_outputs() const override { return 1; }

    const float *scales() const { return &scales_[0]; }

    bool need_output_reorder() const { return dst_md()->data_type != dnnl_f32; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(dst_md()).has_zero_dim();
    }

protected:
    int n_;
    std::vector<float> scales_;
    memory_desc_t dst_md_, dst_acc_md_;
    std::vector<memory_desc_t> src_mds_;

protected:
    sum_desc_t desc_;
    /* inits dst_md_ in simple cases. The call may fail. */
    status_t init(engine_t *engine) {
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(&src_mds_[i]);
            if (!src_d.is_blocking_desc() || src_d.is_additional_buffer())
                return status::unimplemented;
        }
        bool ok = true && set_default_params() == status::success
                && attr()->has_default_values();
        if (!ok) return status::unimplemented;

        // use f32 accumulator to handle float scales w/o accuracy loss
        if (need_output_reorder()) {
            dst_acc_md_ = dst_md_;
            dst_acc_md_.data_type = dnnl_f32;
        }

        return status::success;
    }

    status_t set_default_params() {
        if (dst_md_.format_kind != format_kind::any) return status::success;

        /* The stupidest ever heuristics (but not the same as we had before):
         *  - Pick the first non-plain format;
         *  - If all formats are plain, pick the format of the first input
         */
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(src_mds_[i]);
            if (!src_d.is_plain() && src_d.is_blocking_desc()) {
                return memory_desc_init_by_blocking_desc(
                        dst_md_, src_d.blocking_desc());
            }
        }

        if (src_mds_[0].format_kind != format_kind::blocked)
            return status::unimplemented;

        memory_desc_init_by_md_and_dt(dst_md_, src_mds_[0], dst_md_.data_type);

        return status::success;
    }
};

#define DECLARE_SUM_PD_t(impl_name, ...) \
    static status_t create(sum_pd_t **sum_pd, engine_t *engine, \
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n, \
            const float *scales, const memory_desc_t *src_mds) { \
        using namespace status; \
        auto _pd = new pd_t(attr, dst_md, n, scales, src_mds); \
        if (_pd == nullptr) return out_of_memory; \
        if (_pd->init(engine) != success) { \
            delete _pd; \
            return unimplemented; \
        } \
        _pd->init_scratchpad_md(); \
        return safe_ptr_assign(*sum_pd, _pd); \
    } \
    status_t create_primitive(std::shared_ptr<primitive_t> &primitive, \
            engine_t *engine, bool is_primitive_nested) const override { \
        return primitive_t::create_primitive_common<__VA_ARGS__, pd_t>( \
                primitive, this, engine, false, is_primitive_nested); \
    } \
    pd_t *clone() const override { \
        auto new_pd = utils::make_unique<pd_t>(*this); \
        if (!new_pd->is_initialized()) return nullptr; \
        return new_pd.release(); \
    } \
    const char *name() const override { return impl_name; } \
    std::type_index impl_id() const override { return typeid(pd_t); }

#define DECLARE_SUM_PD_T(impl_name, ...) \
    DECLARE_SUM_PD_t(impl_name, __VA_ARGS__)

} // namespace impl
} // namespace dnnl

#endif
