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

#ifndef COMMON_SOFTMAX_PD_HPP
#define COMMON_SOFTMAX_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

namespace dnnl {
namespace impl {

struct softmax_fwd_pd_t;

struct softmax_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::softmax;

    softmax_pd_t(const softmax_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , data_md_(desc_.data_desc) {}

    const softmax_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::softmax_d:
                *(const softmax_desc_t **)result = desc();
                break;
            case query::logsoftmax_d:
                *(const logsoftmax_desc_t **)result = desc();
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common softmax aux functions */

    dim_t MB() const { return data_desc().dims[0]; }
    dim_t C() const { return data_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? data_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_desc().dims[ndims() - 1] : 1; }

    dim_t outer_size() const {
        return utils::array_product(data_desc().dims, axis());
    }
    dim_t axis_size() const { return data_desc().dims[axis()]; }
    dim_t inner_size() const {
        return utils::array_product(
                data_desc().dims + axis() + 1, ndims() - 1 - axis());
    }

    dim_t outer_stride() const {
        const memory_desc_wrapper data_d(data_desc());
        return axis() > 0 ? data_d.blocking_desc().strides[axis() - 1] : 1;
    }

    int axis() const { return desc_.softmax_axis; }
    int ndims() const { return data_desc().ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(data_desc()).has_zero_dim();
    }

    bool is_softmax() const {
        return desc()->primitive_kind == primitive_kind::softmax;
    }
    bool is_logsoftmax() const {
        return desc()->primitive_kind == primitive_kind::logsoftmax;
    }

protected:
    softmax_desc_t desc_;
    const softmax_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t data_md_;

private:
    const memory_desc_t &data_desc() const { return desc_.data_desc; }
};

struct softmax_fwd_pd_t : public softmax_pd_t {
    typedef softmax_fwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    softmax_fwd_pd_t(const softmax_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd) {}

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }
};

struct softmax_bwd_pd_t : public softmax_pd_t {
    typedef softmax_bwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    softmax_bwd_pd_t(const softmax_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd)
        , diff_data_md_(desc_.diff_desc) {}

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_DST, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_DST: return dst_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + (!types::is_zero_md(workspace_md()));
    }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_data_md_;

    bool set_default_formats_common() {
        if (diff_data_md_.format_kind != format_kind::any) return true;

        return memory_desc_init_by_md_and_dt(
                       diff_data_md_, data_md_, diff_data_md_.data_type)
                == status::success;
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
