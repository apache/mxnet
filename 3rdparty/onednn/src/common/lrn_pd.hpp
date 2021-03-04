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

#ifndef COMMON_LRN_PD_HPP
#define COMMON_LRN_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

namespace dnnl {
namespace impl {

struct lrn_fwd_pd_t;

struct lrn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::lrn;

    lrn_pd_t(const lrn_desc_t *adesc, const primitive_attr_t *attr,
            const lrn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , data_md_(desc_.data_desc)
        , ws_md_() {}

    const lrn_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::lrn_d: *(const lrn_desc_t **)result = desc(); break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common lrn aux functions */

    dim_t MB() const { return data_desc().dims[0]; }
    dim_t C() const { return data_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? data_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_desc().dims[ndims() - 1] : 1; }

    int ndims() const { return data_desc().ndims; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(desc_.data_desc).has_zero_dim();
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    lrn_desc_t desc_;
    const lrn_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t data_md_;
    memory_desc_t ws_md_;

private:
    const memory_desc_t &data_desc() const { return desc_.data_desc; }
};

struct lrn_fwd_pd_t : public lrn_pd_t {
    typedef lrn_fwd_pd_t base_class;
    typedef lrn_fwd_pd_t hint_class;

    lrn_fwd_pd_t(const lrn_desc_t *adesc, const primitive_attr_t *attr,
            const lrn_fwd_pd_t *hint_fwd_pd)
        : lrn_pd_t(adesc, attr, hint_fwd_pd) {}

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
            default: return lrn_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }
};

struct lrn_bwd_pd_t : public lrn_pd_t {
    typedef lrn_bwd_pd_t base_class;
    typedef lrn_fwd_pd_t hint_class;

    lrn_bwd_pd_t(const lrn_desc_t *adesc, const primitive_attr_t *attr,
            const lrn_fwd_pd_t *hint_fwd_pd)
        : lrn_pd_t(adesc, attr, hint_fwd_pd)
        , diff_data_md_(desc_.diff_data_desc) {}

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            default: return lrn_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
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
