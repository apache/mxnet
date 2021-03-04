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

#ifndef COMMON_LAYER_NORMALIZATION_PD_HPP
#define COMMON_LAYER_NORMALIZATION_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct layer_normalization_fwd_pd_t;

struct layer_normalization_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::layer_normalization;

    layer_normalization_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , data_md_(desc_.data_desc)
        , stat_md_(desc_.stat_desc)
        , scaleshift_md_(desc_.data_scaleshift_desc) {}

    const layer_normalization_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::layer_normalization_d:
                *(const layer_normalization_desc_t **)result = desc();
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common layer_normalization aux functions */
    int ndims() const { return desc_.data_desc.ndims; }
    dim_t across_axis() const {
        return utils::array_product(desc_.data_desc.dims, ndims() - 1);
    }
    dim_t norm_axis() const { return desc_.data_desc.dims[ndims() - 1]; }

    bool stats_are_src() const { return desc_.flags & dnnl_use_global_stats; }
    bool stats_are_tmp() const { return !(stats_are_src() || is_training()); }

    bool use_scaleshift() const { return desc_.flags & dnnl_use_scaleshift; }
    bool use_global_stats() const {
        return desc_.flags & dnnl_use_global_stats;
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }
    bool is_bwd() const { return !this->is_fwd(); }
    bool is_training() const {
        return desc_.prop_kind == prop_kind::forward_training;
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(desc_.data_desc).has_zero_dim();
    }

    const memory_desc_t *stat_md() const { return &stat_md_; }

protected:
    layer_normalization_desc_t desc_;
    const layer_normalization_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t data_md_;
    memory_desc_t stat_md_;
    memory_desc_t scaleshift_md_;

    bool set_default_stat_md_format(const memory_desc_t &data_md) {
        if (stat_md_.format_kind != format_kind::any) return true;

        // data memory desc in non-blocked memory format is unsupported
        if (data_md.format_kind != format_kind::blocked) return false;

        // if the normalization axis is blocked, fallback to plain format
        bool is_norm_dim_blocked = false;
        for (int d = 0; d < data_md.format_desc.blocking.inner_nblks; ++d)
            is_norm_dim_blocked |= data_md.format_desc.blocking.inner_idxs[d]
                    == ndims() - 1;
        if (is_norm_dim_blocked)
            return memory_desc_init_by_strides(stat_md_, nullptr)
                    == status::success;

        // the default memory format for stat is derived from data_md by
        // dropping the normalization dimension and keeping the physical order
        // of other dimensions (preserving the blocked structure if any)
        return memory_desc_init_by_blocking_desc(
                       stat_md_, data_md.format_desc.blocking)
                == status::success;
    }

private:
    const memory_desc_t &data_desc() const { return desc_.data_desc; }
};

struct layer_normalization_fwd_pd_t : public layer_normalization_pd_t {
    typedef layer_normalization_fwd_pd_t base_class;
    typedef layer_normalization_fwd_pd_t hint_class;

    layer_normalization_fwd_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : layer_normalization_pd_t(adesc, attr, hint_fwd_pd) {}

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (utils::one_of(arg, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE)) {
            if (stats_are_src()) return arg_usage_t::input;
            if (!stats_are_src() && is_training()) return arg_usage_t::output;
            return arg_usage_t::unused;
        }

        if (arg == DNNL_ARG_SCALE_SHIFT && use_scaleshift())
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0);
            case DNNL_ARG_MEAN: return stats_are_src() ? src_md(1) : dst_md(1);
            case DNNL_ARG_VARIANCE:
                return stats_are_src() ? src_md(2) : dst_md(2);
            case DNNL_ARG_SCALE_SHIFT: return weights_md(0);
            default: return layer_normalization_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        if (index == 0) return &data_md_;
        if (stats_are_src() && (index == 1 || index == 2)) return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(int index = 0) const override {
        if (index == 0) return &data_md_;
        if (!stats_are_src() && is_training() && (index == 1 || index == 2))
            return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(int index = 0) const override {
        return index == 0 ? &scaleshift_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 1 + 2 * stats_are_src() + use_scaleshift();
    }
    int n_outputs() const override {
        return 1 + 2 * (!stats_are_src()) * is_training();
    }

protected:
    bool set_default_formats_common() {
        return set_default_stat_md_format(data_md_);
    }

    bool check_scale_shift_data_type() const {
        return IMPLICATION(
                use_scaleshift(), weights_md()->data_type == data_type::f32);
    }
};

struct layer_normalization_bwd_pd_t : public layer_normalization_pd_t {
    typedef layer_normalization_bwd_pd_t base_class;
    typedef layer_normalization_fwd_pd_t hint_class;

    layer_normalization_bwd_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : layer_normalization_pd_t(adesc, attr, hint_fwd_pd)
        , diff_data_md_(desc_.diff_data_desc)
        , diff_scaleshift_md_(desc_.diff_data_scaleshift_desc) {}

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE,
                    DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_SCALE_SHIFT && use_scaleshift())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_DIFF_SCALE_SHIFT && use_scaleshift())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_MEAN: return src_md(1);
            case DNNL_ARG_VARIANCE: return src_md(2);
            case DNNL_ARG_SCALE_SHIFT: return weights_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
            case DNNL_ARG_DIFF_SCALE_SHIFT: return diff_weights_md(0);
            default: return layer_normalization_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &data_md_ : index <= 2 ? &stat_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return (index == 0) ? &data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(int index = 0) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }

    const memory_desc_t *weights_md(int index = 0) const override {
        return index == 0 ? &scaleshift_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_weights_md(int index = 0) const override {
        return index == 0 ? &diff_scaleshift_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 4 + use_scaleshift(); }
    int n_outputs() const override {
        return 1 + (desc_.prop_kind == prop_kind::backward);
    }

protected:
    memory_desc_t diff_data_md_;
    memory_desc_t diff_scaleshift_md_;

    bool set_default_formats_common() {
        return IMPLICATION(diff_data_md_.format_kind == format_kind::any,
                       memory_desc_init_by_md_and_dt(
                               diff_data_md_, data_md_, diff_data_md_.data_type)
                               == status::success)
                && set_default_stat_md_format(diff_data_md_);
    }

    bool check_scale_shift_data_type() const {
        return IMPLICATION(use_scaleshift(),
                utils::everyone_is(data_type::f32, weights_md()->data_type,
                        diff_weights_md()->data_type));
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
