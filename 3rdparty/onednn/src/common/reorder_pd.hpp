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

#ifndef COMMON_REORDER_PD_HPP
#define COMMON_REORDER_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "primitive_attr.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct reorder_primitive_desc_iface_t : public dnnl_primitive_desc {
    reorder_primitive_desc_iface_t(primitive_desc_t *pd, engine_t *engine,
            engine_t *src_engine, engine_t *dst_engine)
        : dnnl_primitive_desc(pd, engine)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine)
        , scratchpad_engine_(nullptr) {}

    reorder_primitive_desc_iface_t(const std::shared_ptr<primitive_desc_t> &pd,
            engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
        : dnnl_primitive_desc(pd, engine)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine)
        , scratchpad_engine_(nullptr) {}

    dnnl::impl::engine_t *src_engine() const override { return src_engine_; }
    dnnl::impl::engine_t *dst_engine() const override { return dst_engine_; }

    dnnl::impl::engine_t *scratchpad_engine() const override {
        return scratchpad_engine_;
    }

    dnnl::impl::status_t query(
            dnnl::impl::query_t what, int idx, void *result) const override {
        auto status = dnnl::impl::status::success;
        switch (what) {
            case dnnl::impl::query::reorder_src_engine:
                *(dnnl::impl::engine_t **)result = src_engine();
                break;
            case dnnl::impl::query::reorder_dst_engine:
                *(dnnl::impl::engine_t **)result = dst_engine();
                break;
            default: status = dnnl_primitive_desc::query(what, idx, result);
        }
        return status;
    }

    status_t create_primitive_iface(
            primitive_iface_t **primitive_iface) const override {
        // Step 1: create impl::primitive_t or get it from primitive cache
        std::shared_ptr<primitive_t> p;
        auto status = pd_->create_primitive(p, engine(), false);
        if (status != status::success) return status;
        // Step 2: create primitive_iface_t, init and return it to user
        primitive_iface_t *p_iface = nullptr;
        CHECK(safe_ptr_assign(p_iface,
                new primitive_iface_t(p, engine(), src_engine_, dst_engine_)));
        status = p_iface->init();
        if (status != status::success) {
            p_iface->release();
            return status;
        }
        (*primitive_iface) = p_iface;
        return status::success;
    }

private:
    dnnl::impl::engine_t *src_engine_;
    dnnl::impl::engine_t *dst_engine_;
    dnnl::impl::engine_t *scratchpad_engine_;
};

struct reorder_pd_t : public primitive_desc_t {
    reorder_pd_t(const primitive_attr_t *attr, engine_kind_t src_engine_kind,
            const memory_desc_t *src_md, engine_kind_t dst_engine_kind,
            const memory_desc_t *dst_md)
        : primitive_desc_t(attr, primitive_kind::reorder)
        , src_md_(*src_md)
        , dst_md_(*dst_md) {

        // Fill a desc that is intended for internal use only
        desc_ = reorder_desc_t();
        desc_.primitive_kind = primitive_kind::reorder;
        desc_.src_md = src_md_;
        desc_.dst_md = dst_md_;
        desc_.src_engine_kind = src_engine_kind;
        desc_.dst_engine_kind = dst_engine_kind;
    }

    const reorder_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_FROM) return arg_usage_t::input;

        if (arg == DNNL_ARG_TO) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_FROM: return src_md(0);
            case DNNL_ARG_TO: return dst_md(0);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &src_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return 1; }

    float alpha() const { return attr()->output_scales_.scales_[0]; }
    float beta() const {
        const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
        return sum_idx == -1 ? 0 : attr()->post_ops_.entry_[sum_idx].sum.scale;
    }

protected:
    reorder_desc_t desc_;
    memory_desc_t src_md_;
    memory_desc_t dst_md_;
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
