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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

#include "primitive.hpp"
#include "primitive_desc.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;

dnnl_primitive_desc::dnnl_primitive_desc(primitive_desc_t *pd, engine_t *engine)
    : pd_(pd), engine_(engine) {}

dnnl_primitive_desc::dnnl_primitive_desc(
        const std::shared_ptr<primitive_desc_t> &pd, engine_t *engine)
    : pd_(pd), engine_(engine) {}

status_t dnnl_primitive_desc::create_primitive_iface(
        primitive_iface_t **primitive_iface) const {
    // Step 1: create impl::primitive_t or get it from primitive cache
    std::shared_ptr<primitive_t> p;
    auto status = pd_->create_primitive(p, engine(), false);
    if (status != status::success) return status;
    // Step 2: create primitive_iface_t, init and return it to user
    primitive_iface_t *p_iface = nullptr;
    CHECK(safe_ptr_assign(p_iface, new primitive_iface_t(p, engine())));
    status = p_iface->init();
    if (status != status::success) {
        p_iface->release();
        return status;
    }
    (*primitive_iface) = p_iface;
    return status::success;
}

const std::shared_ptr<primitive_desc_t> &dnnl_primitive_desc::impl() const {
    return pd_;
}

dnnl::impl::engine_t *dnnl_primitive_desc::engine() const {
    return engine_;
}
const dnnl::impl::primitive_attr_t *dnnl_primitive_desc::attr() const {
    return pd_->attr();
}

const char *dnnl_primitive_desc::info() const {
    return pd_->info(engine_);
}

dnnl::impl::engine_t *dnnl_primitive_desc::src_engine() const {
    return engine_;
}
dnnl::impl::engine_t *dnnl_primitive_desc::dst_engine() const {
    return engine_;
}

dnnl::impl::engine_t *dnnl_primitive_desc::scratchpad_engine() const {
    return engine_;
}

status_t dnnl_primitive_desc::query(query_t what, int idx, void *result) const {
    auto status = status::success;
    if (what == query::engine) {
        *(engine_t **)result = engine();
    } else {
        status = pd_->query(what, idx, result);
    }
    return status;
}

status_t dnnl_primitive_desc_get_attr(
        const primitive_desc_iface_t *primitive_desc_iface,
        const primitive_attr_t **attr) {
    if (utils::any_null(primitive_desc_iface, attr)) return invalid_arguments;

    *attr = primitive_desc_iface->attr();
    return success;
}
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
