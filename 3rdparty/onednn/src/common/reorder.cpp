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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "reorder_pd.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

static engine_t *get_reorder_engine(
        engine_t *src_engine, engine_t *dst_engine) {
    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    auto s_rk = src_engine->runtime_kind();
    auto d_rk = dst_engine->runtime_kind();

    if (is_native_runtime(d_rk)) return src_engine;

    if (is_native_runtime(s_rk)) return dst_engine;

    if (d_ek == engine_kind::cpu) return src_engine;

    if (s_ek == engine_kind::cpu) return dst_engine;

    assert(s_ek == engine_kind::gpu);
    assert(d_ek == engine_kind::gpu);
    return src_engine;
}

status_t dnnl_reorder_primitive_desc_create(
        primitive_desc_iface_t **reorder_pd_iface, const memory_desc_t *src_md,
        engine_t *src_engine, const memory_desc_t *dst_md, engine_t *dst_engine,
        const primitive_attr_t *attr) {
    if (any_null(reorder_pd_iface, src_engine, src_md, dst_engine, dst_md))
        return invalid_arguments;

    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    if (!IMPLICATION(s_ek != d_ek, one_of(engine_kind::cpu, s_ek, d_ek)))
        return invalid_arguments;

    auto s_mdw = memory_desc_wrapper(*src_md);
    auto d_mdw = memory_desc_wrapper(*dst_md);

    if (!s_mdw.consistent_with(d_mdw)) return invalid_arguments;

    if (attr == nullptr) attr = &default_attr();

    auto e = get_reorder_engine(src_engine, dst_engine);
    for (auto r = e->get_reorder_implementation_list(src_md, dst_md); *r; ++r) {
        reorder_pd_t *reorder_pd = nullptr;
        if ((*r)(&reorder_pd, e, attr, src_engine, src_md, dst_engine, dst_md)
                == success) {
            auto status = safe_ptr_assign(*reorder_pd_iface,
                    new reorder_primitive_desc_iface_t(
                            reorder_pd, e, src_engine, dst_engine));
            if (status != status::success) delete reorder_pd;
            return status;
        }
    }
    return unimplemented;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
