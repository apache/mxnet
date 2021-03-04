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

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "sum_pd.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

status_t dnnl_sum_primitive_desc_create(primitive_desc_iface_t **sum_pd_iface,
        const memory_desc_t *dst_md, int n, const float *scales,
        const memory_desc_t *src_mds, const primitive_attr_t *attr,
        engine_t *engine) {
    bool args_ok = !any_null(sum_pd_iface, src_mds, scales) && n > 0;
    if (!args_ok) return invalid_arguments;

    if (attr == nullptr) attr = &default_attr();

    const int ndims = src_mds[0].ndims;
    const dims_t &dims = src_mds[0].dims;
    if (memory_desc_wrapper(src_mds[0]).has_runtime_dims_or_strides())
        return unimplemented;

    for (int i = 1; i < n; ++i) {
        if (src_mds[i].ndims != ndims) return invalid_arguments;
        if (memory_desc_wrapper(src_mds[i]).has_runtime_dims_or_strides())
            return unimplemented;
        for (int d = 0; d < ndims; ++d) {
            if (src_mds[i].dims[d] != dims[d]) return invalid_arguments;
        }
    }

    memory_desc_t dummy_dst_md;
    if (dst_md) {
        if (dst_md->ndims != ndims) return invalid_arguments;
        if (memory_desc_wrapper(dst_md).has_runtime_dims_or_strides())
            return unimplemented;
        for (int d = 0; d < ndims; ++d) {
            if (dst_md->dims[d] != dims[d]) return invalid_arguments;
        }
    } else {
        dummy_dst_md = src_mds[0];
        dummy_dst_md.format_kind = format_kind::any;
        dst_md = &dummy_dst_md;
    }

    for (auto s = engine->get_sum_implementation_list(); *s; ++s) {
        sum_pd_t *sum_pd = nullptr;
        if ((*s)(&sum_pd, engine, attr, dst_md, n, scales, src_mds)
                == success) {
            auto status = safe_ptr_assign(
                    *sum_pd_iface, new primitive_desc_iface_t(sum_pd, engine));
            if (status != status::success) delete sum_pd;
            return status;
        }
    }
    return unimplemented;
}
