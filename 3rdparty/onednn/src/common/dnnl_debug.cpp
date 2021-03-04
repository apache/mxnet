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

#include <assert.h>
#include <cinttypes>
#include <stdio.h>

#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define DPRINT(...) \
    do { \
        int l = snprintf(str + written_len, str_len, __VA_ARGS__); \
        if (l < 0) return l; \
        if ((size_t)l >= str_len) return -1; \
        written_len += l; \
        str_len -= l; \
    } while (0)

const char *dnnl_runtime2str(unsigned runtime) {
    switch (runtime) {
        case DNNL_RUNTIME_NONE: return "none";
        case DNNL_RUNTIME_SEQ: return "sequential";
        case DNNL_RUNTIME_OMP: return "OpenMP";
        case DNNL_RUNTIME_TBB: return "TBB";
        case DNNL_RUNTIME_OCL: return "OpenCL";
        case DNNL_RUNTIME_THREADPOOL: return "threadpool";
#ifdef DNNL_SYCL_DPCPP
        case DNNL_RUNTIME_SYCL: return "DPC++";
#endif
#ifdef DNNL_SYCL_COMPUTECPP
        case DNNL_RUNTIME_SYCL: return "SYCL";
#endif
        default: return "unknown";
    }
}

int dnnl_md2fmt_str(
        char *str, size_t str_len, const dnnl_memory_desc_t *mdesc) {
    using namespace dnnl::impl;

    if (str == nullptr || str_len <= 1u) return -1;

    int written_len = 0;

    if (mdesc == nullptr) {
        DPRINT("%s::%s::", dnnl_dt2str(data_type::undef),
                dnnl_fmt_kind2str(format_kind::undef));
        return written_len;
    }

    memory_desc_wrapper md(mdesc);

    DPRINT("%s:", dnnl_dt2str(md.data_type()));

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < md.ndims(); ++d) {
        if (md.dims()[d] != md.padded_dims()[d]) padded_dims = true;
        if (md.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = md.offset0();
    DPRINT("%s%s%s:", padded_dims ? "p" : "", padded_offsets ? "o" : "",
            offset0 ? "0" : "");

    DPRINT("%s:", dnnl_fmt_kind2str(md.format_kind()));

    if (!md.is_blocking_desc()) {
        /* TODO: extend */
        DPRINT("%s:", "");
    } else {
        const auto &blk = md.blocking_desc();

        dims_t blocks = {0};
        md.compute_blocks(blocks);

        char dim_chars[DNNL_MAX_NDIMS + 1];

        dims_t ou_blocks = {0};
        utils::array_copy(ou_blocks, md.padded_dims(), md.ndims());

        bool plain = true;
        for (int d = 0; d < md.ndims(); ++d) {
            dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + (char)d;
            if (blocks[d] != 1) plain = false;
            ou_blocks[d] /= blocks[d];
        }

        dims_t strides;
        utils::array_copy(strides, blk.strides, md.ndims());

        utils::simultaneous_sort(strides, ou_blocks, dim_chars, md.ndims(),
                [](dim_t a, dim_t b) { return b - a; });

        dim_chars[md.ndims()] = '\0';
        DPRINT("%s", dim_chars);

        if (!plain) {
            for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
                DPRINT("%d%c", (int)blk.inner_blks[iblk],
                        'a' + (char)blk.inner_idxs[iblk]);
            }
        }

        DPRINT("%s", ":");
    }

    DPRINT("f%lx", (long)md.extra().flags);

    return written_len;
}

int dnnl_md2dim_str(
        char *str, size_t str_len, const dnnl_memory_desc_t *mdesc) {
    using namespace dnnl::impl;

    if (str == nullptr || str_len <= 1) return -1;

    int written_len = 0;

    if (mdesc == nullptr || mdesc->ndims == 0) {
        DPRINT("%s", "");
        return written_len;
    }

    memory_desc_wrapper md(mdesc);

#define DPRINT_RT(val) \
    do { \
        if (is_runtime_value(val)) \
            DPRINT("*"); \
        else \
            DPRINT("%" PRId64, (val)); \
    } while (0)

    for (int d = 0; d < md.ndims() - 1; ++d) {
        DPRINT_RT(md.dims()[d]);
        DPRINT("x");
    }
    DPRINT_RT(md.dims()[md.ndims() - 1]);

#undef DPRINT_RT

    return written_len;
}

#undef DPRINT
