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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void prb_t::generate_oscales() {
    if (attr.oscale.is_def()) return;

    if (attr.oscale.policy == policy_t::COMMON) {
        scales = (float *)zmalloc(sizeof(float), 4);
        SAFE_V(scales != nullptr ? OK : FAIL);
        scales[0] = attr.oscale.scale;
        return;
    }

    assert(attr.oscale.policy == policy_t::PER_OC);

    scales = (float *)zmalloc(sizeof(float) * n, 64);
    SAFE_V(scales != nullptr ? OK : FAIL);

    const float K = 32;
    /* scale in [1/K .. K], with starting point at oscale.scale */
    float s[2] = {attr.oscale.scale, attr.oscale.scale / 2};
    for (int64_t i = 0; i < n; ++i) {
        int64_t si = i % 2; // 0 -> left, 1 -> right
        scales[i] = s[si];
        if (si == 0) {
            s[si] /= 2.;
            if (s[si] < 1. / K) s[si] *= K * K; // turn around to become ~K
        } else {
            s[si] *= 2.;
            if (s[si] > K) s[si] /= K * K; // turn around to become ~K
        }
    }
}

int legacy_str2desc(desc_t *desc, const char *str) {

    desc_t d;

    /* canonical form:
     * mbXmXnXkXnS
     *
     * where:
     * - X is number,
     * - S - string,
     *
     * note: symbol `_` is ignored
     *
     * note: n describes both 1) n - dimension and 2) n - name.
     *       The name is assumed to start with not a number symbol.
     *
     * default values:
     *      mb = 0, S="wip"
     */
    d.is_legacy_desc = true;
    int64_t mb = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;

    const char *s = str;
    assert(s);

#define CASE_NN(prb, c) \
    do { \
        if (!strncmp(prb, s, strlen(prb))) { \
            ok = 1; \
            s += strlen(prb); \
            char *end_s; \
            (c) = strtol(s, &end_s, 10); \
            s += (end_s - s); \
            if ((c) < 0) return FAIL; \
            /* printf("@@@debug: %s: %d\n", prb, (c)); */ \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        // order is important: check for name before n-dim
        if (*s == 'n' && !isdigit(*(s + 1))) {
            d.name = s + 1;
            break;
        }
        CASE_N(mb);
        CASE_N(m);
        CASE_N(n);
        CASE_N(k);
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (mb < 0 || m < 0 || n < 0 || k < 0) return FAIL;
    if (m * n * k == 0) return FAIL;

    d.sdims.resize(2);
    if (mb) {
        d.sdims[0].push_back(mb);
        d.sdims[1].push_back(mb);
    }
    d.sdims[0].push_back(m);
    d.sdims[0].push_back(k);

    d.sdims[1].push_back(k);
    d.sdims[1].push_back(n);

    *desc = d;

    return OK;
}

int str2desc(desc_t *desc, const char *str) {
    const char *s = str;

    while (*s == '_')
        ++s;
    if (*s == 'm') return legacy_str2desc(desc, s);

    desc_t d;
    d.is_legacy_desc = false;
    /* canonical form:
        d0xd1xd2xd3...:d0xd1xd2xd3...:d0xd1xd2xd3...nS
        with dimensions of src, weights and dst matrices are delimited by ':'
        in that order. The number of dims must match for all matrices.
    
        note: the dims are not auto expanded (to prevent undesired behavior by
        accidental expansion with missing dim value)

        S - string for name

        if the dst dims are not provided, then they are computed by benchdnn as
        dst_dims[d] = max(src_dims[d], weights_dims[d]), except for m, n dims
        for which usual convention is followed.

        default value for S = "wip"
    */

    int dims_idx = 0;
    while (*s) {
        if (isdigit(*s)) {
            d.sdims.resize(dims_idx + 1);

            char *end_s;
            d.sdims.back().push_back(strtol(s, &end_s, 10));
            if (d.sdims.back().back() < 0) return FAIL;

            s += (end_s - s);
            if (*s == ':') {
                ++dims_idx;
                ++s;
            } else if (*s == 'x') {
                ++s;
            }
        } else if (*s == 'n') {
            d.name = s + 1;
            break;
        } else if (*s == '_') {
            ++s;
        } else {
            return FAIL;
        }
    }
    if (d.sdims.size() < 2) return FAIL;
    const auto ndims = d.sdims[0].size();
    for (const auto &dims : d.sdims) {
        if (dims.size() != ndims) return FAIL;
    }
    *desc = d;
    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {

    for (size_t i = 0; i < d.sdims.size(); ++i) {
        for (size_t j = 0; j < d.sdims[i].size(); ++j) {
            s << d.sdims[i][j];
            if (j + 1 < d.sdims[i].size()) s << "x";
        }
        if (i + 1 < d.sdims.size()) s << ":";
    }

    if (d.name) s << "_n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.cfg != def.cfg[0]) s << "--cfg=" << prb.cfg << " ";
    if (canonical || prb.stag != def.stag[0]) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.wtag != def.wtag[0]) s << "--wtag=" << prb.wtag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";

    // TODO: switch me on when run-time leading dimensions will be supported
    // if (canonical || prb.ld_src != defaults::ld)
    //     s << "--ld_src=" << prb.ld_src << " ";
    // if (canonical || prb.ld_wei != defaults::ld)
    //     s << "--ld_wei=" << prb.ld_wei << " ";
    // if (canonical || prb.ld_dst != defaults::ld)
    //     s << "--ld_dst=" << prb.ld_dst << " ";

    if (canonical || prb.src_runtime_dim_mask().any()
            || prb.weights_runtime_dim_mask().any())
        s << "--runtime_dims_masks=" << prb.src_runtime_dim_mask().to_ulong()
          << ":" << prb.weights_runtime_dim_mask().to_ulong() << " ";

    if (canonical || prb.bia_dt != def.bia_dt[0]) {
        s << "--bia_dt=" << prb.bia_dt << " ";

        if (canonical || prb.bia_mask != def.bia_mask[0])
            s << "--bia_mask=" << prb.bia_mask << " ";
    }

    s << prb.attr;
    s << static_cast<const desc_t &>(prb);

    return s;
}

} // namespace matmul
