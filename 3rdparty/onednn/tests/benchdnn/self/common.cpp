/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <stdlib.h>
#include <string.h>

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "self/self.hpp"

namespace self {

static int check_simple_enums() {
    /* attr::post_ops::kind */
    using p = attr_t::post_ops_t;
    CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::SUM), "sum");
    CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::RELU), "relu");

    CHECK_EQ(p::str2kind("sum"), p::kind_t::SUM);
    CHECK_EQ(p::str2kind("SuM"), p::kind_t::SUM);

    CHECK_EQ(p::str2kind("relu"), p::kind_t::RELU);
    CHECK_EQ(p::str2kind("ReLU"), p::kind_t::RELU);

    return OK;
}

static int check_attr2str() {
    attr_t attr;
    CHECK_EQ(attr.is_def(), true);

    CHECK_PRINT_EQ(attr, "");

    attr.oscale.policy = policy_t::COMMON;
    attr.oscale.scale = 2.4;
    CHECK_PRINT_EQ(attr, "--attr-oscale=common:2.4 ");

    attr.oscale.policy = policy_t::PER_OC;
    attr.oscale.scale = 3.2;
    attr.oscale.runtime = true;
    CHECK_PRINT_EQ(attr, "--attr-oscale=per_oc:3.2* ");

    attr.oscale.policy = policy_t::PER_DIM_01;
    attr.oscale.scale = 3.2;
    attr.oscale.runtime = false;
    CHECK_PRINT_EQ(attr, "--attr-oscale=per_dim_01:3.2 ");

    attr.zero_points.set(DNNL_ARG_SRC, policy_t::COMMON, 1, false);
    CHECK_PRINT_EQ(attr,
            "--attr-oscale=per_dim_01:3.2 --attr-zero-points=src:common:1 ");

    attr.zero_points.set(DNNL_ARG_WEIGHTS, {policy_t::PER_DIM_1, 2, true});
    CHECK_PRINT_EQ2(attr,
            "--attr-oscale=per_dim_01:3.2 "
            "--attr-zero-points=src:common:1_wei:per_dim_1:2* ",
            "--attr-oscale=per_dim_01:3.2 "
            "--attr-zero-points=wei:per_dim_1:2*_src:common:1 ");

    attr = attr_t();
    attr.scales.set(DNNL_ARG_SRC_0, attr_t::scale_t(policy_t::COMMON, 2.3));
    CHECK_PRINT_EQ(attr, "--attr-scales=src:common:2.3 ");

    attr.scales.set(DNNL_ARG_SRC_0, attr_t::scale_t(policy_t::COMMON, 2.3));
    attr.scales.set(DNNL_ARG_SRC_1, attr_t::scale_t(policy_t::COMMON, 3));
    CHECK_PRINT_EQ(attr, "--attr-scales=src:common:2.3_src1:common:3 ");

    return OK;
}

static int check_str2attr() {
    attr_t attr;

#define CHECK_ATTR(str, os_policy, os_scale, os_runtime) \
    do { \
        CHECK_EQ(str2attr(&attr, str), OK); \
        CHECK_EQ(attr.oscale.policy, policy_t::os_policy); \
        CHECK_EQ(attr.oscale.scale, os_scale); \
        CHECK_EQ(attr.oscale.runtime, os_runtime); \
    } while (0)
#define CHECK_ATTR_ZP(arg, zero_points_value, zero_points_runtime) \
    do { \
        const auto entry = attr.zero_points.get(arg); \
        CHECK_EQ(entry.value, zero_points_value); \
        CHECK_EQ(entry.runtime, zero_points_runtime); \
    } while (0)

    CHECK_ATTR("", COMMON, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("oscale=common:1.0", COMMON, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR(
            "oscale=common:1.0;zero_points=src:common:0_wei:common:0_dst:"
            "common:0",
            COMMON, 1., false);
    CHECK_EQ(attr.is_def(), true);

    CHECK_ATTR("oscale=common:2.0", COMMON, 2., false);
    CHECK_ATTR("oscale=common:2.0*", COMMON, 2., true);
    CHECK_ATTR("oscale=per_oc:.5*;", PER_OC, .5, true);

    CHECK_ATTR("oscale=common:2.0*;zero_points=src:common:0_dst:common:-2*",
            COMMON, 2., true);
    CHECK_ATTR_ZP(DNNL_ARG_SRC, 0, false);
    CHECK_ATTR_ZP(DNNL_ARG_WEIGHTS, 0, false);
    CHECK_ATTR_ZP(DNNL_ARG_DST, -2, true);

    CHECK_EQ(str2attr(&attr, "scales=src1:common:1.5;"), OK);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).policy, policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).scale, 1.5);

    CHECK_EQ(str2attr(&attr, "scales=src:common:2.5_src1:common:1.5;"), OK);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_0).policy, policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_0).scale, 2.5);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).policy, policy_t::COMMON);
    CHECK_EQ(attr.scales.get(DNNL_ARG_SRC_1).scale, 1.5);

#undef CHECK_ATTR
#undef CHECK_ATTR_ZP

    return OK;
}

using pk_t = attr_t::post_ops_t::kind_t;

void append_sum(attr_t::post_ops_t &po, float ascale = 1.f,
        dnnl_data_type_t adt = dnnl_data_type_undef) {
    attr_t::post_ops_t::entry_t e(pk_t::SUM);
    e.sum.scale = ascale;
    e.sum.dt = adt;
    po.entry.push_back(e);
}

void append_convolution(attr_t::post_ops_t &po, pk_t akind,
        dnnl_data_type_t adst_dt = dnnl_f32,
        policy_t apolicy = policy_t::COMMON, float ascale = 1.f) {
    attr_t::post_ops_t::entry_t e(akind);
    e.convolution.stride = e.kind == pk_t::DW_K3S1P1 ? 1 : 2;
    e.convolution.dst_dt = adst_dt;
    e.convolution.oscale = attr_t::scale_t(apolicy, ascale);
    po.entry.push_back(e);
}

void append_eltwise(attr_t::post_ops_t &po, pk_t akind, float aalpha = 0.f,
        float abeta = 0.f, float ascale = 1.f) {
    attr_t::post_ops_t::entry_t e(akind);
    e.eltwise.alg = attr_t::post_ops_t::kind2dnnl_kind(akind);
    e.eltwise.alpha = aalpha;
    e.eltwise.beta = abeta;
    e.eltwise.scale = ascale;
    po.entry.push_back(e);
}

static int check_post_ops2str() {
    attr_t::post_ops_t po;
    CHECK_EQ(po.is_def(), true);
    CHECK_PRINT_EQ(po, "''");

    append_sum(po);
    CHECK_EQ(po.len(), 1);
    CHECK_PRINT_EQ(po, "'sum'");

    append_eltwise(po, pk_t::RELU);
    CHECK_EQ(po.len(), 2);
    CHECK_PRINT_EQ(po, "'sum;relu'");

    append_sum(po, 2.f, dnnl_s8);
    CHECK_EQ(po.len(), 3);
    CHECK_PRINT_EQ(po, "'sum;relu;sum:2:s8'");

    append_eltwise(po, pk_t::LINEAR, 5.f, 10.f, 2.f);
    CHECK_EQ(po.len(), 4);
    CHECK_PRINT_EQ(po, "'sum;relu;sum:2:s8;linear:5:10:2'");

    append_convolution(po, pk_t::DW_K3S1P1);
    CHECK_EQ(po.len(), 5);
    CHECK_PRINT_EQ(po, "'sum;relu;sum:2:s8;linear:5:10:2;dw_k3s1p1'");

    append_convolution(po, pk_t::DW_K3S2P1, dnnl_s32, policy_t::PER_OC, 2.f);
    CHECK_EQ(po.len(), 6);
    CHECK_PRINT_EQ(po,
            "'sum;relu;sum:2:s8;linear:5:10:2;dw_k3s1p1;dw_k3s2p1:s32:per_oc:"
            "2'");

    return OK;
}

static int check_str2post_ops() {
    attr_t::post_ops_t ops;

    CHECK_EQ(ops.is_def(), true);

    auto quick = [&](int len) -> int {
        for (int i = 0; i < 2; ++i) {
            if (2 * i + 0 >= len) return OK;
            CHECK_EQ(ops.entry[2 * i + 0].kind, attr_t::post_ops_t::SUM);
            CHECK_EQ(ops.entry[2 * i + 0].sum.scale, 2. + i);
            if (2 * i + 1 >= len) return OK;
            CHECK_EQ(ops.entry[2 * i + 1].kind, attr_t::post_ops_t::RELU);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.scale, 1.);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.alpha, 0.);
            CHECK_EQ(ops.entry[2 * i + 1].eltwise.beta, 0.);
        }
        return OK;
    };

    ops.from_str("''");
    CHECK_EQ(ops.is_def(), true);

    ops.from_str("'sum:2;'");
    CHECK_EQ(quick(1), OK);

    ops.from_str("'sum:2;relu'");
    CHECK_EQ(quick(2), OK);

    ops.from_str("'sum:2;relu;sum:3'");
    CHECK_EQ(quick(3), OK);

    ops.from_str("'sum:2;relu;sum:3;relu;'");
    CHECK_EQ(quick(4), OK);

    return OK;
}

static int check_tags() {
    dnnl_memory_desc_t md_from_tag;
    dnnl_memory_desc_t md_from_str;

    dnnl_dims_t dims = {29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73};
    for (int tag_ = dnnl_format_tag_undef; tag_ != dnnl_format_tag_last;
            tag_++) {
        dnnl_format_tag_t format_tag = (dnnl_format_tag_t)tag_;
        const char *str_tag = fmt_tag2str(format_tag);
        int ndims = 1;
        for (char c = (char)('a' + DNNL_MAX_NDIMS - 1); c >= 'a'; c--) {
            if (strchr(str_tag, c)) {
                ndims = c - 'a' + 1;
                break;
            }
        }

        SAFE(init_md(&md_from_str, ndims, dims, dnnl_f32, str_tag), CRIT);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(
                         &md_from_tag, ndims, dims, dnnl_f32, format_tag),
                CRIT);
        int eq = dnnl_memory_desc_equal(&md_from_tag, &md_from_str);
        CHECK_EQ(eq, 1);
    }

    return OK;
}

void common() {
    RUN(check_simple_enums());
    RUN(check_attr2str());
    RUN(check_str2attr());
    RUN(check_post_ops2str());
    RUN(check_str2post_ops());
    RUN(check_tags());
}

} // namespace self
