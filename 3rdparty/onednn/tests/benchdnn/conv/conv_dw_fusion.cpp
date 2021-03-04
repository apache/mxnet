/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "conv/conv_dw_fusion.hpp"

namespace conv_dw_fusion {

// This function is copy-pasted from dnn_types.cpp. We want it here for now to
// preserve current benchdnn attr semantics - what's set by user is remained
// untouched - it saves the input state. When fusion is done, it's enough to
// have only output data type, rest can be deduced from first conv.
// TODO: consider merge with common code by changing benchdnn attr semantics.
dnnl_primitive_attr_t create_dnnl_fusion_attr(
        const prb_t *prb, int64_t scale_cnt) {
    const attr_t &attr = prb->attr;

    dnnl_primitive_attr_t dnnl_attr = nullptr;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    if (!attr.oscale.is_def()) {
        int64_t count = attr.oscale.policy == policy_t::COMMON ? 1 : scale_cnt;
        int scale_mask = attr.oscale.policy == policy_t::PER_OC ? 1 << 1 : 0;

        float *scales = prb->scales;

        const bool runtime = attr.oscale.runtime;
        SAFE_V(scales == nullptr && runtime ? FAIL : OK);

        float *gen_scs = nullptr;
        if (scales == nullptr) {
            gen_scs = (float *)zmalloc(count * sizeof(float), 64);
            SAFE_V(gen_scs != nullptr ? OK : FAIL);
            for (int64_t i = 0; i < count; ++i)
                gen_scs[i] = attr.oscale.scale;
            scales = gen_scs;
        }

        DNN_SAFE_V(dnnl_primitive_attr_set_output_scales(dnnl_attr,
                runtime ? 1 : count, scale_mask,
                runtime ? &DNNL_RUNTIME_F32_VAL : scales));
        if (gen_scs) zfree(gen_scs);
    }

    if (!attr.post_ops.is_def()) {
        const auto &po = attr.post_ops;
        dnnl_post_ops_t ops;
        DNN_SAFE_V(dnnl_post_ops_create(&ops));
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_sum(ops, e.sum.scale));
            } else if (e.is_convolution_kind()) {
                const auto &os = e.convolution.oscale;
                int count = 0, mask = 0;
                const float *scales = &os.scale;
                float *gen_scs = nullptr;
                if (!os.is_def()) {
                    count = os.policy == policy_t::COMMON ? 1 : scale_cnt;
                    mask = os.policy == policy_t::PER_OC ? 1 << 1 : 0;
                    if (mask > 0) {
                        gen_scs = conv::generate_oscales(os, count);
                        SAFE_V(gen_scs != nullptr ? OK : FAIL);
                        scales = gen_scs;
                    }
                }
                const auto wei_dt = prb->cfg[WEI].dt;
                const auto bia_dt
                        = prb->dir == FWD_B ? dnnl_f32 : dnnl_data_type_undef;

                if (e.convolution.stride == 1) {
                    DNN_SAFE_V(dnnl_post_ops_append_dw_k3s1p1(ops, wei_dt,
                            bia_dt, e.convolution.dst_dt, count, mask, scales));
                } else {
                    DNN_SAFE_V(dnnl_post_ops_append_dw_k3s2p1(ops, wei_dt,
                            bia_dt, e.convolution.dst_dt, count, mask, scales));
                }
                if (gen_scs) zfree(gen_scs);
            } else if (e.is_eltwise_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_eltwise(ops, e.eltwise.scale,
                        e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));

        const_dnnl_post_ops_t c_ops;
        DNN_SAFE_V(dnnl_primitive_attr_get_post_ops(dnnl_attr, &c_ops));
        SAFE_V(dnnl_post_ops_len(c_ops) == po.len() ? OK : FAIL);

        DNN_SAFE_V(dnnl_post_ops_destroy(ops));
    }

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, attr.scratchpad_mode));

    return dnnl_attr;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_convolution_desc_t cd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_1d_dims = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_2d_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_3d_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dnnl_dims_t wei_1d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kw};
    dnnl_dims_t wei_2d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kh, prb->kw};
    dnnl_dims_t wei_3d_dims = {prb->g, prb->oc / prb->g, prb->ic / prb->g,
            prb->kd, prb->kh, prb->kw};

    dnnl_dims_t bia_dims = {prb->oc};

    dnnl_dims_t dst_1d_dims = {prb->mb, prb->oc, prb->ow};
    dnnl_dims_t dst_2d_dims = {prb->mb, prb->oc, prb->oh, prb->ow};
    dnnl_dims_t dst_3d_dims = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};

    dnnl_data_type_t src_dt = prb->cfg[SRC].dt;
    dnnl_data_type_t wei_dt = prb->cfg[WEI].dt;
    dnnl_data_type_t bia_dt = prb->cfg[BIA].dt;
    dnnl_data_type_t dst_dt = prb->cfg[DST].dt;
    dnnl_data_type_t acc_dt = prb->cfg[ACC].dt;
    std::string src_tag = prb->stag;
    std::string wei_tag = prb->wtag;
    std::string bia_tag = tag::any;
    std::string dst_tag = prb->dtag;

    SAFE(init_md(&src_d, prb->ndims,
                 prb->ndims == 5 ? src_3d_dims
                                 : prb->ndims == 3 ? src_1d_dims : src_2d_dims,
                 src_dt, src_tag),
            CRIT);

    SAFE(init_md(&wei_d, prb->ndims + prb->has_groups,
                 prb->ndims == 5
                         ? &wei_3d_dims[!prb->has_groups]
                         : prb->ndims == 3 ? &wei_1d_dims[!prb->has_groups]
                                           : &wei_2d_dims[!prb->has_groups],
                 wei_dt, wei_tag),
            CRIT);

    SAFE(init_md(&bia_d, 1, bia_dims, bia_dt, bia_tag), CRIT);

    SAFE(init_md(&dst_d, prb->ndims,
                 prb->ndims == 5 ? dst_3d_dims
                                 : prb->ndims == 3 ? dst_1d_dims : dst_2d_dims,
                 dst_dt, dst_tag),
            WARN);

    dnnl_dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dnnl_dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
    dnnl_dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
    dnnl_dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - prb->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - prb->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - prb->ndims);

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (prb->alg == alg_t::WINO) alg = dnnl_convolution_winograd;
    if (prb->alg == alg_t::AUTO) alg = dnnl_convolution_auto;

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_convolution_forward_desc_init(&cd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_dilated_convolution_backward_data_desc_init(&cd, alg,
                             &src_d, &wei_d, &dst_d, strides, dilates, padding,
                             padding_r),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_dilated_convolution_backward_weights_desc_init(&cd,
                             alg, &src_d, &wei_d,
                             prb->dir == BWD_W ? nullptr : &bia_d, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == acc_dt ? dnnl_success : dnnl_unimplemented,
            CRIT);

    auto dnnl_attr = create_dnnl_fusion_attr(prb, prb->oc);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&cpd, &cd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) {
        if (res) res->state = UNIMPLEMENTED;
        return OK;
    }
    SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if (prb->attr.post_ops.convolution_index() == -1) return OK;

    res->impl_name = query_impl_info(cpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(cpd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

std::unique_ptr<prb_t> get_first_conv_prb(const prb_t *prb) {
    const auto &po = prb->attr.post_ops;
    int fusion_index = po.convolution_index();

    attr_t attr;
    attr.oscale.scale = prb->attr.oscale.scale;
    attr.oscale.policy = prb->attr.oscale.policy;
    for (int i = 0; i < fusion_index; ++i) {
        attr.post_ops.entry.push_back(prb->attr.post_ops.entry[i]);
    }

    return std::unique_ptr<prb_t>(new prb_t((desc_t)*prb, prb->dir, prb->cfg,
            prb->stag, prb->wtag, tag::any, prb->alg, attr, prb->mb));
}

std::unique_ptr<prb_t> get_fused_conv_prb(const prb_t *prb) {
    const auto &po = prb->attr.post_ops;
    int fusion_index = po.convolution_index();
    if (fusion_index == -1) return nullptr;
    const auto &fused_conv_po = po.entry[fusion_index].convolution;

    attr_t fusion_attr;
    fusion_attr.oscale.scale = fused_conv_po.oscale.scale;
    fusion_attr.oscale.policy = fused_conv_po.oscale.policy;
    for (int i = fusion_index + 1; i < po.len(); ++i) {
        fusion_attr.post_ops.entry.push_back(prb->attr.post_ops.entry[i]);
    }

    const auto f32 = dnnl_f32;
    std::stringstream dw_cfg_ss;
    if (prb->cfg[DST].dt == f32 && prb->cfg[WEI].dt == f32
            && fused_conv_po.dst_dt == f32)
        dw_cfg_ss << prb->cfg[DST].dt; // f32 is a single name
    else // else have all three dt in cfg name
        dw_cfg_ss << prb->cfg[DST].dt << prb->cfg[WEI].dt
                  << fused_conv_po.dst_dt;
    auto p_dw_cfg = conv::str2cfg(dw_cfg_ss.str().c_str());

    auto stride = fused_conv_po.stride;
    bool is_3d = prb->ndims >= 5;
    bool is_2d = prb->ndims >= 4;

    desc_t cd {0};
    cd.g = prb->oc;
    cd.mb = prb->mb;
    cd.ic = prb->oc;
    cd.id = is_3d ? prb->od : 1;
    cd.ih = is_2d ? prb->oh : 1;
    cd.iw = prb->ow;
    cd.oc = prb->oc;
    cd.od = is_3d ? div_up(cd.id, stride) : 1;
    cd.oh = is_2d ? div_up(cd.ih, stride) : 1;
    cd.ow = div_up(cd.iw, stride);
    cd.kd = is_3d ? 3 : 1;
    cd.kh = is_2d ? 3 : 1;
    cd.kw = 3;
    cd.sd = is_3d ? stride : 1;
    cd.sh = is_2d ? stride : 1;
    cd.sw = stride;
    cd.pd = is_3d;
    cd.ph = is_2d;
    cd.pw = 1;
    cd.has_groups = true;
    cd.ndims = prb->ndims;
    cd.init_pad_r(false); // is_deconv = false for conv descriptor

    return std::unique_ptr<prb_t>(new prb_t(cd, prb->dir, p_dw_cfg, tag::any,
            tag::any, prb->dtag, alg_t::DIRECT, fusion_attr, prb->mb));
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    // Original problem with fusion attributes
    dnnl_primitive_t c {};
    SAFE(init_prim(&c, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_pd), CRIT);

    // Check memory requirements only for original problem though it's broken
    // due to quering not by arg md.
    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(c));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto adjust_alg = [](const_dnnl_primitive_desc_t pd, alg_t &alg) {
        if (alg == alg_t::AUTO) {
            dnnl_convolution_desc_t *temp_conv_desc = {nullptr};
            DNN_SAFE_V(dnnl_primitive_desc_query(
                    pd, dnnl_query_convolution_d, 0, &temp_conv_desc));
            alg = conv::alg_kind2alg(temp_conv_desc->alg_kind);
        }
        return OK;
    };

    alg_t alg = prb->alg;
    adjust_alg(const_pd, alg);
    auto cfg = auto_cfg(alg, prb->cfg);
    prb_t p_new((desc_t)*prb, prb->dir, cfg, prb->stag, prb->wtag, prb->dtag,
            alg, prb->attr, prb->mb);
    prb = &p_new;

    const auto &src_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &wei_md = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_WEIGHTS)
                                             : q(DNNL_ARG_WEIGHTS);
    const auto &bia_md
            = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_BIAS) : q(DNNL_ARG_BIAS);
    const auto &dst_md
            = prb->dir & FLAG_BWD ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &fused_wei_md = prb->dir & FLAG_WEI
            ? q(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DIFF_WEIGHTS)
            : q(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    const auto &fused_bia_md = prb->dir & FLAG_WEI
            ? q(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DIFF_BIAS)
            : q(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t fused_wei_dt(fused_wei_md, test_engine);
    dnn_mem_t fused_bia_dt(fused_bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;
    const auto fused_wei_tag = tag::abx;
    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, test_engine);
    dnn_mem_t fused_wei_fp(fused_wei_md, fp, fused_wei_tag, test_engine);
    dnn_mem_t fused_bia_fp(fused_bia_md, fp, tag::x, test_engine);

    // Current filling doesn't work for fused_wei due to relying on prb values,
    // which are different for fused conv. This can be fixed later by relying
    // on md values, rather than prb desc ones.
    // Filling for this problem is done below.
    // TODO: fix this if irritates.

    // Fill first convolution
    std::unique_ptr<prb_t> p0 = get_first_conv_prb(prb);

    dnnl_primitive_t c0 {};
    SAFE(init_prim(&c0, init_pd, p0.get(), res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd0;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c0, &const_pd0), CRIT);

    const auto q0 = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd0, dnnl_query_exec_arg_md, index);
    };

    alg = p0->alg;
    adjust_alg(const_pd0, alg);
    cfg = auto_cfg(alg, p0->cfg);
    p0.reset(new prb_t((desc_t)*p0, p0->dir, cfg, p0->stag, p0->wtag, p0->dtag,
            alg, p0->attr, p0->mb));

    const auto &src_md0
            = p0->dir == BWD_D ? q0(DNNL_ARG_DIFF_SRC) : q0(DNNL_ARG_SRC);
    const auto &wei_md0 = p0->dir & FLAG_WEI ? q0(DNNL_ARG_DIFF_WEIGHTS)
                                             : q0(DNNL_ARG_WEIGHTS);
    const auto &bia_md0
            = p0->dir & FLAG_WEI ? q0(DNNL_ARG_DIFF_BIAS) : q0(DNNL_ARG_BIAS);
    const auto &dst_md0
            = p0->dir & FLAG_BWD ? q0(DNNL_ARG_DIFF_DST) : q0(DNNL_ARG_DST);
    const auto &scratchpad_md0 = q0(DNNL_ARG_SCRATCHPAD);

    const auto src_tag0 = tag::abx;
    const auto wei_tag0 = tag::abx;

    dnn_mem_t src_dt0(src_md0, test_engine);
    dnn_mem_t wei_dt0(wei_md0, test_engine);
    dnn_mem_t bia_dt0(bia_md0, test_engine);
    dnn_mem_t dst_dt0(dst_md0, test_engine);
    dnn_mem_t scratchpad_dt0(scratchpad_md0, test_engine);

    dnn_mem_t src_fp0(src_md0, fp, src_tag0, test_engine);
    dnn_mem_t wei_fp0(wei_md0, fp, wei_tag0, test_engine);
    dnn_mem_t bia_fp0(bia_md0, fp, tag::x, test_engine);
    dnn_mem_t dst_fp0(dst_md0, fp, src_tag0, test_engine);

    SAFE(conv::fill_src(p0.get(), src_dt0, src_fp0, res), WARN);
    SAFE(conv::fill_wei(p0.get(), wei_dt0, wei_fp0, res), WARN);
    SAFE(conv::fill_bia(p0.get(), bia_dt0, bia_fp0, res), WARN);
    SAFE(conv::fill_dst(p0.get(), dst_dt0, dst_fp0, res), WARN);

    // Fill next convolution
    std::unique_ptr<prb_t> p1 = get_fused_conv_prb(prb);
    if (!p1) SAFE(FAIL, CRIT);

    dnnl_primitive_t c1 {};
    SAFE(init_prim(&c1, init_pd, p1.get(), res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd1;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c1, &const_pd1), CRIT);

    const auto q1 = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd1, dnnl_query_exec_arg_md, index);
    };

    alg = p1->alg;
    adjust_alg(const_pd1, alg);
    cfg = auto_cfg(alg, p1->cfg);
    p1.reset(new prb_t((desc_t)*p1, p1->dir, cfg, p1->stag, p1->wtag, p1->dtag,
            alg, p1->attr, p1->mb));

    const auto &src_md1
            = prb->dir == BWD_D ? q1(DNNL_ARG_DIFF_SRC) : q1(DNNL_ARG_SRC);
    const auto &wei_md1 = prb->dir & FLAG_WEI ? q1(DNNL_ARG_DIFF_WEIGHTS)
                                              : q1(DNNL_ARG_WEIGHTS);

    const auto &bia_md1
            = prb->dir & FLAG_WEI ? q1(DNNL_ARG_DIFF_BIAS) : q1(DNNL_ARG_BIAS);
    const auto &dst_md1
            = prb->dir & FLAG_BWD ? q1(DNNL_ARG_DIFF_DST) : q1(DNNL_ARG_DST);
    const auto &scratchpad_md1 = q(DNNL_ARG_SCRATCHPAD);

    const auto src_tag1 = tag::abx;
    const auto wei_tag1 = tag::abx;
    dnn_mem_t src_dt1(src_md1, test_engine);
    dnn_mem_t wei_dt1(wei_md1, test_engine);
    dnn_mem_t bia_dt1(bia_md1, test_engine);
    dnn_mem_t dst_dt1(dst_md1, test_engine);
    dnn_mem_t scratchpad_dt1(scratchpad_md1, test_engine);

    dnn_mem_t wei_fp1(wei_md1, fp, wei_tag1, test_engine);
    dnn_mem_t bia_fp1(bia_md1, fp, tag::x, test_engine);
    dnn_mem_t dst_fp1(dst_md1, fp, src_tag1, test_engine);

    SAFE(conv::fill_wei(p1.get(), wei_dt1, wei_fp1, res), WARN);
    SAFE(conv::fill_bia(p1.get(), bia_dt1, bia_fp1, res), WARN);
    SAFE(conv::fill_dst(p1.get(), dst_dt1, dst_fp1, res), WARN);

    // TODO: fix this if irritates.
    // SAFE(conv::fill_src(prb, src_dt, src_fp, res), WARN);
    // SAFE(conv::fill_wei(prb, wei_dt, wei_fp, res), WARN);
    // SAFE(conv::fill_bia(prb, bia_dt, bia_fp, res), WARN);
    // SAFE(conv::fill_dst(prb, dst_dt, dst_fp, res), WARN);
    // SAFE(conv::fill_wei(prb, fused_wei_dt, fused_wei_fp, res), WARN);
    // SAFE(conv::fill_bia(prb, fused_bia_dt, fused_bia_fp, res), WARN);
    // Work around for the issue above
    SAFE(src_dt.reorder(src_fp0), WARN);
    SAFE(wei_dt.reorder(wei_fp0), WARN);
    if (bia_md.data_type != dnnl_data_type_undef)
        SAFE(bia_dt.reorder(bia_fp0), WARN);
    SAFE(dst_dt.reorder(dst_fp1), WARN);
    SAFE(fused_wei_dt.reorder(wei_fp1), WARN);
    if (fused_bia_md.data_type != dnnl_data_type_undef)
        SAFE(fused_bia_dt.reorder(bia_fp1), WARN);

    args_t args, args0, args1;

    if (prb->dir & FLAG_FWD) {
        args0.set(DNNL_ARG_SRC, src_dt0);
        args0.set(DNNL_ARG_WEIGHTS, wei_dt0);
        args0.set(DNNL_ARG_BIAS, bia_dt0);
        args0.set(DNNL_ARG_DST, dst_dt0);
        args0.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt0);

        SAFE(execute_and_wait(c0, args0), WARN);
        SAFE(src_dt1.reorder(dst_dt0), WARN);

        args1.set(DNNL_ARG_SRC, src_dt1);
        args1.set(DNNL_ARG_WEIGHTS, wei_dt1);
        args1.set(DNNL_ARG_BIAS, bia_dt1);
        args1.set(DNNL_ARG_DST, dst_dt1);
        args1.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt1);

        SAFE(execute_and_wait(c1, args1), WARN);

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        if (bia_md.data_type != dnnl_data_type_undef)
            args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, fused_wei_dt);
        if (fused_bia_md.data_type != dnnl_data_type_undef)
            args.set(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS, fused_bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            dnn_mem_t dst_fused(dst_dt, fp, src_tag, test_engine);
            dnn_mem_t dst_unfused(dst_dt1, fp, src_tag, test_engine);
            // Used p1 to avoid writing separate compare function. Compare uses
            // prb->cfg which can be u8s8u8 while after fusion it may be u8s8s8,
            // thus, compare() will saturate values which is not correct.
            SAFE(conv::compare_dst(p1.get(), dst_fused, dst_unfused, res, true),
                    WARN);
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(res->timer, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));
    DNN_SAFE_V(dnnl_primitive_destroy(c0));
    DNN_SAFE_V(dnnl_primitive_destroy(c1));

    return OK;
}

} // namespace conv_dw_fusion
