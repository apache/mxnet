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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "norm.hpp"

#include "binary/binary.hpp"
#include "conv/deconv.hpp"
using namespace conv;

namespace deconv {

inline static void swap(int64_t &a, int64_t &b) {
    int64_t temp = a;
    a = b;
    b = temp;
}

inline int transpose_data_wei(
        const prb_t *prb, dnn_mem_t &wei, dnn_mem_t &wei_tr) {
    dnnl::impl::parallel_nd(prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                size_t idx = (((((size_t)g * prb->ic / prb->g + ic) * prb->oc
                                               / prb->g
                                       + oc) * prb->kd
                                      + kd) * prb->kh
                                     + kh)
                                * prb->kw
                        + kw;
                ((float *)wei_tr)[idx]
                        = ((float *)wei)[wei_off_f(prb, g, oc, ic, kd, kh, kw)];
            });

    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &dpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_deconvolution_desc_t cd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_1d_dims = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_2d_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_3d_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_3d_dims
            : prb->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t wei_1d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kw};
    dnnl_dims_t wei_2d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kh, prb->kw};
    dnnl_dims_t wei_3d_dims = {prb->g, prb->oc / prb->g, prb->ic / prb->g,
            prb->kd, prb->kh, prb->kw};
    dnnl_dim_t *wei_dims = prb->ndims == 5
            ? &wei_3d_dims[!prb->has_groups]
            : prb->ndims == 4 ? &wei_2d_dims[!prb->has_groups]
                              : &wei_1d_dims[!prb->has_groups];

    dnnl_dims_t bia_dims = {prb->oc};

    dnnl_dims_t dst_1d_dims = {prb->mb, prb->oc, prb->ow};
    dnnl_dims_t dst_2d_dims = {prb->mb, prb->oc, prb->oh, prb->ow};
    dnnl_dims_t dst_3d_dims = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};
    dnnl_dim_t *dst_dims = prb->ndims == 5
            ? dst_3d_dims
            : prb->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->cfg[SRC].dt, prb->stag),
            CRIT);
    SAFE(init_md(&wei_d, prb->ndims + prb->has_groups, wei_dims,
                 prb->cfg[WEI].dt, prb->wtag),
            CRIT);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&bia_d, 1, bia_dims, prb->cfg[BIA].dt,
                     dnnl_format_tag_any),
            WARN);
    SAFE(init_md(&dst_d, prb->ndims, dst_dims, prb->cfg[DST].dt, prb->dtag),
            CRIT);

    dnnl_dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dnnl_dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
    dnnl_dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
    dnnl_dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - prb->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - prb->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - prb->ndims);

    dnnl_alg_kind_t alg = dnnl_deconvolution_direct;
    if (prb->alg == WINO) alg = dnnl_deconvolution_winograd;

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_deconvolution_forward_desc_init(&cd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_dilated_deconvolution_backward_data_desc_init(&cd,
                             alg, &src_d, &wei_d, &dst_d, strides, dilates,
                             padding, padding_r),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_dilated_deconvolution_backward_weights_desc_init(&cd,
                             alg, &src_d, &wei_d,
                             prb->dir == BWD_W ? nullptr : &bia_d, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == prb->cfg[ACC].dt ? dnnl_success
                                                    : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_binary_post_op_mds(prb->attr, prb->ndims, dst_dims);
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&dpd, &cd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) {
        return res->state = UNIMPLEMENTED, OK;
    }
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(dpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(dpd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    if (res->state == SKIPPED) return;

    // GPU:
    //     * BWD: doesn't support any attributes
    //     * FWD: support only post ops
    if (engine_tgt_kind == dnnl_gpu
            && (((prb->dir & FLAG_BWD) != 0 && !prb->attr.is_def())
                    || ((prb->dir & FLAG_FWD) != 0
                            && (!prb->attr.oscale.is_def()
                                    || !prb->attr.scales.is_def()
                                    || !prb->attr.zero_points.is_def())))) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    prb_t p_tr((desc_t)*prb, prb->dir, prb->cfg, prb->stag, prb->wtag,
            prb->dtag, prb->alg, prb->attr, prb->mb, true);
    swap(p_tr.ic, p_tr.oc);
    swap(p_tr.ih, p_tr.oh);
    swap(p_tr.id, p_tr.od);
    swap(p_tr.iw, p_tr.ow);

    dnnl_primitive_t d {};
    SAFE(init_prim(&d, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(d, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(d));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &wei_md = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_WEIGHTS)
                                             : q(DNNL_ARG_WEIGHTS);
    const auto &bia_md
            = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_BIAS) : q(DNNL_ARG_BIAS);
    const auto &dst_md
            = prb->dir & FLAG_BWD ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    auto wei_tr_md = wei_md;

    const bool with_groups = true;
    swap(wei_tr_md.dims[with_groups + 0], wei_tr_md.dims[with_groups + 1]);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, test_engine);
    dnn_mem_t wei_tr_fp(wei_tr_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);

    /* fill memory + reorders <-> */
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(transpose_data_wei(prb, wei_fp, wei_tr_fp), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);

    args_t args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(&p_tr, nullptr, dst_fp, wei_tr_fp, bia_fp,
                    binary_po_fp, src_fp);
            dnn_mem_t dst(dst_dt, fp, src_tag, test_engine);
            SAFE(compare_dst(prb, dst, dst_fp, res, true), WARN);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            dnn_mem_t zero_fp;
            compute_ref_fwd(&p_tr, nullptr, dst_fp, wei_tr_fp, zero_fp,
                    std::vector<dnn_mem_t>(), src_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_src(prb, src, src_fp, res, true), WARN);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_weights(&p_tr, dst_fp, wei_tr_fp, src_fp);
            transpose_data_wei(&p_tr, wei_tr_fp, wei_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            SAFE(compare_wei(&p_tr, wei, wei_fp, res, true), WARN);
            if (prb->dir & FLAG_BIA) {
                compute_ref_bwd_bias(prb, bia_fp, dst_fp);
                dnn_mem_t bia(bia_dt, fp, tag::x, test_engine);
                SAFE(compare_bia(prb, bia, bia_fp, res, true), WARN);
            }
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(res->timer, d, args);

    DNN_SAFE_V(dnnl_primitive_destroy(d));

    return OK;
}

} // namespace deconv
