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

#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "binary/binary.hpp"
#include "pool/pool.hpp"

namespace pool {

inline int compare_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(prb->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = false;
        if (std::isinf(fp))
            ok = std::isinf(dt) && std::signbit(fp) == std::signbit(dt);
        else if (fp < prb->cfg[kind].min)
            ok = dt == prb->cfg[kind].min;
        else
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= prb->cfg[kind].eps;

        res->errors += !ok;

        bool dump = (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            int64_t mb = 0, ic = 0, d = 0, h = 0, w = 0;
            switch (kind) {
                case SRC: inv_src_off_f(prb, i, mb, ic, d, h, w); break;
                case DST: inv_dst_off_f(prb, i, mb, ic, d, h, w); break;
            }
            BENCHDNN_PRINT(0,
                    "[%4ld][" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                    "] "
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, mb, ic, d, h, w, fp, fp0, dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int compare_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return compare_dat(prb, SRC, mem_dt, mem_fp, res);
}

int compare_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return compare_dat(prb, DST, mem_dt, mem_fp, res);
}

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const int64_t MB {prb->mb};
    const int64_t IC {prb->ic};
    const int64_t D {kind == SRC ? prb->id : prb->od};
    const int64_t H {kind == SRC ? prb->ih : prb->oh};
    const int64_t W {kind == SRC ? prb->iw : prb->ow};
    const int64_t ker_size {prb->kd * prb->kh * prb->kw};
    const auto &c = prb->cfg[kind];

    dnnl::impl::parallel_nd(MB, IC, D, H, W,
            [&](int64_t mb, int64_t ic, int64_t d, int64_t h, int64_t w) {
                const int64_t factor = prb->alg == MAX ? 1 : ker_size;
                // keep values for avg_exclude_pad positive to prevent cancellation err
                const int64_t f_min = prb->alg == MAX ? c.f_min / factor : 0;
                // divide on factor to keep value in the range
                const int64_t range = c.f_max / factor - f_min + 1;
                const int64_t gen
                        = 5 * d + 17 * h + 13 * w + 13 * mb + 19 * ic + 1637;
                const float value = (f_min + gen % range) * factor;

                const size_t off = kind == SRC
                        ? src_off_f(prb, mb, ic, d, h, w)
                        : dst_off_f(prb, mb, ic, d, h, w);
                ((float *)mem_fp)[off] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, SRC, mem_dt, mem_fp, res);
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, DST, mem_dt, mem_fp, res);
}

// fill ws with big numbers to reliably cause a correctness issue (and not
// anything else) in case of a bug in the library
int fill_ws(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    dnnl::impl::parallel_nd(mem_fp.nelems(),
            [&](int64_t i) { mem_fp.set_elem(i, (1 << 24) - 1); });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &ppd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_memory_desc_t src_d, dst_d;

    dnnl_dims_t src_1d_dims = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_2d_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_3d_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_3d_dims
            : prb->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t dst_1d_dims = {prb->mb, prb->ic, prb->ow};
    dnnl_dims_t dst_2d_dims = {prb->mb, prb->ic, prb->oh, prb->ow};
    dnnl_dims_t dst_3d_dims = {prb->mb, prb->ic, prb->od, prb->oh, prb->ow};
    dnnl_dim_t *dst_dims = prb->ndims == 5
            ? dst_3d_dims
            : prb->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    const auto src_tag = (dir & FLAG_FWD) ? prb->tag : tag::any;
    const auto dst_tag = tag::any;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->cfg[SRC].dt, src_tag),
            CRIT);

    SAFE(init_md(&dst_d, prb->ndims, dst_dims, prb->cfg[DST].dt, dst_tag),
            CRIT);

    dnnl_dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dnnl_dim_t kernel_nd[] = {prb->kd, prb->kh, prb->kw};
    dnnl_dim_t dilation_nd[] = {prb->dd, prb->dh, prb->dw};
    dnnl_dim_t padding_l_nd[] = {prb->pd, prb->ph, prb->pw};
    dnnl_dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - prb->ndims);
    dnnl_dim_t *kernel = kernel_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_l = padding_l_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - prb->ndims);
    dnnl_dim_t *dilation = dilation_nd + (5 - prb->ndims);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);
    dnnl_pooling_v2_desc_t pd;

    if (dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        DNN_SAFE(dnnl_pooling_v2_forward_desc_init(&pd, prop_kind, alg, &src_d,
                         &dst_d, strides, kernel, dilation, padding_l,
                         padding_r),
                WARN);
    } else {
        DNN_SAFE(dnnl_pooling_v2_backward_desc_init(&pd, alg, &src_d, &dst_d,
                         strides, kernel, dilation, padding_l, padding_r),
                WARN);
    }

    attr_args_t attr_args;
    attr_args.prepare_binary_post_op_mds(prb->attr, prb->ndims, dst_dims);
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status;

    init_status
            = dnnl_primitive_desc_create(&ppd, &pd, dnnl_attr, engine, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (prb->dir & FLAG_FWD)) return OK;

    res->impl_name = query_impl_info(ppd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(ppd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[DST].dt}, prb->dir, res);
    if (res->state == SKIPPED) return;

    if (prb->alg == AVG_NP) {
        bool ker_in_pad_d = prb->pd >= prb->kd || prb->pd_r >= prb->kd;
        bool ker_in_pad_h = prb->ph >= prb->kh || prb->ph_r >= prb->kh;
        bool ker_in_pad_w = prb->pw >= prb->kw || prb->pw_r >= prb->kw;
        bool ker_in_pad = ker_in_pad_d || ker_in_pad_h || ker_in_pad_w;

        if (ker_in_pad) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
            return;
        }
    }

    if (engine_tgt_kind == dnnl_cpu && prb->cfg[SRC].dt != prb->cfg[DST].dt) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t pp {};
    SAFE(init_prim(&pp, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(pp, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(pp));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_md = q(const_fpd, DNNL_ARG_SRC);
    const auto &dst_md = q(const_fpd, DNNL_ARG_DST);
    const auto &ws_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    SAFE(!check_md_consistency_with_tag(dst_md, prb->tag), WARN);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(src_md, fp, tag, test_engine);
    dnn_mem_t src_dt(src_md, test_engine);

    dnn_mem_t dst_fp(dst_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);

    if (prb->dir & FLAG_INF) SAFE(ws_md.ndims == 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(ws_md, test_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_fpd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t d_src_dt, d_dst_dt;

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(pp, args), WARN);

    // want this pass on backward to get ws_fp filled properly
    if (bench_mode & CORR) {
        compute_ref_fwd(prb, src_fp, binary_po_fp, dst_fp, ws_fp);
        if (prb->dir & FLAG_FWD) {
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare_dst(prb, dst, dst_fp, res), WARN);
        }
    }

    if (prb->dir & FLAG_BWD) {
        dnnl_primitive_t bwd_p {};
        int status = init_prim(&bwd_p, init_pd, prb, res, FLAG_BWD, const_fpd);
        dnnl_primitive_destroy(pp);
        if (status != OK) return status;
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        pp = bwd_p;

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(pp, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(pp));
            return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
        }

        const auto &d_dst_md = q(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_src_md = q(const_bpd, DNNL_ARG_DIFF_SRC);
        const auto &d_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp = dnn_mem_t(d_dst_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_dst_md, prb->cfg[DST].dt, test_engine);

        dnn_mem_t d_src_fp = dnn_mem_t(d_src_md, fp, tag, test_engine);
        d_src_dt = dnn_mem_t(d_src_md, prb->cfg[SRC].dt, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(fill_dst(prb, d_dst_dt, d_dst_fp, res), WARN);

        args.clear();
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(pp, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(prb, d_src_fp, d_dst_fp, ws_fp);
            dnn_mem_t diff_src(d_src_dt, fp, tag, test_engine);
            SAFE(compare_src(prb, diff_src, d_src_fp, res), WARN);
        }
    }

    measure_perf(res->timer, pp, args);

    DNN_SAFE_V(dnnl_primitive_destroy(pp));

    return OK;
}

} // namespace pool
