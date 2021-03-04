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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

int compare(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    const int summands = compute_n_summands(prb);
    float trh = 1e-6 * summands;
    if (prb->dt == dnnl_f16) trh = 1e-3 * summands;
    if (prb->dt == dnnl_bf16) trh = 1e-2 * summands;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(prb->dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= trh;

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            int64_t mb = 0, ic = 0, d = 0, h = 0, w = 0;
            inv_data_off(prb, i, mb, ic, d, h, w);

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

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const int range = 16;
    const int f_min = prb->dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = kind == SRC ? 1091 * i + 1637 : 1279 * i + 1009;
        const float value = f_min + gen % range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, SRC, mem_dt, mem_fp);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, DST, mem_dt, mem_fp);
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &lpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_lrn_desc_t ld;
    dnnl_memory_desc_t data_d;

    dnnl_dims_t data_dims_0d = {prb->mb, prb->ic};
    dnnl_dims_t data_dims_1d = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t data_dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t data_dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dnnl_dim_t *data_dims = prb->ndims == 5
            ? data_dims_3d
            : prb->ndims == 4 ? data_dims_2d
                              : prb->ndims == 3 ? data_dims_1d : data_dims_0d;

    SAFE(init_md(&data_d, prb->ndims, data_dims, prb->dt, prb->tag), CRIT);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);
    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE(dnnl_lrn_forward_desc_init(&ld, prop, alg, &data_d, prb->ls,
                         prb->alpha, prb->beta, prb->k),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, prb->ndims,
                         data_dims, prb->dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_lrn_backward_desc_init(&ld, alg, &diff_data_d, &data_d,
                         prb->ls, prb->alpha, prb->beta, prb->k),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&lpd, &ld, dnnl_attr, engine, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (prb->dir & FLAG_FWD)) return OK;

    res->impl_name = query_impl_info(lpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(lpd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t l {};
    SAFE(init_prim(&l, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(l, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(l));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(const_fpd, DNNL_ARG_SRC);
    const auto &ws_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, fp, tag, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t dst_fp(data_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(data_md, test_engine);

    if (prb->dir & FLAG_INF) SAFE(ws_md.ndims == 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(ws_md, test_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, d_src_dt;

    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE(execute_and_wait(l, args), WARN);

    if (prb->dir & FLAG_FWD) {
        if (bench_mode & CORR) {
            compute_ref_fwd(prb, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare(prb, dst, dst_fp, res), WARN);
        }
    }

    if (prb->dir & FLAG_BWD) {
        dnnl_primitive_t bwd_p {};
        int status = init_prim(&bwd_p, init_pd, prb, res, FLAG_BWD, const_fpd);
        dnnl_primitive_destroy(l);
        if (status != OK) return status;
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        l = bwd_p;

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(l, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(l));
            return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
        }

        const auto &d_data_md = q(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t d_src_fp(d_data_md, fp, tag, test_engine);
        d_src_dt = dnn_mem_t(d_data_md, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(fill_dst(prb, d_dst_dt, d_dst_fp), WARN);

        args.clear();
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(l, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(prb, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, test_engine);
            SAFE(compare(prb, d_src, d_src_fp, res), WARN);
        }
    }
    measure_perf(res->timer, l, args);

    DNN_SAFE_V(dnnl_primitive_destroy(l));

    return OK;
}

} // namespace lrn
