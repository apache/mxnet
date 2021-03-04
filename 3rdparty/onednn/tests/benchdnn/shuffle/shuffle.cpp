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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    auto get_range = [](const dnnl_data_type_t dt) {
        if (dt == dnnl_s8 || dt == dnnl_u8)
            return 256;
        else if (dt == dnnl_bf16 || dt == dnnl_f16)
            return 128;
        return 1024;
    };

    const auto nelems = mem_fp.nelems();
    const int range = get_range(prb->dt);
    const int f_min = prb->dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) + 101) % range;
        const float value = (prb->dt == dnnl_bf16 || prb->dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, round_to_nearest_representable(prb->dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

static int compare(const prb_t *prb, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *res) {
    const float trh = 0;
    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp = fp_mem.get_elem(i);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(prb->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0, "[%4ld][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &spd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_memory_desc_t data_d;
    dnnl_shuffle_desc_t sd;

    SAFE(init_md(&data_d, prb->ndims, prb->dims.data(), prb->dt, prb->tag),
            CRIT);

    auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                         : dnnl_forward_training;
    DNN_SAFE(dnnl_shuffle_forward_desc_init(
                     &sd, prop_kind, &data_d, prb->axis, prb->group),
            WARN);

    dnnl_primitive_desc_t _hint = nullptr;
    auto cleanup_pd = [&]() { dnnl_primitive_desc_destroy(_hint); };
    if (prb->dir & FLAG_BWD) {
        dnnl_status_t init_fwd_status = dnnl_primitive_desc_create(
                &_hint, &sd, nullptr, engine, nullptr);
        if (init_fwd_status == dnnl_unimplemented)
            return res->state = UNIMPLEMENTED, OK;
        SAFE(init_fwd_status, WARN);

        DNN_SAFE_CLEAN(dnnl_memory_desc_init_by_tag(&data_d, prb->ndims,
                               prb->dims.data(), prb->dt, dnnl_format_tag_any),
                WARN, cleanup_pd);

        DNN_SAFE_CLEAN(dnnl_shuffle_backward_desc_init(
                               &sd, &data_d, prb->axis, prb->group),
                WARN, cleanup_pd);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&spd, &sd, dnnl_attr, engine, _hint);

    dnnl_primitive_desc_destroy(_hint);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(spd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t s {};
    SAFE(init_prim(&s, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(s, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(s));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md
            = prb->dir & FLAG_FWD ? q(DNNL_ARG_SRC) : q(DNNL_ARG_DIFF_SRC);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, fp, tag, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t dst_fp(data_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(data_md, test_engine);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    const int i_arg = prb->dir == FWD_D ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const int o_arg = prb->dir == FWD_D ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;

    args_t args;

    args.set(i_arg, src_dt);
    args.set(o_arg, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE(execute_and_wait(s, args), WARN);

    if (bench_mode & CORR) {
        compute_shuffle(prb, src_fp, dst_fp);
        dnn_mem_t data(dst_dt, fp, tag, test_engine);
        SAFE(compare(prb, dst_fp, data, res), WARN);
    }

    measure_perf(res->timer, s, args);

    DNN_SAFE_V(dnnl_primitive_destroy(s));

    return OK;
}

} // namespace shuffle
