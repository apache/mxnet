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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &spd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_softmax_desc_t sd;
    dnnl_memory_desc_t data_d;

    SAFE(init_md(&data_d, prb->ndims, prb->dims.data(), prb->dt, prb->tag),
            CRIT);

    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        if (prb->alg == SOFTMAX)
            DNN_SAFE(dnnl_softmax_forward_desc_init(
                             &sd, prop, &data_d, prb->axis),
                    WARN);
        else if (prb->alg == LOGSOFTMAX)
            DNN_SAFE(dnnl_logsoftmax_forward_desc_init(
                             &sd, prop, &data_d, prb->axis),
                    WARN);
        else
            SAFE_V(FAIL);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, prb->ndims,
                         prb->dims.data(), prb->dt, dnnl_format_tag_any),
                WARN);
        if (prb->alg == SOFTMAX)
            DNN_SAFE(dnnl_softmax_backward_desc_init(
                             &sd, &diff_data_d, &data_d, prb->axis),
                    WARN);
        else if (prb->alg == LOGSOFTMAX)
            DNN_SAFE(dnnl_logsoftmax_backward_desc_init(
                             &sd, &diff_data_d, &data_d, prb->axis),
                    WARN);
        else
            SAFE_V(FAIL);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&spd, &sd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(spd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(spd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

static int compare(const prb_t *prb, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *res) {
    const int f32_mant_digits = 24;
    const float trh_coeff_dt = (1 << (f32_mant_digits - digits_dt(prb->dt)));
    const float trh_coeff_log = prb->alg == LOGSOFTMAX ? 4 : 1;
    const float trh = trh_coeff_dt * trh_coeff_log * 1e-6;

    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp = fp_mem.get_elem(i);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        // check for abs error
        if (!ok) ok = diff < 1e-7;

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

int fill_data_fwd(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    int64_t outer_size = 0, inner_size = 0, axis_size = 0;
    get_sizes(prb, outer_size, inner_size, axis_size);

    // Fill data the way it tests two modes: max_val < 0 and max_val >= 0;
    // Test max_val < 0 by using only negative numbers to check correct max_val
    // subtraction, mostly if library used signed value, not abs.
    // Test max_val >= 0 by exceeding `exp_overflow_arg` value to check answer
    // does not contain +infinity (nan).
    // Distribute several top-1 values to check softmax works right. Also use
    // bit more top-2 values so they contribute in final exp sum as well. Fill
    // much more values with top-3 to check we apply correct maths for whole
    // input.
    // Filling data such way prevents cancellation error for LOGSOFTMAX due to
    // log(sum(x_j)) won't be close to zero as in case of single top-1 value.
    const int exp_overflow_arg = 88;
    const int top1_val = exp_overflow_arg + 2;
    const int top2_val = exp_overflow_arg + 1;
    const int top3_val = exp_overflow_arg;
    const float top1_prob = 4. / axis_size;
    const float top2_prob = 7. * top1_prob;
    const float top3_prob = 3. * top2_prob;

    dnnl::impl::parallel_nd(outer_size, axis_size, inner_size,
            [&](int64_t ou, int64_t as, int64_t in) {
                const int sign = (outer_size > 1 ? ou : in) % 2 == 0 ? -1 : 1;
                const int gen = 13 * ou + 101 * as + 7 * in + 1637;
                const bool top1 = flip_coin(gen, top1_prob);
                const bool top2 = !top1 && flip_coin(gen, top2_prob);
                const bool top3 = !top1 && !top2 && flip_coin(gen, top3_prob);
                const int value = sign
                        * (top1 * top1_val + top2 * top2_val + top3 * top3_val);
                const int64_t ou_in_offset = ou * axis_size * inner_size + in;
                mem_fp.set_elem(ou_in_offset + as * inner_size, value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int seed) {
    const auto nelems = mem_fp.nelems();
    const int range = 128;

    // to avoid any cancellation erros it's better to have d_dst and dst of
    // different signs (refer to ref computations).
    // softmax := (d_dst - SUM (d_dst * dst); keep +d_dst and -dst.
    // logsoftmax := d_dst - exp(dst) * SUM (d_dst); keep -d_dst and +dst.
    // seed decides about the sign.
    const float sign = seed % 2 == 0 ? 1.f : -1.f;
    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((11 * i) + 37 + 19 * seed) % range;
        const float value = sign * gen / range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

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

    const auto &data_md = q(DNNL_ARG_DST); // src_md is not defined for BWD
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, fp, tag, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) { placeholder_dst_dt = dnn_mem_t(data_md, test_engine); }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    args_t args;

    if (prb->dir & FLAG_FWD) {
        SAFE(fill_data_fwd(prb, src_dt, src_fp), WARN);

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(s, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(prb, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare(prb, dst_fp, dst, res), WARN);
        }
    } else {
        const auto &d_data_md = q(DNNL_ARG_DIFF_DST);

        dnn_mem_t d_dst_fp = dnn_mem_t(d_data_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place reference
        if (!prb->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, test_engine);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        const bool neg_sign = prb->alg == SOFTMAX ? true : false;
        SAFE(fill_data_bwd(prb, src_dt, src_fp, neg_sign), WARN);
        SAFE(fill_data_bwd(prb, d_dst_dt, d_dst_fp, !neg_sign), WARN);

        args.set(DNNL_ARG_DST, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(s, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(prb, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, test_engine);
            SAFE(compare(prb, d_src_fp, d_src, res), WARN);
        }
    }

    measure_perf(res->timer, s, args);

    DNN_SAFE_V(dnnl_primitive_destroy(s));

    return OK;
}

} // namespace softmax
