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

#include <cmath>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

int get_n_scales(const prb_t *prb) {
    const int mask = attr_t::get_default_mask(prb->attr.oscale.policy);
    assert(IMPLICATION(mask >= (1 << 1), prb->ndims > 1));
    switch (mask) {
        case 0: return 1;
        case (1 << 0): return prb->reorder.dims[0];
        case (1 << 1): return prb->reorder.dims[1];
        case (1 << 1) + (1 << 0):
            return prb->reorder.dims[1] * prb->reorder.dims[0];
        default: assert(!"unsupported mask"); return 1;
    }
}

int fill_memory(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem) {
    const dt_conf_t c_src = prb->conf_in;
    const auto dt = c_src->dt;
    const int range = c_src->range;
    const int max = c_src->min + range - 1;
    const int scale_mask = attr_t::get_default_mask(prb->attr.oscale.policy);

    const auto nelems = mem.nelems();

    for (int64_t idx = 0; idx < nelems; ++idx) {
        const int64_t mask_idx = mem.get_scale_idx(idx, scale_mask);
        const float scale = prb->scales[mask_idx];

        const float gen[7] = {
                (float)max, /* saturate to max of output data type */
                (float)c_src->min, /* saturate to min of output data type */
                (float)1.6 / scale, /* rounding check */
                (float)0.2 / scale, /* saturate to 0 */
                (float)1.0,
                (float)2.0,
                (float)scale,
        };

        const int rng = kind == SRC ? (idx % 7) : ((idx * 8 / 7) % 7);
        mem.set_elem(idx, round_to_nearest_representable(dt, gen[rng]));
    }

    return OK;
}

int fill_memory_extra(const prb_t *prb, dnnl_memory_extra_desc_t &extra) {
    extra.flags = dnnl_memory_extra_flag_none;

    if (prb->is_reorder_with_compensation()) {
        int with_groups
                = (prb->oflag & (FLAG_GCONV_S8S8 | FLAG_GCONV_ZP_COMP)) ? 1 : 0;
        if (prb->oflag & (FLAG_CONV_S8S8 | FLAG_GCONV_S8S8)) {
            extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            extra.compensation_mask = (1 << 0) + with_groups * (1 << 1);
        }
        if (prb->oflag & (FLAG_CONV_ZP_COMP | FLAG_GCONV_ZP_COMP)) {
            extra.flags
                    |= dnnl_memory_extra_flag_compensation_conv_asymmetric_src;
            extra.asymm_compensation_mask = (1 << 0) + with_groups * (1 << 1);
        }
    }

    return OK;
}

int ref_reorder(const prb_t *prb, dnn_mem_t &dst, const dnn_mem_t &src) {
    auto dst_dt = dst.dt();

    const auto nelems = src.nelems();
    const int scale_mask = attr_t::get_default_mask(prb->attr.oscale.policy);
    const int src_zero_point = prb->src_zp ? prb->src_zp[0] : 0;
    const int dst_zero_point = prb->dst_zp ? prb->dst_zp[0] : 0;

    float beta = 0;
    const auto &po = prb->attr.post_ops;
    const int beta_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
    if (beta_idx >= 0) beta = po.entry[beta_idx].sum.scale;

    for (int64_t idx = 0; idx < nelems; ++idx) {
        float s = src.get_elem(idx) - src_zero_point;
        float d = 0;
        if (beta_idx >= 0) d = dst.get_elem(idx) - dst_zero_point;

        const int64_t scale_idx = dst.get_scale_idx(idx, scale_mask);
        const float alpha = prb->scales[scale_idx];
        const float value = alpha * s + beta * d + dst_zero_point;

        dst.set_elem(idx, round_to_nearest_representable(dst_dt, value));
    }

    return OK;
}

int compare_bootstrap(dnn_mem_t &mem_ref, dnn_mem_t &mem_got, res_t *res) {
    bool ok = false;
    // demand bit-wise identical results
    const auto size_ref = mem_ref.size();
    if (size_ref == 0) return res->state = PASSED, OK;

    if (size_ref == mem_got.size())
        ok = !memcmp((void *)mem_ref, (void *)mem_got, size_ref);

    res->errors = !ok;
    res->state = ok ? PASSED : FAILED;
    res->total = 1;

    return res->state == FAILED ? FAIL : OK;
}

static int compare(const prb_t *prb, const dnn_mem_t &mem_ref,
        const dnn_mem_t &mem_got, res_t *res) {
    const auto nelems = mem_got.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    int64_t inf_p = 0, inf_n = 0, zeros = 0, reg = 0;

    const auto dt_out = mem_ref.dt();
    const size_t width = mem_ref.sizeof_dt() * 8;
    const float dt_out_min
            = dt_out == dnnl_u8 ? 0.f : -(float)(1l << (width - 1));
    const float dt_out_max
            = dt_out == dnnl_u8 ? 255.f : (float)((1l << (width - 1)) - 1);
    const float tolerance = (dt_out == dnnl_bf16)
            ? 4e-3 // due to bf16 truncation (7th mantissa bit -> 1/129)
            : 0.;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = mem_got.get_elem(i);
        const float fp = mem_ref.get_elem(i);

        if (fp == dt_out_max)
            inf_p++;
        else if (fp == dt_out_min)
            inf_n++;
        else if (fp == 0.0)
            zeros++;
        else
            reg++;

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = rel_diff <= tolerance;

        // f32->f16 results in inf for FLT_MAX input
        if (!ok) ok = std::isinf(fp) && std::isinf(dt);

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(prb->reorder.dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp:% 12.6g dt:% 12.6g diff:%8.3g rdiff:%8.3g\n",
                    (long)i, ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    if (res->state != FAILED) {
        float max_scale = prb->scales[0];
        for (int i = 1; i < get_n_scales(prb); ++i)
            max_scale = MAX2(max_scale, prb->scales[i]);

        dt_conf_t c_src = prb->conf_in;
        dt_conf_t c_dst = prb->conf_out;
        const int c_src_max = c_src->min + c_src->range - 1;
        const int c_dst_max = c_dst->min + c_dst->range - 1;

        bool check_int_overflow = (dt_out != dnnl_f32 && dt_out != dnnl_f16
                && dt_out != dnnl_bf16);
        bool check_inf_p = (check_int_overflow && dt_out != dnnl_s32)
                && (c_src_max * max_scale > c_dst_max);
        bool check_inf_n = (check_int_overflow && dt_out != dnnl_s32)
                && (c_src->min * max_scale < c_dst->min);
        bool check_zeros = (check_int_overflow)
                && (dt_out_min != 0 && dt_out_max != 0)
                && IMPLICATION(prb->src_zp, prb->src_zp[0] == 0)
                && IMPLICATION(prb->dst_zp, prb->dst_zp[0] == 0)
                && prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        == -1;

        bool mistrusted = (check_inf_p && inf_p == 0)
                || (check_inf_n && inf_n == 0) || (check_zeros && zeros == 0);

        bool expect_regular = max_scale < 2e9 || dt_out == dnnl_f32;
        if (expect_regular) mistrusted = mistrusted || reg == 0;

        if (mistrusted) res->state = MISTRUSTED;
    }

    return res->state == FAILED ? FAIL : OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &rpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    const auto &rc = prb->reorder;
    auto dims = rc.dims;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    dnnl_memory_desc_t src_d, dst_d;
    SAFE(init_md(&src_d, prb->ndims, dims.data(), prb->conf_in->dt, rc.tag_in),
            CRIT);
    SAFE(init_md(&dst_d, prb->ndims, dims.data(), prb->conf_out->dt,
                 rc.tag_out),
            CRIT);

    // assign extra for dst_md
    dnnl_memory_extra_desc_t dst_md_extra {};
    fill_memory_extra(prb, dst_md_extra);
    dst_d.extra = dst_md_extra;

    dnnl_engine_t src_engine = engine, dst_engine = engine;
    if (engine_tgt_kind == dnnl_gpu) {
        switch (prb->cross_engine) {
            case CPU2GPU: src_engine = get_cpu_engine(); break;
            case GPU2CPU: dst_engine = get_cpu_engine(); break;
            default: break;
        }
    }

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, get_n_scales(prb));
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status = dnnl_reorder_primitive_desc_create(
            &rpd, &src_d, src_engine, &dst_d, dst_engine, dnnl_attr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(rpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    const auto sdt = prb->conf_in->dt;
    const auto ddt = prb->conf_out->dt;
    check_known_skipped_case_common({sdt, ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    // zero points for dst do not support sum by design
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)
            && prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) != -1) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (prb->is_reorder_with_compensation()) {
        // compensation is supported for dst_dt = s8 so far
        // compensation does not support any attributes or runtime dims
        if (ddt != dnnl_s8 || !prb->attr.is_def()
                || prb->runtime_dim_mask != 0) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    // bf16 reorder on cpu supports only bf16/f32 src_dt/dst_dt
    if (engine_tgt_kind == dnnl_cpu
            && (!IMPLICATION(sdt == dnnl_bf16,
                        ddt == dnnl_f32 || ddt == dnnl_bf16 || ddt == dnnl_s8
                                || ddt == dnnl_u8)
                    || !IMPLICATION(ddt == dnnl_bf16,
                            sdt == dnnl_f32 || sdt == dnnl_bf16
                                    || sdt == dnnl_s8 || sdt == dnnl_u8))) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (engine_tgt_kind == dnnl_gpu) {
        // GPU does not support run-time dims and zero-points
        if (prb->runtime_dim_mask != 0 || !prb->attr.zero_points.is_def()
                || prb->attr.oscale.runtime) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    //                                       ___________________
    //                                      |                   |
    //                                      | performance timer |
    //                                      |___________________|
    //                                                |
    //   _______________           ______________     V     ________________
    //  |               | oneDNN  |              | oneDNN  |                |
    //  | dt_in fmt_ref |-------->| dt_in fmt_in |-------->| dt_out fmt_out |
    //  |_______________|         |______________|    ^    |________________|
    //           |                                    |            |
    //  benchdnn |<-------------------------------- scales         | oneDNN
    //   ________V_______                                   _______V________
    //  |                |                                 |                |
    //  | dt_out fmt_ref |         <= compare =>           | dt_out fmt_ref |
    //  |________________|                                 |________________|
    //
    // Steps:
    // 1. fill scales
    // 2. create target reorder primitive
    // 3. create memories
    // 4. fill input memory
    // 5. execute oneDNN and benchdnn reorders / q10n
    // 6. compare results
    // 7. performance measurement

    /* Step 2: create target reorder primitive */
    dnnl_primitive_t rp {};
    SAFE(init_prim(&rp, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(rp, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(rp));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    /* Step 3: create memories */
    dnnl_memory_desc_t src_md, dst_md;
    if (prb->runtime_dim_mask != 0) {
        // re-create memory descriptors with defined dims
        const auto &rc = prb->reorder;
        SAFE(init_md(&src_md, prb->ndims, rc.dims.data(), prb->conf_in->dt,
                     rc.tag_in),
                CRIT);
        SAFE(init_md(&dst_md, prb->ndims, rc.dims.data(), prb->conf_out->dt,
                     rc.tag_out),
                CRIT);
    } else {
        src_md = q(DNNL_ARG_SRC);
        dst_md = q(DNNL_ARG_DST);
    }
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto tag = tag::abx;
    const auto src_dt = src_md.data_type;
    const auto dst_dt = dst_md.data_type;

    dnnl_engine_t src_engine, dst_engine;
    DNN_SAFE(dnnl_primitive_desc_query(
                     const_pd, dnnl_query_reorder_src_engine, 0, &src_engine),
            WARN);
    DNN_SAFE(dnnl_primitive_desc_query(
                     const_pd, dnnl_query_reorder_dst_engine, 0, &dst_engine),
            WARN);

    dnn_mem_t src_dt_in_fmt_ref(src_md, src_dt, tag, src_engine);
    dnn_mem_t src_dt_in_fmt_in(src_md, src_engine);

    dnn_mem_t scratchpad_dt(scratchpad_md, src_engine);

    dnn_mem_t dst_dt_out_fmt_ref(dst_md, dst_dt, tag, dst_engine);
    dnn_mem_t dst_dt_out_fmt_out(dst_md, dst_engine);

    /* Step 4: fill input memory */
    SAFE(fill_memory(prb, SRC, src_dt_in_fmt_ref), WARN);

    /* Step 5: execute necessary reorders */
    SAFE(src_dt_in_fmt_in.reorder(src_dt_in_fmt_ref), WARN);

    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0) {
        SAFE(fill_memory(prb, DST, dst_dt_out_fmt_ref), WARN);
        SAFE(dst_dt_out_fmt_out.reorder(dst_dt_out_fmt_ref), WARN);
    }

    dnn_mem_t scales, src_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(
            scales, prb->attr, get_n_scales(prb), prb->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, 1, prb->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, 1, prb->dst_zp);

    args_t args;
    args.set(DNNL_ARG_FROM, src_dt_in_fmt_in);
    args.set(DNNL_ARG_TO, dst_dt_out_fmt_out);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);

    SAFE(execute_and_wait(rp, args), WARN);

    /* Step 6: check correctness */
    if (bench_mode & CORR) {
        if (prb->is_reorder_with_compensation()) {
            /* "bootstrap" algorithm: compare to another oneDNN reorder. use
             * this when benchdnn does not know about all details of the data
             * layout, as is the case for compensated weights formats. */

            /* Step 5a: oneDNN reorder from ref format to output format */
            dnnl_memory_extra_desc_t dst_extra {};
            fill_memory_extra(prb, dst_extra);
            dnn_mem_t ref_dst_dt_out_fmt_out(dst_md, dst_engine);
            ref_dst_dt_out_fmt_out.md_.extra = dst_extra;

            SAFE(ref_dst_dt_out_fmt_out.reorder(src_dt_in_fmt_ref), WARN);

            /* Step 5b: compare results (expect bit-wise exactness) */
            SAFE(compare_bootstrap(
                         ref_dst_dt_out_fmt_out, dst_dt_out_fmt_out, res),
                    WARN);
        } else {
            /* (default) "reference" algorithm: compare to benchdnn reorder */

            /* Step 5b: execute benchdnn reorder */
            SAFE(ref_reorder(prb, dst_dt_out_fmt_ref, src_dt_in_fmt_ref), WARN);

            /* Step 5c: compare benchdnn and oneDNN output */
            dnn_mem_t dst_dt_out(dst_md, dst_dt, tag, dst_engine);
            SAFE(dst_dt_out.reorder(dst_dt_out_fmt_out), WARN);
            SAFE(compare(prb, dst_dt_out_fmt_ref, dst_dt_out, res), WARN);
        }
    }

    /* Step 7: performance measurement */
    measure_perf(res->timer, rp, args);

    DNN_SAFE_V(dnnl_primitive_destroy(rp));

    return OK;
}

} // namespace reorder
