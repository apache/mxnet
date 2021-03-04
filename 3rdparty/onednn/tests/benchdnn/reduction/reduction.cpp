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

#include <math.h>

#include <sstream>

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reduction/reduction.hpp"

namespace reduction {

int init_pd(dnnl_engine_t engine, const prb_t *prb, dnnl_primitive_desc_t &rpd,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint) {
    dnnl_reduction_desc_t rd;
    dnnl_memory_desc_t src_desc, dst_desc;

    SAFE(init_md(&src_desc, prb->ndims, prb->src_dims.data(), prb->sdt,
                 prb->stag),
            WARN);

    SAFE(init_md(&dst_desc, prb->ndims, prb->dst_dims.data(), prb->ddt,
                 prb->dtag),
            WARN);

    DNN_SAFE(dnnl_reduction_desc_init(&rd, alg2alg_kind(prb->alg), &src_desc,
                     &dst_desc, prb->p, prb->eps),
            WARN);

    attr_args_t attr_args;
    const auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&rpd, &rd, dnnl_attr, engine, nullptr);

    DNN_SAFE(dnnl_primitive_attr_destroy(dnnl_attr), WARN);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(rpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto dt = mem_dt.dt();
    const int range = prb->alg == alg_t::MUL
            ? (dt == dnnl_u8 || dt == dnnl_s8) ? 1024 : 4
            : 16;
    const int f_min = dt == dnnl_u8 ? 1 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) + 101) % (range + 1);
        float value = 0.0f;
        if (prb->alg == alg_t::MUL) {
            if (dt == dnnl_s8 || dt == dnnl_u8) {
                // generate {1, 2}, but probability of 2 is 1/range
                value = gen == range ? 2.0f : 1.0f;
            } else {
                // generate {-2, -0.5, 1, 0.5, 2} to avoid underflow/overflow
                value = powf(f_min + gen, 2.0f) / 2;
                if (f_min + gen != 0.0f) {
                    const float sign = fabs(f_min + gen) / (f_min + gen);
                    value *= sign;
                } else {
                    value = 1.0f;
                }
            }
        } else {
            value = (dt == dnnl_bf16 || dt == dnnl_f16)
                    ? (f_min + gen) / range
                    : (f_min + gen) * (1.0f + 4.0f / range);
        }
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int compare(const prb_t *prb, const dnn_mem_t &fp_mem, const dnn_mem_t &dt_mem,
        res_t *res) {
    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    int non_zero = 0;
    const double trust_nz_level = 0.5;

    const float trh = epsilon_dt(prb->ddt == dnnl_f16 ? dnnl_f16 : dnnl_f32);

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = round_to_nearest_representable(prb->ddt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        res->errors += !ok;

        const bool dump = (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(prb->dst_dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp0:%8g fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp0, fp, dt, diff, rel_diff);
        }

        non_zero += fp != 0.0f;
    }

    if (res->errors) res->state = FAILED;

    const double trust_nz = (double)non_zero / res->total;
    if (trust_nz < trust_nz_level) {
        if (res->state != FAILED) res->state = MISTRUSTED;
    }

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->sdt, prb->ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    if (engine_tgt_kind != dnnl_cpu) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    bool is_invalid = false;
    switch (prb->alg) {
        case alg_t::MEAN:
            is_invalid = prb->sdt != dnnl_f32 && prb->sdt != dnnl_bf16;
            break;
        case alg_t::NORM_LP_MAX:
        case alg_t::NORM_LP_SUM:
        case alg_t::NORM_LP_POWER_P_MAX:
        case alg_t::NORM_LP_POWER_P_SUM:
            is_invalid = (prb->sdt != dnnl_f32 && prb->sdt != dnnl_bf16)
                    || prb->p < 1.f;
            break;
        default: break;
    }
    if (is_invalid) res->state = SKIPPED, res->reason = INVALID_CASE;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t reduction {};
    SAFE(init_prim(&reduction, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(reduction, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(reduction));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto fp_dt = dnnl_f32;
    const auto abx_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    const auto &src_md = q(DNNL_ARG_SRC);
    dnn_mem_t src_fp(src_md, fp_dt, abx_tag, test_engine);
    dnn_mem_t src_dt(src_md, test_engine);
    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    const auto &dst_md = q(DNNL_ARG_DST);
    dnn_mem_t dst_fp(dst_md, fp_dt, abx_tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);

    SAFE(execute_and_wait(reduction, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src_fp, dst_fp);
        dnn_mem_t dst(dst_dt, fp_dt, abx_tag, test_engine);
        SAFE(compare(prb, dst_fp, dst, res), WARN);
    }

    measure_perf(res->timer, reduction, args);

    DNN_SAFE_V(dnnl_primitive_destroy(reduction));

    return OK;
}

} // namespace reduction
