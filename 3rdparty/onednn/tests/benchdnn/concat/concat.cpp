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

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "concat/concat.hpp"

namespace concat {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(prb->n_inputs());

    dnnl_memory_desc_t dst_d;

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_sdims = prb->sdims[i_input];
        SAFE(init_md(&src_d[i_input], prb->ndims, i_sdims.data(), prb->sdt,
                     prb->stag[i_input]),
                CRIT);
    }

    if (prb->dtag != tag::undef) {
        SAFE(init_md(&dst_d, prb->ndims, prb->ddims.data(), prb->ddt,
                     prb->dtag),
                CRIT);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status = dnnl_concat_primitive_desc_create(&cpd,
            prb->dtag != tag::undef ? &dst_d : nullptr, prb->n_inputs(),
            prb->axis, src_d.data(), dnnl_attr, engine);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(cpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

static int compare(const prb_t *prb, const dnnl_data_type_t dst_data_type,
        const dnn_mem_t &fp_mem, const dnn_mem_t &dt_mem, res_t *res) {
    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = round_to_nearest_representable(dst_data_type, fp0);

        const bool ok = dt == fp; // expect exact answer due to int values

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t ddims_idx = off2dims_idx(prb->ddims, i);
            ss << ddims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0, "[%4ld][%s] fp0:%8g fp:%8g dt:%8g\n", (long)i,
                    ind_str.c_str(), fp0, fp, dt);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int fill_src(
        const prb_t *prb, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    auto get_range = [](const dnnl_data_type_t dt) {
        if (dt == dnnl_s8 || dt == dnnl_u8)
            return 256;
        else if (dt == dnnl_bf16 || dt == dnnl_f16)
            return 128;
        return 1024;
    };

    const auto nelems = mem_fp.nelems();
    const int range = get_range(prb->sdt);
    const int f_min = prb->sdt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * input_idx + 101) % range;
        const float value = f_min + gen;
        mem_fp.set_elem(i, round_to_nearest_representable(prb->sdt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->sdt, prb->ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    // ref concat is reorder-based, hence, inherits some reorder limitations.
    // bf16 reorder on cpu supports only bf16/f32 src_dt/dst_dt
    bool valid_bf16_input = IMPLICATION(prb->sdt == dnnl_bf16,
            prb->dtag == tag::undef || prb->ddt == dnnl_f32
                    || prb->ddt == dnnl_bf16);
    bool valid_bf16_output
            = IMPLICATION(prb->ddt == dnnl_bf16 && prb->dtag != tag::undef,
                    (prb->sdt == dnnl_f32 || prb->sdt == dnnl_bf16));

    if (engine_tgt_kind == dnnl_cpu
            && (!valid_bf16_input || !valid_bf16_output)) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t c {};
    SAFE(init_prim(&c, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(c));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &dst_md = q(DNNL_ARG_DST);
    const auto dst_data_type = dst_md.data_type; // needed for deduced dst

    const auto &test_engine = get_test_engine();

    dnn_mem_t dst_fp(dst_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);

    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args;
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const auto &src_md = q(DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, fp, tag, test_engine);
        src_dt.emplace_back(src_md, test_engine);
        SAFE(fill_src(prb, i_input, src_dt[i_input], src_fp[i_input]), WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
    }

    SAFE(execute_and_wait(c, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src_fp, dst_fp);

        // convert dst_fp into target precision back and forth for proper
        // answer comparison
        dnn_mem_t dst_fp_dt(dst_fp, dst_data_type, tag, test_engine);
        SAFE(dst_fp_dt.reorder(dst_fp), WARN);
        SAFE(dst_fp.reorder(dst_fp_dt), WARN);

        dnn_mem_t dst(dst_dt, fp, tag, test_engine);
        SAFE(compare(prb, dst_data_type, dst_fp, dst, res), WARN);
    }

    measure_perf(res->timer, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));

    return OK;
}

} // namespace concat
