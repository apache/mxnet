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

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace binary {

int fill_src(int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 5 * input_idx + 101) % (range + 1);
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int setup_binary_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &mem_dt, std::vector<dnn_mem_t> &mem_fp) {
    // TODO: currently run-time dimensions are not supported in binary post-op.
    // To add a support two ways are possible: 1) add query support to the
    // library and extract expected md from pd; 2) pass a vector of pre-defined
    // (no run-time values) of `po_md`s and create memories from them in case
    // the library will lack of query mechanism.
    const_dnnl_primitive_attr_t const_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(pd, &const_attr), WARN);

    const_dnnl_post_ops_t const_attr_po;
    DNN_SAFE(
            dnnl_primitive_attr_get_post_ops(const_attr, &const_attr_po), WARN);

    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind != dnnl_binary) continue;

        const dnnl_memory_desc_t *po_md;
        DNN_SAFE(dnnl_post_ops_get_params_binary(
                         const_attr_po, idx, nullptr, &po_md),
                WARN);

        const auto tag = tag::abx;
        // Following call can not be executed if po_md has runtime dimension due
        // to undefined size.
        mem_fp.emplace_back(*po_md, dnnl_f32, tag, get_test_engine());
        mem_dt.emplace_back(*po_md, get_test_engine());
        args.push_back((DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));

        fill_src((DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1),
                mem_dt.back(), mem_fp.back());
    }
    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &bpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_binary_desc_t bd;
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_sdims = prb->sdims[i_input];
        SAFE(init_md(&src_d[i_input], prb->ndims[i_input], i_sdims.data(),
                     prb->sdt[i_input], prb->stag[i_input]),
                CRIT);
    }

    if (prb->ndims[1] < prb->ndims[0]) { // need to reshape B
        dnnl_dims_t dims;
        for (int d = 0; d < prb->ndims[1]; ++d)
            dims[d] = prb->sdims[1][d];
        for (int d = prb->ndims[1]; d < prb->ndims[0]; ++d)
            dims[d] = 1;
        DNN_SAFE(dnnl_memory_desc_reshape(
                         &src_d[1], &src_d[1], prb->ndims[0], dims),
                WARN);
    }

    dnnl_memory_desc_t dst_d;
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, prb->ndims[0],
                     prb->sdims[0].data(), prb->ddt, dnnl_format_tag_any),
            WARN);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    DNN_SAFE(dnnl_binary_desc_init(&bd, alg, &src_d[0], &src_d[1], &dst_d),
            WARN);

    attr_args_t attr_args;
    attr_args.prepare_binary_post_op_mds(
            prb->attr, prb->ndims[0], prb->sdims[0].data());
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&bpd, &bd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(bpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

// Check that on a given input specific alg may return NaN or inf.
bool check_extreme_values(float a, float b, alg_t alg) {
    switch (alg) {
        case alg_t::DIV:
        // It is impossible to reliably test against reference in binary
        // post-op chain since some algs may produce inf or NaN in the middle
        // which is not expected for a standalone testing. Thus, when passing
        // alg == BINARY_END, accept this fact. This alg to be used when
        // comparing results with binary post-op chain.
        case alg_t::BINARY_END:
            if (std::isnan(a) && std::isnan(b)) return true;
            if (std::isinf(a) && std::isinf(b)
                    && std::signbit(a) == std::signbit(b))
                return true;
        default: break;
    }
    return false;
}

static int compare(const prb_t *prb, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *res) {
    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    float trh = epsilon_dt(prb->ddt == dnnl_f16 ? dnnl_f16 : dnnl_f32)
            * prb->n_inputs();

    // Update trh with the largest value from all eltwise post-ops
    const auto &po = prb->attr.post_ops;
    bool has_eltwise = po.eltwise_index() != -1;
    if (has_eltwise) {
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (e.is_eltwise_kind())
                trh = MAX2(
                        trh, eltwise::get_eltwise_threshold(prb->ddt, e.kind));
        }
    }

    const bool has_binary = po.binary_index() != -1;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = round_to_nearest_representable(prb->ddt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        // XXX: if reference fp0 value is nan, allow to return anything from the
        // library for integral target data types.
        if (!ok) ok = std::isnan(fp0) && is_integral_dt(prb->ddt);
        // XXX: fp16 result can slightly mismatch for division due to difference
        // in backends implementations
        if (!ok && prb->alg == alg_t::DIV) ok = diff <= epsilon_dt(prb->ddt);
        if (!ok) ok = check_extreme_values(fp, dt, prb->alg);
        if (!ok && has_eltwise)
            ok = eltwise::check_extreme_values(fp, dt, alg_t::ELTWISE_END);
        if (!ok && has_binary)
            ok = check_extreme_values(fp, dt, alg_t::BINARY_END);

        // XXX: CPU and OpenCL behavior of int8 saturation is not aligned for
        // NaN. Accroding to OpenCL 2.0 specification NaN value is saturated to
        // 0. On CPU library saturates NaN value into lowest value representable
        // in destination data type.
        // TODO: Check CUDA specification.
        if (!ok && std::isnan(fp0) && engine_tgt_kind == dnnl_gpu
                && (prb->ddt == dnnl_s8 || prb->ddt == dnnl_s32)) {
            ok = diff == 128;
        }

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(prb->sdims[0], i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp0:%8g fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp0, fp, dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(prb->sdt, FWD_D, res);
    if (res->state == SKIPPED) return;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t b {};
    SAFE(init_prim(&b, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(b, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(b));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src0_md = q(DNNL_ARG_SRC_0);
    const auto &src1_md = q(DNNL_ARG_SRC_1);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src0_fp(src0_md, fp, tag, test_engine);
    dnn_mem_t src0_dt(src0_md, test_engine);
    SAFE(fill_src(0, src0_dt, src0_fp), WARN);

    dnn_mem_t src1_fp(src1_md, fp, tag, test_engine);
    dnn_mem_t src1_dt(src1_md, test_engine);
    SAFE(fill_src(1, src1_dt, src1_fp), WARN);

    dnn_mem_t &dst_fp = src0_fp; // in-place in ref code
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) {
        const auto &dst_md = q(DNNL_ARG_DST);
        placeholder_dst_dt = dnn_mem_t(dst_md, test_engine);

        if (prb->attr.post_ops.find(alg_t::SUM) >= 0)
            SAFE(placeholder_dst_dt.reorder(dst_fp), WARN);
    }
    dnn_mem_t &dst_dt = prb->inplace ? src0_dt : placeholder_dst_dt;

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(setup_binary_po(const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    args_t args;
    args.set(DNNL_ARG_SRC_0, src0_dt);
    args.set(DNNL_ARG_SRC_1, src1_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(b, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src0_fp, src1_fp, binary_po_fp, dst_fp);
        dnn_mem_t dst(dst_dt, fp, tag, test_engine);
        SAFE(compare(prb, dst_fp, dst, res), WARN);
    }

    measure_perf(res->timer, b, args);

    DNN_SAFE_V(dnnl_primitive_destroy(b));

    return OK;
}

} // namespace binary
