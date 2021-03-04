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

#include <float.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "ip/ip.hpp"

namespace ip {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &ippd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_inner_product_desc_t ipd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_dims_0d = {prb->mb, prb->ic};
    dnnl_dims_t src_dims_1d = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dims_t wei_dims_0d = {prb->oc, prb->ic};
    dnnl_dims_t wei_dims_1d = {prb->oc, prb->ic, prb->iw};
    dnnl_dims_t wei_dims_2d = {prb->oc, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t wei_dims_3d = {prb->oc, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dims_t bia_dims = {prb->oc};
    dnnl_dims_t dst_dims = {prb->mb, prb->oc};

    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_dims_3d
            : prb->ndims == 4 ? src_dims_2d
                              : prb->ndims == 3 ? src_dims_1d : src_dims_0d;

    dnnl_dim_t *wei_dims = prb->ndims == 5
            ? wei_dims_3d
            : prb->ndims == 4 ? wei_dims_2d
                              : prb->ndims == 3 ? wei_dims_1d : wei_dims_0d;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->cfg[SRC].dt, prb->stag),
            CRIT);
    SAFE(init_md(&wei_d, prb->ndims, wei_dims, prb->cfg[WEI].dt, prb->wtag),
            CRIT);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&bia_d, 1, bia_dims, prb->cfg[BIA].dt,
                     dnnl_format_tag_any),
            WARN);
    SAFE(init_md(&dst_d, 2, dst_dims, prb->cfg[DST].dt, prb->dtag), CRIT);

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_inner_product_forward_desc_init(&ipd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_inner_product_backward_data_desc_init(
                             &ipd, &src_d, &wei_d, &dst_d),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_inner_product_backward_weights_desc_init(&ipd, &src_d,
                             &wei_d, prb->dir == BWD_W ? nullptr : &bia_d,
                             &dst_d),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(ipd.accum_data_type == prb->cfg[ACC].dt ? dnnl_success
                                                     : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_binary_post_op_mds(prb->attr, 2, dst_dims);
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &ippd, &ipd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(ippd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(ippd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

inline int compare_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    int64_t non_zero = 0;
    const char *skind = data_kind2str(kind);

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(prb->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok
                = (fabs(fp) > 1e-5 ? rel_diff : diff) <= prb->cfg[kind].eps;

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            BENCHDNN_PRINT(0,
                    "[%4ld][%s]"
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, skind, fp, fp0, dt, diff, rel_diff);
        }
        non_zero += fp != 0;
    }

    const double trust_nz = (double)non_zero / res->total;
    bool no_trust = trust_nz < 0.1;
    if (no_trust) {
        res->state = MISTRUSTED;
        const char *skind = data_kind2str(kind);
        BENCHDNN_PRINT(0,
                "@@@ [%s] test-bug: trust is too low."
                " Nonzeros in output: %.2f\n",
                skind, trust_nz);
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = prb->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int mb, int ic, int id, int ih, int iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;

                ((float *)mem_00)[src_off_f(prb, mb, ic, id, ih, iw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool s8_s8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dnnl_s8;
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();
    const bool check_reorder = diff_data_type && !s8_s8;

    dnn_mem_t extra_mem;
    if (check_reorder) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const auto &c = prb->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->oc, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int oc, int ic, int kd, int kh, int kw) {
                const int gen
                        = 127 * kd + 131 * kh + 137 * kw + 139 * oc + 149 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_00)[wei_off_f(prb, oc, ic, kd, kh, kw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem)
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::x, get_test_engine());
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const size_t nelems = mem_00.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }
    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = prb->cfg[DST];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->oc, [&](int mb, int oc) {
        const int gen = 173 * mb + 179 * oc;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_00)[dst_off_f(prb, mb, oc)] = value;
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    if (res->state == SKIPPED) return;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t ip {};
    SAFE(init_prim(&ip, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(ip, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(ip));
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

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, tag::abx, test_engine);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    args_t args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(test_engine, prb, src_fp, wei_fp, bia_fp,
                    binary_po_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag::abx, test_engine);
            SAFE(compare_dat(prb, DST, dst, dst_fp, res), WARN);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(prb, src_fp, wei_fp, dst_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_dat(prb, SRC, src, src_fp, res), WARN);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_w(prb, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            if (compare_dat(prb, WEI, wei, wei_fp, res) != OK) return FAIL;
            if (prb->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, tag::x, test_engine);
                SAFE(compare_dat(prb, BIA, bia, bia_fp, res), WARN);
            }
        }
    }

    measure_perf(res->timer, ip, args);

    DNN_SAFE_V(dnnl_primitive_destroy(ip));

    return OK;
}

} // namespace ip
