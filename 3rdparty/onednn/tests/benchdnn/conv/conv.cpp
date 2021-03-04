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
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "binary/binary.hpp"
#include "conv/conv_common.hpp"
#include "eltwise/eltwise.hpp"

namespace conv {

double get_trust_nz_level(
        const prb_t *prb, data_kind_t kind, bool final_compare) {
    if (!final_compare) return prb->cfg[kind].f_sparsity;

    auto negative_to_zero = [&]() {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &po = prb->attr.post_ops;
        int count = 0;
        for (int i = 0; i < po.len(); ++i) {
            auto k = po.entry[i].kind;
            count += k == pk::RELU || k == pk::ELU || k == pk::SQRT
                    || k == pk::BRELU;
        }
        return !!count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC: trust /= prb->sd * prb->sh * prb->sw; break;
        case WEI:
            trust /= 1. * prb->kd * prb->kh * prb->kw
                    / MIN3(prb->kd * prb->kh * prb->kw,
                            prb->id * prb->ih * prb->iw,
                            prb->od * prb->oh * prb->ow);
            break;
        case BIA:
            trust = 0.8 * prb->cfg[DST].f_sparsity; /* why? */
            break;
        case DST: trust /= negative_to_zero() == 0 ? 1 : 2; break;
    }

    return trust;
}

inline bool post_ops_require_integral_check(const prb_t *prb) {
    const auto &po = prb->attr.post_ops;
    if (po.len() == 0) return false;

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        using pk_t = attr_t::post_ops_t::kind_t;

        if (e.kind == pk_t::SUM || e.kind == pk_t::ABS) continue;
        if (e.kind == pk_t::RELU && e.eltwise.alpha == 0.f) continue;
        return true;
    }

    return false;
}

inline double get_eps(const prb_t *prb, const data_kind_t kind) {
    // Winograd specifics
    if (prb->alg & WINO && prb->dir & FLAG_WEI) {
        /*This is an empirical equation derived by observing growth error
          with increasing 'k' dimension in gemm of winograd*/
        return prb->cfg[kind].eps
                * (MAX2(1,
                        pow(10,
                                0.4
                                        * log10(0.125 * prb->mb * prb->oh
                                                * prb->ow))));
    }

    // post-ops specifics
    if (post_ops_require_integral_check(prb))
        return MAX2(1e-5, prb->cfg[kind].eps);

    return prb->cfg[kind].eps;
}

inline void get_result(const prb_t *prb, const data_kind_t kind, res_t *res,
        const diff_norm_t diff_norm) {
    const float eps = get_eps(prb, kind);

    /* Ignoring element-wise errors for Winograd and in some cases of post-ops,
     * since large relative error in few elements (which are anyways close
     * to zero) results in false positive failures */

    bool wino_test = (prb->alg & WINO) && diff_norm.rel_diff(norm_t::L2) <= eps;
    if (wino_test) res->errors = 0;

    bool post_ops_test = post_ops_require_integral_check(prb)
            && diff_norm.rel_diff(norm_t::L2) <= eps;
    if (post_ops_test) res->errors = 0;

    if (res->errors) res->state = FAILED;
}

inline int compare_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res, bool final_compare = false) {

    const bool dont_complain = false || (prb->alg & WINO)
            || post_ops_require_integral_check(prb);

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return res->state = PASSED, OK;
    res->total = nelems;

    const char *skind = data_kind2str(kind);

    int in = 0, below = 0, above = 0;
    int in_ok = 0, below_ok = 0, above_ok = 0;
    int non_zero = 0;

    // Update trh with the largest value from all eltwise post-ops
    const auto &po = prb->attr.post_ops;
    bool has_eltwise = po.eltwise_index() != -1;
    const bool has_binary = po.binary_index() != -1;
    using pk_t = attr_t::post_ops_t::kind_t;

    int sum_ind = po.find(pk_t::SUM);
    auto sum_dt
            = (sum_ind != -1) ? po.entry[sum_ind].sum.dt : dnnl_data_type_undef;

    bool diff_sum_dt = kind == DST && !final_compare
            && sum_dt != dnnl_data_type_undef && sum_dt != prb->cfg[kind].dt;
    dnnl_data_type_t f_dt = diff_sum_dt ? sum_dt : prb->cfg[kind].dt;
    float f_min = diff_sum_dt ? lowest_dt(f_dt) : prb->cfg[kind].min;
    float f_max = diff_sum_dt ? max_dt(f_dt) : prb->cfg[kind].max;

    diff_norm_t diff_norm;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(f_dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = true;
        if (fp < f_min) {
            diff_norm.update(f_min, dt);
            ok = dt == f_min;
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(fp, dt, pk_t::ELTWISE_END);
            if (!ok && has_binary)
                ok = binary::check_extreme_values(fp, dt, pk_t::BINARY_END);
            below += 1;
            below_ok += ok;
        } else if (fp > f_max) {
            diff_norm.update(f_max, dt);
            ok = dt == f_max;
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(fp, dt, pk_t::ELTWISE_END);
            if (!ok && has_binary)
                ok = binary::check_extreme_values(fp, dt, pk_t::BINARY_END);
            above += 1;
            above_ok += ok;
        } else {
            diff_norm.update(fp, dt);
            float trh = get_eps(prb, kind);
            if (has_eltwise) {
                for (int i = 0; i < po.len(); ++i) {
                    const auto &e = po.entry[i];
                    if (e.is_eltwise_kind())
                        trh = MAX2(trh,
                                eltwise::get_eltwise_threshold(
                                        f_dt, e.kind, prb->dir & FLAG_FWD));
                }
            }
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= trh;
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(fp, dt, pk_t::ELTWISE_END);
            if (!ok && has_binary)
                ok = binary::check_extreme_values(fp, dt, pk_t::BINARY_END);
            in += 1;
            in_ok += ok;
        }

        res->errors += !ok;

        bool dump = (!ok
                            && ((!dont_complain && res->errors < 10)
                                    || verbose >= 10))
                || (final_compare
                        && ((verbose >= 50 && i < 30) || (verbose >= 99)));

        if (dump) {
            int64_t mb_or_g = 0, g_or_oc = 0, c = 0, d = 0, h = 0, w = 0;
            switch (kind) {
                case SRC:
                    inv_src_off_f(prb, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
                case WEI:
                    inv_wei_off_f(prb, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
                case BIA: inv_bia_off_f(prb, i, mb_or_g, g_or_oc); break;
                case DST:
                    inv_dst_off_f(prb, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
            }
            BENCHDNN_PRINT(0,
                    "[%4ld][%s%s]"
                    "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                    "] "
                    "fp:% 12.6g fp0:% 12.6g dt:% 12.6g diff:%8g rdiff:%8g\n",
                    (long)i, final_compare ? "" : "REORDER ", skind, mb_or_g,
                    g_or_oc, c, d, h, w, fp, fp0, dt, diff, rel_diff);
        }

        non_zero += fp != 0;
    }

    diff_norm.done();
    get_result(prb, kind, res, diff_norm);

    if (final_compare || res->errors) {
        const int vl = res->errors ? 0 : 2;
        BENCHDNN_PRINT(vl,
                "@@@ [%s] %sdiff: err:%d, l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, final_compare ? "final: " : "", (int)res->errors,
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    const double trust_rg_level = 0.3;
    const double trust_nz_level = get_trust_nz_level(prb, kind, final_compare);

    const double trust_rg = (double)in / res->total;
    const double trust_nz = (double)non_zero / res->total;

    const bool no_trust = true /* ...in the test ...at all */
            && final_compare
            && (trust_rg < trust_rg_level || trust_nz < trust_nz_level);

    const bool dump = verbose >= 20
            || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    if (dump) {
        BENCHDNN_PRINT(0,
                "@@@ [%s] %strust range:%.2f nz:%.2f "
                "(level range:%.2f nz:%.2f). "
                "in:%d (ok:%d) below:%d (ok:%d) above:%d (ok:%d) nz:%d "
                "total:%lu\n",
                skind, final_compare ? "final: " : "", trust_rg, trust_nz,
                trust_rg_level, trust_nz_level, in, in_ok, below, below_ok,
                above, above_ok, non_zero, (unsigned long)res->total);
    }

    if (no_trust) {
        if (res->state != FAILED) res->state = MISTRUSTED;
        BENCHDNN_PRINT(0,
                "@@@ [%s] test-bug: trust is too low. "
                "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %lu)\n",
                skind, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
                non_zero, (unsigned long)res->total);
    }

    if (final_compare && res->state == UNTESTED)
        res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int compare_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare) {
    return compare_dat(prb, SRC, mem_dt, mem_fp, res, final_compare);
}
int compare_wei(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare) {
    return compare_dat(prb, WEI, mem_dt, mem_fp, res, final_compare);
}
int compare_bia(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare) {
    return compare_dat(prb, BIA, mem_dt, mem_fp, res, final_compare);
}
int compare_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare) {
    return compare_dat(prb, DST, mem_dt, mem_fp, res, final_compare);
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = prb->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int mb, int ic, int id, int ih, int iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                float value = non_base ? c.f_min + gen * c.f_step % range
                                       : c.f_base;

                maybe_zero_point(
                        prb->attr, value, prb->src_zp, ic, DNNL_ARG_SRC, true);

                ((float *)mem_00)[src_off_f(prb, mb, 0, ic, id, ih, iw)]
                        = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_src(prb, mem_fp, mem_00, res), WARN);
    }

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool wino_s8 = prb->alg == WINO && prb->cfg[WEI].dt == dnnl_s8;
    const bool s8_s8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dnnl_s8;
    const bool is_def_zp = prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();
    const bool check_reorder
            = diff_data_type && !wino_s8 && !s8_s8 && is_def_zp;

    dnn_mem_t extra_mem;
    if (check_reorder) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const auto &c = prb->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw,
            [&](int g, int oc, int ic, int kd, int kh, int kw) {
                const int gen
                        = 127 * kd + 131 * kh + 137 * kw + 139 * oc + 149 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value = non_base ? c.f_min + gen * c.f_step % range
                                             : c.f_base;

                ((float *)mem_00)[wei_off_f(prb, g, oc, ic, kd, kh, kw)]
                        = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_wei(prb, mem_fp, mem_00, res), WARN);
    }

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
        const float value
                = non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float *)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_bia(prb, mem_fp, mem_00, res), WARN);
    }
    return OK;
}

int fill_dst_with_params(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        dnnl_data_type_t dt, double sparsity, int min, int max, int base,
        int step, res_t *res) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }

    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;
    const int range = max - min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->oc, prb->od, prb->oh, prb->ow,
            [&](int mb, int oc, int od, int oh, int ow) {
                const int gen
                        = 157 * od + 163 * oh + 167 * ow + 173 * mb + 179 * oc;
                const bool non_base = flip_coin(gen, sparsity);
                const float value = non_base ? min + gen * step % range : base;

                ((float *)mem_00)[dst_off_f(prb, mb, 0, oc, od, oh, ow)]
                        = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_dst(prb, mem_fp, mem_00, res), WARN);
    }

    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    auto dst_dt = mem_dt.dt();
    int sum_ind = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? prb->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;
    bool diff_sum_dst_types
            = sum_dt != dnnl_data_type_undef && sum_dt != dst_dt;

    const auto &c = prb->cfg[DST];
    float f_min = (diff_sum_dst_types) ? lowest_dt(sum_dt) : c.f_min;
    float f_max = (diff_sum_dst_types) ? max_dt(sum_dt) : c.f_max;

    // Change mem dt to sum dt, so we can save sum data properly.
    if (diff_sum_dst_types) { mem_dt.set_dt(sum_dt); }

    fill_dst_with_params(prb, mem_dt, mem_fp, sum_dt, c.f_sparsity, f_min,
            f_max, c.f_base, c.f_step, res);

    // Return dst data type back.
    if (diff_sum_dst_types) { mem_dt.set_dt(dst_dt); }
    return OK;
}

inline int init_pd_custom(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &cpd, res_t *res,
        dnnl_data_type_t src_dt = dnnl_data_type_undef,
        dnnl_data_type_t wei_dt = dnnl_data_type_undef,
        dnnl_data_type_t bia_dt = dnnl_data_type_undef,
        dnnl_data_type_t dst_dt = dnnl_data_type_undef,
        dnnl_data_type_t acc_dt = dnnl_data_type_undef,
        std::string src_tag = tag::undef, std::string wei_tag = tag::undef,
        std::string bia_tag = tag::undef, std::string dst_tag = tag::undef) {
    dnnl_convolution_desc_t cd;
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

    if (src_dt == dnnl_data_type_undef) src_dt = prb->cfg[SRC].dt;
    if (wei_dt == dnnl_data_type_undef) wei_dt = prb->cfg[WEI].dt;
    if (bia_dt == dnnl_data_type_undef) bia_dt = prb->cfg[BIA].dt;
    if (dst_dt == dnnl_data_type_undef) dst_dt = prb->cfg[DST].dt;
    if (acc_dt == dnnl_data_type_undef) acc_dt = prb->cfg[ACC].dt;
    if (src_tag == tag::undef) src_tag = normalize_tag(prb->stag, prb->ndims);
    if (wei_tag == tag::undef) wei_tag = normalize_tag(prb->wtag, prb->ndims);
    if (bia_tag == tag::undef) bia_tag = tag::any;
    if (dst_tag == tag::undef) dst_tag = normalize_tag(prb->dtag, prb->ndims);

    SAFE(init_md(&src_d, prb->ndims, src_dims, src_dt, src_tag), WARN);

    SAFE(init_md(&wei_d, prb->ndims + prb->has_groups, wei_dims, wei_dt,
                 wei_tag),
            WARN);

    SAFE(init_md(&bia_d, 1, bia_dims, bia_dt, bia_tag), WARN);

    SAFE(init_md(&dst_d, prb->ndims, dst_dims, dst_dt, dst_tag), WARN);

    dnnl_dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dnnl_dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
    dnnl_dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
    dnnl_dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - prb->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - prb->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - prb->ndims);

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (prb->alg == WINO) alg = dnnl_convolution_winograd;
    if (prb->alg == AUTO) alg = dnnl_convolution_auto;

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_convolution_forward_desc_init(&cd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_dilated_convolution_backward_data_desc_init(&cd, alg,
                             &src_d, &wei_d, &dst_d, strides, dilates, padding,
                             padding_r),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_dilated_convolution_backward_weights_desc_init(&cd,
                             alg, &src_d, &wei_d,
                             prb->dir == BWD_W ? nullptr : &bia_d, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == acc_dt ? dnnl_success : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_binary_post_op_mds(prb->attr, prb->ndims, dst_dims);
    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&cpd, &cd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (!res) return OK;

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(cpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(cpd), WARN);
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

    // Winograd implementation limitations.
    if (prb->alg == WINO) {
        if (engine_tgt_kind == dnnl_cpu) {
            static auto isa = dnnl_get_effective_cpu_isa();
            static bool has_avx512_common = isa >= dnnl_cpu_isa_avx512_mic;
            static bool has_avx512_bw = isa >= dnnl_cpu_isa_avx512_core;
            bool is_int8 = prb->cfg[WEI].dt == dnnl_s8;

            bool pad_ok_f32 = prb->pw <= 1 && prb->ph <= 1 && prb->pw_r <= 1
                    && prb->ph_r <= 1;
            bool pad_ok_int8 = prb->pw <= 1 && prb->ph <= 1
                    && prb->pw == prb->pw_r && prb->ph == prb->ph_r;

            bool shape_ok = prb->ndims == 4 && prb->g == 1 && prb->kh == 3
                    && prb->kw == 3 && prb->sh == 1 && prb->sw == 1
                    && prb->dh == 0 && prb->dw == 0
                    && IMPLICATION(!is_int8, pad_ok_f32)
                    && IMPLICATION(is_int8,
                            (prb->ic % 16 == 0) && (prb->oc % 16 == 0)
                                    && pad_ok_int8);
            bool bwd_is_syncable = IMPLICATION(
                    (prb->dir & FLAG_BWD), dnnl::impl::dnnl_thr_syncable());

            const auto stag = normalize_tag(prb->stag, prb->ndims);
            const bool stag_is_abx
                    = stag == normalize_tag(tag::abx, prb->ndims);
            const bool stag_is_axb
                    = stag == normalize_tag(tag::axb, prb->ndims);
            const auto dtag = normalize_tag(prb->dtag, prb->ndims);
            const bool dtag_is_abx
                    = dtag == normalize_tag(tag::abx, prb->ndims);
            const bool dtag_is_axb
                    = dtag == normalize_tag(tag::axb, prb->ndims);
            const bool is_plain
                    = stag_is_abx || stag_is_axb || dtag_is_abx || dtag_is_axb;
            const bool plain_ok = is_int8 && !stag_is_abx && !dtag_is_abx
                    && (stag_is_axb || dtag_is_axb);

            const auto &po = prb->attr.post_ops;
            const auto sum_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
            const bool sum_post_op_ok
                    = sum_idx == -1 || po.entry[sum_idx].sum.scale == 1.f;

            if (!has_avx512_common || !shape_ok || (!has_avx512_bw && is_int8)
                    || !bwd_is_syncable || (is_plain && !plain_ok)
                    || !sum_post_op_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        } else if (engine_tgt_kind == dnnl_gpu) {
            bool shape_ok = prb->ndims == 4 && prb->g == 1 && prb->kh == 3
                    && prb->kw == 3 && prb->sh == 1 && prb->sw == 1
                    && prb->dh == 0 && prb->dw == 0;
            if (!shape_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            }
            return;
        } else {
            assert(!"Unknown Engine");
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t c {};
    // TODO: align init_pd interface with a common one which is used
    // in the rest of the benchdnn drivers
    auto init_pd = [&](dnnl_engine_t engine, const prb_t *prb,
                           dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(init_pd_custom(engine, prb, cpd, res), WARN);
        return OK;
    };

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

    alg_t alg = prb->alg;
    if (alg == AUTO) {
        dnnl_convolution_desc_t *temp_conv_desc = {nullptr};
        DNN_SAFE(dnnl_primitive_desc_query(const_pd, dnnl_query_convolution_d,
                         0, &temp_conv_desc),
                CRIT);
        alg = alg_kind2alg(temp_conv_desc->alg_kind);
    }
    const auto cfg = auto_cfg(alg, prb->cfg);
    prb_t p_new((desc_t)*prb, prb->dir, cfg, prb->stag, prb->wtag, prb->dtag,
            alg, prb->attr, prb->mb);
    prb = &p_new;

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

    // Try to use CPU primitive as the reference in GPU testing to reduce
    // testing time
    dnnl_primitive_t c_ref {};

    if (bench_mode & CORR && engine_tgt_kind == dnnl_gpu && fast_ref_gpu
            && // TODO: temporary disable cpu as ref for testcases with binary post-ops
            prb->attr.post_ops.binary_index() == -1) {
        dnnl_primitive_desc_t cpd_ref = nullptr;
        SAFE(init_pd_custom(get_cpu_engine(), prb, cpd_ref, nullptr, fp, fp, fp,
                     fp, fp, src_tag, wei_tag, tag::x, src_tag),
                WARN);
        if (cpd_ref) {
            DNN_SAFE(dnnl_primitive_create(&c_ref, cpd_ref), WARN);
            BENCHDNN_PRINT(
                    5, "%s\n", "benchdnn: use CPU primitive as the reference");
            DNN_SAFE(dnnl_primitive_desc_destroy(cpd_ref), CRIT);
        }
    }

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    dnn_mem_t scales;
    dnn_mem_t src_zero_points_m;
    dnn_mem_t dst_zero_points_m;
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    maybe_prepare_runtime_scales(scales, prb->attr, prb->oc, prb->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, prb->ic, prb->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, prb->oc, prb->dst_zp);

    args_t args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(
                    prb, c_ref, src_fp, wei_fp, bia_fp, binary_po_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, src_tag, test_engine);
            SAFE(compare_dst(prb, dst, dst_fp, res, true), WARN);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(prb, c_ref, src_fp, wei_fp, bia_fp,
                    std::vector<dnn_mem_t>(), dst_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_src(prb, src, src_fp, res, true), WARN);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_w(prb, c_ref, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            SAFE(compare_wei(prb, wei, wei_fp, res, true), WARN);
            if (prb->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, tag::x, test_engine);
                SAFE(compare_bia(prb, bia, bia_fp, res, true), WARN);
            }
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(res->timer, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));
    DNN_SAFE_V(dnnl_primitive_destroy(c_ref));

    return OK;
}

} // namespace conv
