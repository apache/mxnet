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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

static int prepare_fwd_with_stats(const prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &ss) {
    dnnl::impl::parallel_nd(prb->ic, prb->mb, prb->id, prb->ih, prb->iw,
            [&](int64_t c, int64_t mb, int64_t d, int64_t h, int64_t w) {
                int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
                float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;
                const int64_t value = (l % 65) - 32;
                s[sp] = round_to_nearest_representable(prb->dt, value);

                ((float *)mean)[c] = 4 * ((c % 5) - 2);
                ((float *)var)[c] = ((c % 7) << 1);

                if (prb->flags & USE_SCALESHIFT) {
                    ((float *)ss)[c] = (1 << (c % 7));
                    ((float *)ss)[prb->ic + c]
                            = ((c % 3) - 1) * ((float *)ss)[c];
                } else {
                    ((float *)ss)[c] = 1;
                    ((float *)ss)[prb->ic + c] = 0;
                }
            });

    return OK;
}

static int prepare_fwd_no_stats(const prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &ss) {
    /** Idea: choose src[] values so that both mean and variance are computed
     * exactly (independently of the order of the computations).
     *
     * The `exactness` is achieved via [a1]: src[i] + src[i+1] = 2 * mean.
     *
     * The variation in src is allowed in the last flex_bits bits.
     * If the sequence (L) is too big (flex_bits <= min_flex_bits), the mean
     * value is set to 0 and src is partially filled with zeros (according to
     * density so that at least want_flex_bits is reserved for src variation.
     * Once src is set, variance is computed.
     *
     * ALG_0: mean is set to 0
     * ALG_1: mean is set to 2^prb, where prb \in {-2, -1, ..., 4}
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically */
    const int64_t exact_bits = digits_dt(prb->dt);
    const int64_t L = prb->mb * prb->id * prb->ih * prb->iw;
    const int64_t logL = (int64_t)ceilf(log2f(L));

    assert(logL <= 0 || (1LL << (logL - 1)) < L);
    assert(L <= (1LL << logL));

    const int64_t min_flex_bits = 3;
    const int64_t want_flex_bits = MIN2(6, exact_bits / 2);

    check_alg_t alg = prb->check_alg;
    if (alg == ALG_AUTO) /* choose appropriate checking algorithm */
        alg = (exact_bits - logL) / 2 - 1 >= min_flex_bits ? ALG_1 : ALG_0;

    const int64_t flex_bits = alg == ALG_0
            ? want_flex_bits /* BFloat16 has only 7 bits of mantissa */
            : MIN2(prb->dt == dnnl_bf16 ? 7 : exact_bits,
                    (exact_bits - logL) / 2 - 1);

    if (flex_bits < min_flex_bits) return FAIL;

    const int64_t flex_mask = (1 << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ALG_0
            ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
            : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    BENCHDNN_PRINT(6, "check_alg: %s, density = %g, flex_bits = " IFMT "\n",
            check_alg2str(alg), density, flex_bits);

    dnnl::impl::parallel_nd(prb->ic, [&](int64_t c) {
        const float m = ((float *)mean)[c]
                = alg == ALG_0 ? 0.f : 0.25f * (1 << (c % 7));
        float v = 0; /* current variance */

        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            int64_t l_base = mb * prb->id * prb->ih * prb->iw
                    + c * 239 * 2; // l[0] must be even
            float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;

                if (alg == ALG_0 && !flip_coin(l / 2 * 257ULL, density)) {
                    s[sp] = 0;
                    continue;
                }

                const int64_t gen = (l / 2 * 1637) & flex_mask;
                const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
                const float f = 1.f * sgn * gen / (1 << flex_bits);

                s[sp] = alg == ALG_0 ? f : m * (1.f + f);
                if (L % 2 && (mb * prb->id * prb->ih * prb->iw + sp == L - 1)) {
                    s[sp] = m;
                }
                v += (s[sp] - m) * (s[sp] - m);
            }
        }

        ((float *)var)[c] = v / (prb->mb * prb->id * prb->ih * prb->iw);

        if (prb->flags & USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 8 * (1 << (c % 7));
            ((float *)ss)[prb->ic + c] = ((c % 3) - 1) * ((float *)ss)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)ss)[prb->ic + c] = 0;
        }
    });

    return OK;
}

static int prepare_fwd(const prb_t *prb, dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, dnn_mem_t &ss) {
    if (prb->flags & GLOB_STATS)
        return prepare_fwd_with_stats(prb, src, mean, var, ss);
    else
        return prepare_fwd_no_stats(prb, src, mean, var, ss);
}

static int prepare_bwd(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Idea behind filling: integer diff_dst values decrease norms unlike fp32
    // values in [-1.f, 1.f] range. To decrease norms more, make data pretty
    // sparse as answers sum all diff_dst values.

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> igen_val(-2, 2);
        std::uniform_int_distribution<> igen_coin(0, 256 * 1024);

        // at least 20 non-zero elems
        float sparsity = MAX2(0.05f, MIN2(1.f, 20.f / nelems));

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = flip_coin(igen_coin(msr), sparsity)
                    ? round_to_nearest_representable(prb->dt, igen_val(msr))
                    : 0;
            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

static int compare(const prb_t *prb, data_kind_t kind, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *res, const dnn_mem_t *ss = nullptr) {
    const char *skind = data_kind2str(kind);

    const int f32_mant_digits = 24;
    const float eps_coeff = (1 << (f32_mant_digits - digits_dt(prb->dt)));
    float eps = eps_coeff * (kind == DATA ? 5e-7 : 0);
    if (kind == SS && prb->dir & FLAG_BWD) eps = eps_coeff * 5e-6;

    // Since bwd testing is done using results from forward which are random
    // fp32 values, diff_ss starts fluctuating, so we check norm for both data
    // and SS.
    const bool rely_on_norm = prb->dir & FLAG_BWD;

    const int64_t N = kind == DATA ? prb->mb : 1;
    const int64_t C = kind == DATA ? prb->ic : prb->ic * (kind == SS ? 2 : 1);
    const int64_t SP = kind == DATA ? prb->id * prb->ih * prb->iw : 1;

    const auto nelems = N * C * SP;
    if (nelems == 0) return res->state = PASSED, OK;

    res->total += rely_on_norm ? 1 : nelems;

    diff_norm_t diff_norm;
    for_(int64_t n = 0; n < N; n++)
    for_(int64_t c = 0; c < C; c++)
    for (int64_t sp = 0; sp < SP; ++sp) {
        int64_t i = (n * C + c) * SP + sp;
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = kind == DATA
                ? round_to_nearest_representable(prb->dt, fp0)
                : fp0;
        diff_norm.update(fp, dt);

        if (rely_on_norm) continue;

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= eps;

        /* When the error is larger than eps, It could be
         * due to catastrophic cancellation in final result
         * which is computed as `Y = a * X + b`.
         * When `a * X`  is close to `b` and `sign(a * X) = - sign(b)`.
         * Then large error in `a * X` could result in a final
         * result (which has a cancellation i.e. `|Y| = |a*X - (-b)|`)
         * which has no meaningful digits left in mantissa.*/
        if (!ok && (prb->dir & FLAG_FWD) && kind == DATA && ss) {
            const float beta = ((float *)*ss)[prb->ic + c];
            /* Using an empirically derived threshold,
             * check if cancellation error
             * in `|Y| = |a*X - (-b)|` is huge.*/
            bool maybe_cancellation_error
                    = (fabsf(fp - beta) / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1))
                    > 1.0f;
            if (maybe_cancellation_error) {
                /* Check for error in `a * X` */
                float diff_aX = fabsf((fp - beta) - (dt - beta));
                float rel_diff_aX = diff_aX
                        / (fabsf(fp - beta) > FLT_MIN ? fabsf(fp - beta) : 1);
                ok = rel_diff_aX <= eps;
            }
        }

        res->errors += !ok;

        bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            if (kind == DATA) {
                int64_t mb, c, d, h, w;
                inv_data_off(prb, i, mb, c, d, h, w);
                ss << mb << "," << c << "," << d << "," << h << "," << w;
            } else if (kind == SS) {
                ss << i / prb->ic << "," << i % prb->ic;
            } else {
                ss << i;
            }

            std::string ind_str = ss.str();
            BENCHDNN_PRINT(0,
                    "[%4ld][%s%s][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, prb->dir & FLAG_BWD ? "D_" : "", skind,
                    ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    diff_norm.done();

    if (rely_on_norm) {
        const bool ok = diff_norm.rel_diff(norm_t::L1) <= eps
                && diff_norm.rel_diff(norm_t::L2) <= eps
                && diff_norm.rel_diff(norm_t::L8) <= eps;
        res->errors += !ok;
    }

    if (res->errors || verbose >= 5) {
        const int vl = res->errors ? 0 : 2;
        BENCHDNN_PRINT(vl,
                "@@@ [%s%s] diff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                prb->dir & FLAG_BWD ? "D_" : "", skind,
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int check_fwd_ws(const dnn_mem_t &dst_dt, const dnn_mem_t &ws_dt, res_t *res) {
    if (ws_dt.md_.ndims == 0) return OK;

    /* so far we know ws is just bit-mask of whether value was negative or
     * positive */
    const auto nelems = dst_dt.nelems(true);
    const uint8_t *ws = (const uint8_t *)ws_dt;

    /* some internal knowledge: flags in ws are either stored as bytes (e.g.
     * for the ref implementation) or as bits (e.g. for the jitted one); in
     * the latter case the ws memory has fewer elements than the data memory */
    enum { ws_byte, ws_bit } ws_type;
    ws_type = ws_dt.nelems(true) < nelems ? ws_bit : ws_byte;

    /* more internal knowledge: dst_dt and ws_dt are expected to have exactly
     * the same data layout, and dst_dt padded regions are expected to be
     * zero, and the respective ws_dt elements should be set accordingly */
    for (int64_t i = 0; i < nelems; i += 8) {
        for (int64_t j = 0; j < MIN2(8, nelems - i); ++j) {
            const float data = dst_dt.get_elem(i + j);
            const bool want = data > 0.f;
            const bool bit_set = ws_type == ws_byte ? *ws : !!(*ws & (1 << j));

            const bool ok = bit_set == want;
            res->errors += !ok;

            bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                    || (verbose >= 50 && i < 30);
            if (dump) {
                BENCHDNN_PRINT(0, "[%4ld] ws exp:%d got:%d (data:%g:%a)\n",
                        (long)(i + j), want, bit_set, data, data);
            }

            // XXX: GPU implementation uses int32_t for workspace
            if (engine_tgt_kind == dnnl_gpu) {
                ws += sizeof(int32_t);
            } else {
                if (ws_type == ws_byte) ++ws;
            }
        }
        if (ws_type == ws_bit) ++ws;
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int init_pd(dnnl_engine_t engine, const prb_t *prb, dnnl_primitive_desc_t &bpd,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint) {
    dnnl_batch_normalization_desc_t bd;
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

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE(dnnl_batch_normalization_forward_desc_init(
                         &bd, prop, &data_d, prb->eps, flags),
                WARN);

    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, prb->ndims,
                         data_dims, prb->dt, dnnl_format_tag_any),
                WARN);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        DNN_SAFE(dnnl_batch_normalization_backward_desc_init(
                         &bd, prop, &diff_data_d, &data_d, prb->eps, flags),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&bpd, &bd, dnnl_attr, engine, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (prb->dir & FLAG_FWD)) return OK;

    res->impl_name = query_impl_info(bpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(bpd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
        if (!strstr(res->impl_name.c_str(), "jit")) {
            BENCHDNN_PRINT(2, "WARNING: %s",
                    "accuracy of the implementation being tested "
                    "depends on the compiler and might give "
                    "false-positives.\n");
            BENCHDNN_PRINT(2, "         %s",
                    "please consider recompiling the sources with"
                    " `-prec-div -fp-model precise` for a reliable testing.\n");
        }
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

    dnnl_primitive_t b {};
    SAFE(init_prim(&b, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(b, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(b));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(const_fpd, DNNL_ARG_SRC);
    const auto &mean_md = q(const_fpd, DNNL_ARG_MEAN);
    const auto &var_md = q(const_fpd, DNNL_ARG_VARIANCE);
    const auto &ss_md = q(const_fpd, DNNL_ARG_SCALE_SHIFT);
    const auto &ws_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, fp, tag, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);
    // stash for bwd: src_hat[i] = (src[i] - mean) / sqrt(var + prb->eps)
    dnn_mem_t src_hat_fp(data_md, fp, tag, test_engine);

    dnn_mem_t &dst_fp = src_fp; // in-place in ref code
    dnn_mem_t placeholder_dst_dt;
    const bool inplace_fwd = prb->inplace && (prb->dir & FLAG_FWD);
    if (!inplace_fwd) { placeholder_dst_dt = dnn_mem_t(data_md, test_engine); }
    dnn_mem_t &dst_dt = inplace_fwd ? src_dt : placeholder_dst_dt;

    // On inference w/o global stats the batch norm doesn't require stat
    // memories. Hence, we need to prepare the mean_fp and var_fp ourselves.
    const dnnl_dims_t dims1d = {prb->ic};
    dnn_mem_t mean_fp(1, dims1d, fp, tag::abx, test_engine);
    dnn_mem_t mean_dt(mean_md, test_engine);
    dnn_mem_t var_fp(1, dims1d, fp, tag::abx, test_engine);
    dnn_mem_t var_dt(var_md, test_engine);

    dnn_mem_t ss_fp(ss_md, fp, tag::abx, test_engine);
    dnn_mem_t ss_dt(ss_md, test_engine);
    dnn_mem_t d_ss_fp(ss_md, fp, tag::abx, test_engine);
    dnn_mem_t d_ss_dt(ss_md, test_engine);

    if (prb->need_ws()) SAFE(ws_md.ndims != 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(data_md, dnnl_u8, tag, test_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    if (prepare_fwd(prb, src_fp, mean_fp, var_fp, ss_fp) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(b));
        return res->state = MISTRUSTED, OK;
    }

    SAFE(src_dt.reorder(src_fp), WARN);
    if (prb->flags & GLOB_STATS) {
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
    }
    if (prb->flags & USE_SCALESHIFT) { SAFE(ss_dt.reorder(ss_fp), WARN); }

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_MEAN, mean_dt);
    args.set(DNNL_ARG_VARIANCE, var_dt);
    args.set(DNNL_ARG_SCALE_SHIFT, ss_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE(execute_and_wait(b, args), WARN);

    // Running ref to collect src_hat (used instead of src + mean) and ws, if
    // fuse_relu flag is requested.
    if (bench_mode & CORR) {
        compute_ref_fwd(
                prb, src_fp, mean_fp, var_fp, ss_fp, ws_fp, dst_fp, src_hat_fp);
        if (prb->dir & FLAG_FWD) {
            if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                SAFE(compare(prb, MEAN, mean_fp, mean_dt, res), WARN);
                SAFE(compare(prb, VAR, var_fp, var_dt, res), WARN);
            }
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare(prb, DATA, dst_fp, dst, res, &ss_fp), WARN);
            if (prb->debug_check_ws)
                SAFE(check_fwd_ws(dst_dt, ws_dt, res), WARN);
        }
    }

    if (prb->dir & FLAG_BWD) {
        dnnl_primitive_t bwd_p {};
        int status = init_prim(&bwd_p, init_pd, prb, res, FLAG_BWD, const_fpd);
        dnnl_primitive_destroy(b);
        if (status != OK) return status;
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        b = bwd_p;

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(b, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(b));
            return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
        }

        const auto &d_data_md = q(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        if (!prb->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, test_engine);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(prepare_bwd(prb, d_dst_dt, d_dst_fp), WARN);

        args.clear();
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(DNNL_ARG_SCALE_SHIFT, ss_dt);
        args.set(DNNL_ARG_DIFF_SCALE_SHIFT, d_ss_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(b, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(prb, src_hat_fp, var_fp, d_dst_fp, ss_fp, ws_fp,
                    d_src_fp, d_ss_fp);
            if ((prb->flags & USE_SCALESHIFT) && (prb->dir & FLAG_WEI)) {
                SAFE(compare(prb, SS, d_ss_fp, d_ss_dt, res), WARN);
            }
            dnn_mem_t d_src(d_src_dt, fp, tag, test_engine);
            SAFE(compare(prb, DATA, d_src_fp, d_src, res), WARN);
        }
    }
    measure_perf(res->timer, b, args);

    DNN_SAFE_V(dnnl_primitive_destroy(b));

    return OK;
}

} // namespace bnorm
