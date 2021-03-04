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

#include "tests/test_thread.hpp"

#include "conv/conv_common.hpp"

namespace conv {

template <typename get_args_func>
void exec_conv(get_args_func get_args, const prb_t *prb, dnnl_primitive_t c_ref,
        dnn_mem_t &src_m, dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        dnn_mem_t &dst_m) {
    const_dnnl_primitive_desc_t pd_ref;
    dnnl_engine_t engine_ref;

    SAFE_V(dnnl_primitive_get_primitive_desc(c_ref, &pd_ref));
    SAFE_V(dnnl_primitive_desc_query(
            pd_ref, dnnl_query_engine, 0, &engine_ref));
    const auto &scratchpad_md = *dnnl_primitive_desc_query_md(
            pd_ref, dnnl_query_exec_arg_md, DNNL_ARG_SCRATCHPAD);

    auto src_ref = dnn_mem_t::create_from_host_ptr(
            src_m.md_, engine_ref, (void *)src_m);
    auto wei_ref = dnn_mem_t::create_from_host_ptr(
            wei_m.md_, engine_ref, (void *)wei_m);
    dnn_mem_t bia_ref;
    if (prb->dir & FLAG_BIA)
        bia_ref = dnn_mem_t::create_from_host_ptr(
                bia_m.md_, engine_ref, (void *)bia_m);
    auto dst_ref = dnn_mem_t::create_from_host_ptr(
            dst_m.md_, engine_ref, (void *)dst_m);
    dnn_mem_t scratchpad(scratchpad_md, engine_ref);

    args_t args = get_args(prb, src_ref, wei_ref, bia_ref, dst_ref);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad);
    SAFE_V(execute_and_wait(c_ref, args));
}

args_t get_args_conv_fwd(const prb_t *prb, dnn_mem_t &src_ref,
        dnn_mem_t &wei_ref, dnn_mem_t &bia_ref, dnn_mem_t &dst_ref) {

    args_t args;
    args.set(DNNL_ARG_SRC, src_ref);
    args.set(DNNL_ARG_WEIGHTS, wei_ref);
    if (prb->dir & FLAG_BIA) args.set(DNNL_ARG_BIAS, bia_ref);
    args.set(DNNL_ARG_DST, dst_ref);
    return args;
}

args_t get_args_conv_bwd_d(const prb_t *prb, dnn_mem_t &src_ref,
        dnn_mem_t &wei_ref, dnn_mem_t &bia_ref, dnn_mem_t &dst_ref) {

    args_t args;
    args.set(DNNL_ARG_DIFF_SRC, src_ref);
    args.set(DNNL_ARG_WEIGHTS, wei_ref);
    args.set(DNNL_ARG_DIFF_DST, dst_ref);
    return args;
}

args_t get_args_conv_bwd_w(const prb_t *prb, dnn_mem_t &src_ref,
        dnn_mem_t &wei_ref, dnn_mem_t &bia_ref, dnn_mem_t &dst_ref) {

    args_t args;
    args.set(DNNL_ARG_SRC, src_ref);
    args.set(DNNL_ARG_DIFF_WEIGHTS, wei_ref);
    if (prb->dir & FLAG_BIA) args.set(DNNL_ARG_DIFF_BIAS, bia_ref);
    args.set(DNNL_ARG_DIFF_DST, dst_ref);
    return args;
}

void compute_ref_fwd(const prb_t *prb, dnnl_primitive_t c_ref, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m) {
    if (c_ref) {
        exec_conv(get_args_conv_fwd, prb, c_ref, src_m, wei_m, bia_m, dst_m);
        return;
    }
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_fwd(prb, src_m, wei_m, bia_m, dst_m);
    } else {
        compute_ref_direct_fwd(prb, src_m, wei_m, bia_m, binary_po, dst_m);
    }
}

void compute_ref_bwd_d(const prb_t *prb, dnnl_primitive_t c_ref,
        dnn_mem_t &diff_src_m, dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &diff_dst_m) {
    if (c_ref) {
        exec_conv(get_args_conv_bwd_d, prb, c_ref, diff_src_m, wei_m, bia_m,
                diff_dst_m);
        return;
    }
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_bwd_d(prb, diff_src_m, wei_m, bia_m, diff_dst_m);
    } else {
        compute_ref_direct_bwd_d(
                prb, diff_src_m, wei_m, bia_m, binary_po, diff_dst_m);
    }
}

void compute_ref_bwd_w(const prb_t *prb, dnnl_primitive_t c_ref,
        dnn_mem_t &src_m, dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m,
        dnn_mem_t &diff_dst_m) {
    if (c_ref) {
        exec_conv(get_args_conv_bwd_w, prb, c_ref, src_m, diff_wei_m,
                diff_bia_m, diff_dst_m);
        return;
    }
    if (prb->alg == WINO && prb->cfg[SRC].dt == dnnl_f32) {
        compute_wino_ref_bwd_w(prb, src_m, diff_wei_m, diff_bia_m, diff_dst_m);
    } else {
        compute_ref_direct_bwd_w(
                prb, src_m, diff_wei_m, diff_bia_m, diff_dst_m);
    }
}

void compute_ref_direct_fwd(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m) {
    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    auto ker = [&](float &d, int64_t g, int64_t mb, int64_t oc, int64_t od,
                       int64_t oh, int64_t ow) {
        const float *__restrict src_loc
                = (const float *)src_m + (mb * IC + g * ICG) * ID * IH * IW;
        const float *__restrict wei_loc
                = (const float *)wei_m + (g * OCG + oc) * ICG * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd * DD;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh * DH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw * DW;
                    if (iw < 0 || iw >= IW) continue;

                    for (int64_t ic = 0; ic < ICG; ++ic) {
                        int64_t src_off = ((ic * ID + id) * IH + ih) * IW + iw;
                        int64_t wei_off = ((ic * KD + kd) * KH + kh) * KW + kw;
                        float s = src_loc[src_off];
                        maybe_zero_point(prb->attr, s, prb->src_zp,
                                g * ICG + ic, DNNL_ARG_SRC);
                        d += s * wei_loc[wei_off];
                    }
                }
            }
        }
    };

    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();
    dnnl::impl::parallel_nd(G, MB, OCG, OD, OH, OW,
            [&](int64_t g, int64_t mb, int64_t oc, int64_t od, int64_t oh,
                    int64_t ow) {
                const size_t dst_off = dst_off_f(prb, mb, g, oc, od, oh, ow);
                float &dst = ((float *)dst_m)[dst_off];

                float conv_res = 0;
                ker(conv_res, g, mb, oc, od, oh, ow);

                if (prb->dir & FLAG_BIA) {
                    const size_t bia_off = bia_off_f(prb, g, oc);
                    conv_res += ((float *)bia_m)[bia_off];
                }

                maybe_oscale(prb->attr, conv_res, prb->scales, g * OCG + oc);

                std::vector<float> v_binary_vals;
                v_binary_vals.reserve(v_bin_po_mask.size());
                for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
                    auto bin_po_offset
                            = dst_m.get_scale_idx(dst_off, v_bin_po_mask[d]);
                    float binary_val = binary_po[d].get_elem(bin_po_offset);
                    v_binary_vals.push_back(binary_val);
                }
                maybe_post_ops(prb->attr, conv_res, dst, v_binary_vals);

                maybe_zero_point(prb->attr, conv_res, prb->dst_zp, g * OCG + oc,
                        DNNL_ARG_DST, true);

                dst = conv_res;
            });
}

void compute_ref_direct_bwd_d(const prb_t *prb, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &diff_dst_m) {

    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    enum { precompute_size = 16 };
    const bool fast = MAX3(KD, KH, KW) <= precompute_size;

    /* pre-computes arrays of oh(ow) and kh(kw) for traversing in kernel */
    auto precompute_ok
            = [](int64_t i, int64_t O, int64_t K, int64_t S, int64_t P,
                      int64_t D, int64_t &num, int64_t *_o, int64_t *_k) {
                  assert(K <= precompute_size);
                  num = 0;
                  for (int64_t k = 0; k < K; ++k) {
                      int64_t o = i - k * D + P;
                      if (o < 0 || o % S) continue;
                      o /= S;
                      if (o >= O) continue;
                      _k[num] = k;
                      _o[num] = o;
                      ++num;
                  }
              };

    auto ker_fast = [&](float &ds, int64_t g, int64_t mb, int64_t ic,
                            int64_t id, int64_t ih, int64_t iw) {
        int64_t kd[precompute_size], od[precompute_size], num_d;
        int64_t kh[precompute_size], oh[precompute_size], num_h;
        int64_t kw[precompute_size], ow[precompute_size], num_w;
        precompute_ok(id, OD, KD, SD, PD, DD, num_d, od, kd);
        precompute_ok(ih, OH, KH, SH, PH, DH, num_h, oh, kh);
        precompute_ok(iw, OW, KW, SW, PW, DW, num_w, ow, kw);

        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for_(int64_t d = 0; d < num_d; ++d)
        for_(int64_t h = 0; h < num_h; ++h)
        for (int64_t w = 0; w < num_w; ++w) {
            for (int64_t oc = 0; oc < OCG; ++oc) {
                const int64_t diff_dst_off
                        = ((oc * OD + od[d]) * OH + oh[h]) * OW + ow[w];
                const int64_t wei_off
                        = ((oc * ICG * KD + kd[d]) * KH + kh[h]) * KW + kw[w];
                ds += diff_dst_loc[diff_dst_off] * wei_loc[wei_off];
            }
        }
    };

    auto ker = [&](float &ds, int64_t g, int64_t mb, int64_t ic, int64_t id,
                       int64_t ih, int64_t iw) {
        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            int64_t od = id - kd * DD + PD;
            if (od < 0 || od % SD || od >= OD * SD) continue;
            od /= SD;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t oh = ih - kh * DH + PH;
                if (oh < 0 || oh % SH || oh >= OH * SH) continue;
                oh /= SH;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t ow = iw - kw * DW + PW;
                    if (ow < 0 || ow % SW || ow >= OW * SW) continue;
                    ow /= SW;
                    for (int64_t oc = 0; oc < OCG; ++oc) {
                        const int64_t diff_dst_off
                                = ((oc * OD + od) * OH + oh) * OW + ow;
                        const int64_t wei_off
                                = ((oc * ICG * KD + kd) * KH + kh) * KW + kw;
                        ds += diff_dst_loc[diff_dst_off] * wei_loc[wei_off];
                    }
                }
            }
        }
    };

    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();
    dnnl::impl::parallel_nd(G, MB, ICG, ID, IH, IW,
            [&](int64_t g, int64_t mb, int64_t ic, int64_t id, int64_t ih,
                    int64_t iw) {
                size_t src_off = src_off_f(prb, mb, g, ic, id, ih, iw);
                float &ds = ((float *)diff_src_m)[src_off];
                float conv_res = 0;
                if (fast)
                    ker_fast(conv_res, g, mb, ic, id, ih, iw);
                else
                    ker(conv_res, g, mb, ic, id, ih, iw);

                if (prb->dir & FLAG_BIA) {
                    const size_t bia_off = (size_t)g * ICG + ic;
                    conv_res += ((float *)bia_m)[bia_off];
                }
                maybe_oscale(prb->attr, conv_res, prb->scales, g * ICG + ic);

                std::vector<float> v_binary_vals;
                v_binary_vals.reserve(v_bin_po_mask.size());
                for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
                    auto bin_po_offset = diff_src_m.get_scale_idx(
                            src_off, v_bin_po_mask[d]);
                    float binary_val = binary_po[d].get_elem(bin_po_offset);
                    v_binary_vals.push_back(binary_val);
                }
                maybe_post_ops(prb->attr, conv_res, ds, v_binary_vals);

                ds = conv_res;
            });
}

void compute_ref_bwd_weights(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_dst_m) {
    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    auto compute_bounds
            = [](int64_t I, int64_t O, int64_t k, int64_t S, int64_t P,
                      int64_t D, int64_t &o_s, int64_t &o_e) {
                  const float tmp = P - k * D;
                  o_s = MAX2(0, ceilf(tmp / S));
                  o_e = MIN2(O, ceilf((I + tmp) / S));
              };

    auto ker = [&](float &dw, int64_t g, int64_t oc, int64_t ic, int64_t kd,
                       int64_t kh, int64_t kw) {
        int64_t od_s, od_e, oh_s, oh_e, ow_s, ow_e;
        compute_bounds(ID, OD, kd, SD, PD, DD, od_s, od_e);
        compute_bounds(IH, OH, kh, SH, PH, DH, oh_s, oh_e);
        compute_bounds(IW, OW, kw, SW, PW, DW, ow_s, ow_e);
        const int64_t id_s = kd * DD - PD;
        const int64_t ih_s = kh * DH - PH;
        const int64_t iw_s = kw * DW - PW;

        for (int64_t mb = 0; mb < MB; ++mb) {
            const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                    + (mb * OC + g * OCG + oc) * OD * OH * OW;
            const float *__restrict src_loc = (const float *)src_m
                    + (mb * IC + g * ICG + ic) * ID * IH * IW;

            for_(int64_t od = od_s; od < od_e; ++od)
            for_(int64_t oh = oh_s; oh < oh_e; ++oh)
            for (int64_t ow = ow_s; ow < ow_e; ++ow) {
                const int64_t id = od * SD + id_s;
                const int64_t ih = oh * SH + ih_s;
                const int64_t iw = ow * SW + iw_s;

                size_t diff_dst_off = (od * OH + oh) * OW + ow;
                size_t src_off = (id * IH + ih) * IW + iw;
                dw += diff_dst_loc[diff_dst_off] * src_loc[src_off];
            }
        }
    };

    dnnl::impl::parallel_nd(G, OCG, ICG, KD, KH, KW,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                size_t wei_off = wei_off_f(prb, g, oc, ic, kd, kh, kw);
                float &dw = ((float *)diff_wei_m)[wei_off];
                dw = 0;
                ker(dw, g, oc, ic, kd, kh, kw);
            });
}

void compute_ref_bwd_bias(
        const prb_t *prb, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc;
    const int64_t OCG = OC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;

    dnnl::impl::parallel_nd(G, OCG, [&](int64_t g, int64_t oc) {
        size_t bia_off = bia_off_f(prb, g, oc);
        double sum = 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t od = 0; od < OD; ++od)
        for_(int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
            size_t dst_off = dst_off_f(prb, mb, g, oc, od, oh, ow);
            sum += ((float *)diff_dst_m)[dst_off];
        }
        ((float *)diff_bia_m)[bia_off] = (float)sum;
    });
}

void compute_ref_direct_bwd_w(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {
    compute_ref_bwd_weights(prb, src_m, diff_wei_m, diff_dst_m);
    if (!(prb->dir & FLAG_BIA)) return;
    compute_ref_bwd_bias(prb, diff_bia_m, diff_dst_m);
}

} // namespace conv
