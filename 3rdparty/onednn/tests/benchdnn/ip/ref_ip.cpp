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

#include "ip/ip.hpp"

namespace ip {

void compute_ref_fwd(const engine_t &engine_tgt, const prb_t *prb,
        dnn_mem_t &src_m, dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m) {

    int64_t M = prb->mb;
    int64_t N = prb->oc;
    int64_t K = prb->ic * prb->id * prb->ih * prb->iw;

    dnn_mem_t dst_tmp(dst_m.md_, dnnl_f32, tag::abx, engine_tgt);

    gemm("C", "N", "T", M, N, K, 1.f, (float *)src_m, K, (float *)wei_m, K, 0.f,
            (float *)dst_tmp, N);

    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();
    dnnl::impl::parallel_nd(prb->mb, prb->oc, [&](int64_t mb, int64_t oc) {
        size_t dst_off = dst_off_f(prb, mb, oc);
        float &dst = ((float *)dst_m)[dst_off];

        float d = ((float *)dst_tmp)[dst_off];
        if (prb->dir & FLAG_BIA) {
            size_t bia_off = bia_off_f(prb, oc);
            d += ((float *)bia_m)[bia_off];
        }
        maybe_oscale(prb->attr, d, prb->scales, oc);

        std::vector<float> v_binary_vals;
        v_binary_vals.reserve(v_bin_po_mask.size());
        for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
            auto bin_po_offset = dst_m.get_scale_idx(dst_off, v_bin_po_mask[d]);
            float binary_val = binary_po[d].get_elem(bin_po_offset);
            v_binary_vals.push_back(binary_val);
        }
        maybe_post_ops(prb->attr, d, dst, v_binary_vals);

        dst = d;
    });
}

void compute_ref_bwd_d(const prb_t *prb, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &diff_dst_m) {

    int64_t M = prb->mb;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->oc;

    gemm("C", "N", "N", M, N, K, 1.f, (float *)diff_dst_m, K, (float *)wei_m, N,
            0.f, (float *)diff_src_m, N);
}

void compute_ref_bwd_w(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m) {

    int64_t M = prb->oc;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->mb;

    gemm("C", "T", "N", M, N, K, 1.f, (float *)diff_dst_m, M, (float *)src_m, N,
            0.f, (float *)diff_wei_m, N);

    if (!(prb->dir & FLAG_BIA)) return;

    dnnl::impl::parallel_nd(prb->oc, [&](int64_t oc) {
        size_t bia_off = bia_off_f(prb, oc);
        float &db = ((float *)diff_bia_m)[bia_off];
        db = 0;
        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            size_t dst_off = dst_off_f(prb, mb, oc);
            db += ((float *)diff_dst_m)[dst_off];
        }
    });
}

} // namespace ip
