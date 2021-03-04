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

#include "tests/test_thread.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void compute_ref(const engine_t &engine_tgt, const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m) {
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;
    const int64_t MB = dst_m.nelems() / (M * N);
    const int batch_ndims = dst_m.md_.ndims - 2;

    const int src_zero_point = prb->attr.zero_points[DNNL_ARG_SRC];
    const int wei_zero_point = prb->attr.zero_points[DNNL_ARG_WEIGHTS];
    const int dst_zero_point = prb->attr.zero_points[DNNL_ARG_DST];

    dnn_mem_t dst_tmp(dst_m.md_, dnnl_f32, tag::undef, engine_tgt);

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();

    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        auto src = (const float *)src_m;
        auto wei = (const float *)wei_m;

        float dst = 0;
        const int64_t src_mb
                = dst_m.get_scale_idx(mb, src_broadcast_mask, batch_ndims);
        const int64_t wei_mb
                = dst_m.get_scale_idx(mb, wei_broadcast_mask, batch_ndims);
        for (int64_t k = 0; k < K; ++k) {
            dst += (src[src_off_f(prb, src_mb, m, k)] - src_zero_point)
                    * (wei[wei_off_f(prb, wei_mb, k, n)] - wei_zero_point);
        }
        ((float *)dst_tmp)[dst_off_f(prb, mb, m, n)] = dst;
    });

    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();
    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(prb, mb, m, n);
        float &dst = ((float *)dst_m)[dst_off];

        float tmp = ((float *)dst_tmp)[dst_off];
        if (prb->bia_dt != dnnl_data_type_undef) {
            int64_t bia_off = dst_m.get_scale_idx(dst_off, bias_broadcast_mask);
            float *bia_ptr = (float *)bia_m;
            tmp += bia_ptr[bia_off];
        }
        maybe_oscale(prb->attr, tmp, prb->scales, n);

        std::vector<float> v_binary_vals;
        v_binary_vals.reserve(v_bin_po_mask.size());
        for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
            auto bin_po_offset = dst_m.get_scale_idx(dst_off, v_bin_po_mask[d]);
            float binary_val = binary_po[d].get_elem(bin_po_offset);
            v_binary_vals.push_back(binary_val);
        }
        maybe_post_ops(prb->attr, tmp, dst, v_binary_vals);

        tmp += dst_zero_point;
        dst = tmp;
    });
}

} // namespace matmul
