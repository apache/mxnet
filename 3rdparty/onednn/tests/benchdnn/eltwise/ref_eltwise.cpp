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

#include "eltwise/eltwise.hpp"

namespace eltwise {

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst) {
    const float *src_ptr = (const float *)src;
    float *dst_ptr = (float *)dst;
    const auto nelems = src.nelems();
    std::vector<int> v_bin_po_mask = prb->attr.post_ops.get_binary_po_masks();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        float res = compute_eltwise_fwd(
                prb->alg, src_ptr[i], 1.0, prb->alpha, prb->beta);
        std::vector<float> v_binary_vals;
        v_binary_vals.reserve(v_bin_po_mask.size());
        for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
            auto bin_po_offset = src.get_scale_idx(i, v_bin_po_mask[d]);
            float binary_val = binary_po[d].get_elem(bin_po_offset);
            v_binary_vals.push_back(binary_val);
        }
        maybe_post_ops(prb->attr, res, 0.f, v_binary_vals);
        dst_ptr[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src) {
    const float *src_ptr = (const float *)src;
    const float *d_dst_ptr = (const float *)diff_dst;
    float *d_src_ptr = (float *)diff_src;
    const auto nelems = src.nelems();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        d_src_ptr[i] = compute_eltwise_bwd(
                prb->alg, d_dst_ptr[i], src_ptr[i], prb->alpha, prb->beta);
    });
}

} // namespace eltwise
