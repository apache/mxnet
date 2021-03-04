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

#include "sum/sum.hpp"

namespace sum {

void compute_ref(
        const prb_t *prb, const std::vector<dnn_mem_t> &src, dnn_mem_t &dst) {
    float *dst_ptr = (float *)dst;
    const auto nelems = dst.nelems();

    dnnl::impl::parallel_nd(nelems, [&](int64_t k) {
        dst_ptr[k] = 0;
        for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
            const float *src_ptr = (const float *)src[i_input];
            dst_ptr[k] += (src_ptr[k] * prb->scales[i_input]);
        }
    });
}

} // namespace sum
