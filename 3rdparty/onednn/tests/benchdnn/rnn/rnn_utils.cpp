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

#include "rnn/rnn_aux.hpp"
#include "tests/test_thread.hpp"

namespace rnn {

void copy(int64_t dimc, int64_t dimr, int64_t ld_src, int64_t ld_dst,
        const float *src_, float *dst_, rnn_action_t action,
        bool saturate_to_u8) {
    AOC<const float> src(src_, dimc, ld_src);
    AOC<float> dst(dst_, dimc, ld_dst);

    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++) {
            dst(i, j) = (action == action_sum ? dst(i, j) : 0) + src(i, j);
            if (saturate_to_u8)
                dst(i, j) = saturate_and_round<dnnl_u8>(dst(i, j));
        }
    });
}

void data_q10n(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_,
        float data_scale, float data_shift) {
    AOC<float> src(src_, dimc, ld_src);
    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++)
            src(i, j) = saturate_and_round<dnnl_u8>(
                    data_scale * src(i, j) + data_shift);
    });
}

void data_deq10n(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_,
        float data_scale, float data_shift) {
    AOC<float> src(src_, dimc, ld_src);
    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++)
            src(i, j) = (src(i, j) - data_shift) / data_scale;
    });
}

} // namespace rnn
