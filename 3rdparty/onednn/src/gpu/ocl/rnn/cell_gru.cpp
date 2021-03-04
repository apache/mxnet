/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
#define PART_ONE 1
#define PART_TWO 2

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution_gru)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    data_type_t src_t = this->pd()->src_type;

    cl_ulong cell_scratch_offset, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_wei_iter_offset, cell_ws_iter_offset2, cell_wei_iter_offset2,
            cell_scratch_offset2;

    set_offsets_fwd_gemm(rnn, iter, dir, lay, src_t, wei_iter_offset_ptr,
            ws_states_offset_, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_scratch_offset, cell_wei_iter_offset);

    cell_scratch_offset2 = cell_scratch_offset;

    set_gru_offsets_part2(rnn, iter, dir, lay, src_t, wei_iter_offset_ptr,
            ws_states_offset_, cell_wei_iter_offset2, cell_scratch_offset2,
            cell_ws_iter_offset2);

    if (aprop == prop_kind::forward) {
        // 1. gemm Wx[0-2],x
        if (!rnn.merge_gemm_layer)
            gemm_primitive(engine, ctx, wei_layer, wei_layer_offset[0],
                    workspace, cell_ws_lay_offset, scratch_gates,
                    cell_scratch_offset, gemm_layer_fwd);

        // 2. gemm Wh[0-1],h
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset, workspace,
                cell_ws_iter_offset, scratch_gates, cell_scratch_offset,
                gemm_iter_fwd);

        // 3. activation zt and rt + elemwise multiplication rt,ht-1
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_ONE);

        // 4. gemm Wh[2],h~t
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset2, workspace,
                cell_ws_iter_offset2, scratch_gates, cell_scratch_offset2,
                gemm_iter_fwd_2);

        // 5. activation h~t + calculate ht
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_TWO);
    } else {
        cl_ulong cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_diff_ws_iter_off, cell_diff_ws_lay_off,
                cell_diff_wei_iter_off2;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, ws_diff_states_offset_,
                cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_diff_ws_lay_off, cell_diff_ws_iter_off,
                cell_diff_wei_iter_off2);

        // 1. calculate dG2, dG1, and part of dht-1
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_ONE);

        // 2. calculate intermediate d(hG1)
        // d(hG1) = dG2 * W2h^t
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset2,
                scratch_gates, cell_scratch_offset2, workspace, ws_dhG1_offset_,
                gemm_iter_bwd_2);

        // 3. calculate dG1^ and part of dht-1
        // hg1 needs to be bf16 as it is used as gemm output
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_TWO);

        // 4. calculate diff weights
        // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
        gemm_primitive(engine, ctx, scratch_gates, cell_scratch_offset,
                workspace, cell_ws_iter_offset, diff_weights_iter,
                cell_diff_wei_iter_off, gemm_diff_wei_iter);

        gemm_primitive(engine, ctx, scratch_gates, cell_scratch_offset2,
                scratch_cell, 0, diff_weights_iter, cell_diff_wei_iter_off2,
                gemm_diff_wei_iter_2);

        // 5. calculate diff states
        // dht-1 += dG1 * W1h + dG0 * W0h
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                scratch_gates, cell_scratch_offset, workspace,
                cell_diff_ws_iter_off, gemm_iter_bwd);

        if (!rnn.merge_gemm_layer) {
            // dWx += [dG0 dG1 dG2] * [x]
            gemm_primitive(engine, ctx, scratch_gates, cell_scratch_offset,
                    workspace, cell_ws_lay_offset, diff_weights_layer,
                    cell_diff_wei_lay_off, gemm_diff_wei_layer);

            // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
            gemm_primitive(engine, ctx, wei_layer, wei_layer_offset[0],
                    scratch_gates, cell_scratch_offset, workspace,
                    cell_diff_ws_lay_off, gemm_layer_bwd);
        }

        // 6. calculate diff bias
        gates_reduction(ctx, dir, lay, iter, rnn.n_gates, rnn.dhc, rnn.mb,
                scratch_gates, scratch_cell, diff_bias);
    }
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
