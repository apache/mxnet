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

// Common for RNN and LSTM cell execution

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
#define PART 1

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    data_type_t src_t = this->pd()->src_type;

    cl_ulong cell_scratch_offset, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_wei_iter_offset;

    set_offsets_fwd_gemm(rnn, iter, dir, lay, src_t, wei_iter_offset_ptr,
            ws_states_offset_, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_scratch_offset, cell_wei_iter_offset);

    if (aprop == prop_kind::forward) {
        if (!rnn.merge_gemm_layer) {
            gemm_primitive(engine, ctx, wei_layer, wei_layer_offset[0],
                    workspace, cell_ws_lay_offset, scratch_gates,
                    cell_scratch_offset, gemm_layer_fwd);
        }

        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset, workspace,
                cell_ws_iter_offset, scratch_gates, cell_scratch_offset,
                gemm_iter_fwd);

        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART);

    } else { // backward
        cl_ulong cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_diff_ws_iter_off, cell_diff_ws_lay_off;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, ws_diff_states_offset_,
                cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_diff_ws_lay_off, cell_diff_ws_iter_off);

        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART);

        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                scratch_gates, cell_scratch_offset, workspace,
                cell_diff_ws_iter_off, gemm_iter_bwd);

        if (!rnn.merge_gemm_layer) {
            gemm_primitive(engine, ctx, wei_layer, wei_layer_offset[0],
                    scratch_gates, cell_scratch_offset, workspace,
                    cell_diff_ws_lay_off, gemm_layer_bwd);

            gemm_primitive(engine, ctx, scratch_gates, cell_scratch_offset,
                    workspace, cell_ws_lay_offset, diff_weights_layer,
                    cell_diff_wei_lay_off, gemm_diff_wei_layer);
        }

        if (!rnn.merge_gemm_iter) {
            gemm_primitive(engine, ctx, scratch_gates, cell_scratch_offset,
                    workspace, cell_ws_iter_offset, diff_weights_iter,
                    cell_diff_wei_iter_off, gemm_diff_wei_iter);
        }

        gates_reduction(ctx, dir, lay, iter, rnn.n_gates, rnn.dhc, rnn.mb,
                scratch_gates, scratch_cell, diff_bias);
    }
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
