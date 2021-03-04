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

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::rnn_elemwise)) {
    auto nd_range = compute::nd_range_t({dhc, batch});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scratch_gates);
    arg_list.set(5, bias);
    arg_list.set(6, pd()->desc()->alpha);
    // for test mode
    arg_list.set(7, tm_scales ? *tm_scales : memory_storage_t::empty_storage());
    arg_list.set(8, pd()->rnn_conf.tm_cscale);
    parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise)) {
    auto nd_range = compute::nd_range_t({dhc, batch});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scratch_gates);
    arg_list.set(5, bias);
    arg_list.set(6, pd()->desc()->alpha);
    // for test mode
    arg_list.set(7, tm_scales ? *tm_scales : memory_storage_t::empty_storage());
    arg_list.set(8, pd()->rnn_conf.tm_cscale);
    parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise_u8s8)) {
    auto nd_range = compute::nd_range_t({dhc, batch});

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scratch_gates);
    arg_list.set(5, scales ? *scales : memory_storage_t::empty_storage());
    arg_list.set(6, bias);
    arg_list.set(7, pd()->desc()->alpha);
    arg_list.set(8, data_shift);
    arg_list.set(9, data_scale);
    // for test mode
    arg_list.set(
            10, tm_scales ? *tm_scales : memory_storage_t::empty_storage());
    arg_list.set(11, pd()->rnn_conf.tm_cscale);
    parallel_for(ctx, nd_range, elemwise_fwd_kernel_, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::gru_lbr_elemwise)) {
    auto nd_range = compute::nd_range_t({dhc, batch});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scratch_gates);
    arg_list.set(5, bias);
    arg_list.set(6, pd()->desc()->alpha);
    // for test mode
    arg_list.set(7, tm_scales ? *tm_scales : memory_storage_t::empty_storage());
    arg_list.set(8, pd()->rnn_conf.tm_cscale);
    arg_list.set(9, scratch_cell);
    parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_bwd_t::gru_lbr_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::gru_elemwise)) {
    auto nd_range = compute::nd_range_t({dhc, batch});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scratch_gates);
    arg_list.set(5, bias);
    arg_list.set(6, pd()->desc()->alpha);
    arg_list.set(7, tm_scales ? *tm_scales : memory_storage_t::empty_storage());
    arg_list.set(8, pd()->rnn_conf.tm_cscale);
    arg_list.set(9, part);
    if (aprop != dnnl_forward) { arg_list.set(10, scratch_cell); }
    parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::gru_elemwise);
template elemwise_sig(ref_rnn_bwd_t::gru_elemwise);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
