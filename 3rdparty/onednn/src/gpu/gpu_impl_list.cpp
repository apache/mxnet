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

#include "gpu/gpu_impl_list.hpp"

#include "gpu/jit/gemm/gen_gemm.hpp"
#include "gpu/ocl/convolution_inner_product.hpp"
#include "gpu/ocl/gemm/gen12lp_gemm.hpp"
#include "gpu/ocl/gemm/gen9_gemm.hpp"
#include "gpu/ocl/gemm/gen9_gemm_x8x8s32.hpp"
#include "gpu/ocl/gemm/ref_gemm.hpp"
#include "gpu/ocl/gemm_inner_product.hpp"
#include "gpu/ocl/gemm_matmul.hpp"
#include "gpu/ocl/gemm_post_ops_inner_product.hpp"
#include "gpu/ocl/gen12lp_x8s8s32x_1x1_convolution.hpp"
#include "gpu/ocl/gen12lp_x8s8s32x_convolution.hpp"
#include "gpu/ocl/gen9_batch_normalization.hpp"
#include "gpu/ocl/gen9_binary.hpp"
#include "gpu/ocl/gen9_convolution.hpp"
#include "gpu/ocl/gen9_pooling.hpp"
#include "gpu/ocl/gen9_softmax.hpp"
#include "gpu/ocl/gen9_wino_convolution.hpp"
#include "gpu/ocl/ref_batch_normalization.hpp"
#include "gpu/ocl/ref_binary.hpp"
#include "gpu/ocl/ref_convolution.hpp"
#include "gpu/ocl/ref_deconvolution.hpp"
#include "gpu/ocl/ref_eltwise.hpp"
#include "gpu/ocl/ref_inner_product.hpp"
#include "gpu/ocl/ref_layer_normalization.hpp"
#include "gpu/ocl/ref_lrn.hpp"
#include "gpu/ocl/ref_matmul.hpp"
#include "gpu/ocl/ref_pooling.hpp"
#include "gpu/ocl/ref_resampling.hpp"
#include "gpu/ocl/ref_shuffle.hpp"
#include "gpu/ocl/ref_softmax.hpp"
#include "gpu/ocl/ref_zero_pad.hpp"
#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

using pd_create_f = dnnl::impl::engine_t::primitive_desc_create_f;

namespace {

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
const pd_create_f gpu_impl_list[] = {
        // Elementwise
        INSTANCE(ocl::ref_eltwise_fwd_t),
        INSTANCE(ocl::ref_eltwise_bwd_t),

        // Deconvolution
        INSTANCE(ocl::ref_deconvolution_fwd_t),
        INSTANCE(ocl::ref_deconvolution_bwd_data_t),
        INSTANCE(ocl::ref_deconvolution_bwd_weights_t),

        // Convolution
        INSTANCE(ocl::gen12lp_x8s8s32x_1x1_convolution_fwd_t),
        INSTANCE(ocl::gen12lp_x8s8s32x_convolution_fwd_t),
        INSTANCE(ocl::gen12lp_x8s8s32x_convolution_bwd_data_t),
        INSTANCE(ocl::gen9_wino_convolution_fwd_t),
        INSTANCE(ocl::gen9_convolution_fwd_t),
        INSTANCE(ocl::gen9_convolution_bwd_data_t),
        INSTANCE(ocl::gen9_convolution_bwd_weights_t),
        INSTANCE(ocl::ref_convolution_fwd_t),
        INSTANCE(ocl::ref_convolution_bwd_data_t),
        INSTANCE(ocl::ref_convolution_bwd_weights_t),

        // Batch Normalization
        INSTANCE(ocl::gen9_batch_normalization_fwd_t),
        INSTANCE(ocl::gen9_batch_normalization_bwd_t),
        INSTANCE(ocl::ref_batch_normalization_fwd_t),
        INSTANCE(ocl::ref_batch_normalization_bwd_t),

        // Pooling
        INSTANCE(ocl::gen9_pooling_fwd_t),
        INSTANCE(ocl::gen9_pooling_bwd_t),
        INSTANCE(ocl::ref_pooling_fwd_t),
        INSTANCE(ocl::ref_pooling_bwd_t),

        // LRN
        INSTANCE(ocl::ref_lrn_fwd_t),
        INSTANCE(ocl::ref_lrn_bwd_t),

        // Inner Product
        INSTANCE(ocl::gemm_inner_product_fwd_t),
        INSTANCE(ocl::gemm_post_ops_inner_product_fwd_t),
        INSTANCE(ocl::convolution_inner_product_fwd_t),
        INSTANCE(ocl::gemm_inner_product_bwd_data_t),
        INSTANCE(ocl::gemm_inner_product_bwd_weights_t),
        INSTANCE(ocl::ref_inner_product_fwd_t),
        INSTANCE(ocl::ref_inner_product_bwd_data_t),
        INSTANCE(ocl::ref_inner_product_bwd_weights_t),

        // Softmax
        INSTANCE(ocl::gen9_softmax_fwd_t),
        INSTANCE(ocl::ref_softmax_fwd_t),
        INSTANCE(ocl::ref_softmax_bwd_t),

        // GEMM (internal)
        INSTANCE(jit::gen_gemm_t),
        INSTANCE(ocl::gen12lp_gemm_t),
        INSTANCE(ocl::gen9_gemm_x8x8s32_t),
        INSTANCE(ocl::gen9_gemm_t),
        INSTANCE(ocl::ref_gemm_t),

        // RNN
        INSTANCE(ocl::ref_rnn_fwd_t),
        INSTANCE(ocl::ref_rnn_bwd_t),

        // Shuffle
        INSTANCE(ocl::ref_shuffle_t),

        // Layer Normalization
        INSTANCE(ocl::ref_layer_normalization_fwd_t),
        INSTANCE(ocl::ref_layer_normalization_bwd_t),

        // Binary
        INSTANCE(ocl::gen9_binary_t),
        INSTANCE(ocl::ref_binary_t),

        // MatMul
        INSTANCE(ocl::gemm_matmul_t),
        INSTANCE(ocl::ref_matmul_t),

        // Resampling
        INSTANCE(ocl::ref_resampling_fwd_t),
        INSTANCE(ocl::ref_resampling_bwd_t),

        // Zero Pad
        INSTANCE(ocl::ref_zero_pad_t),
        nullptr,
};

#undef INSTANCE
} // namespace

const pd_create_f *gpu_impl_list_t::get_implementation_list() {
    return gpu_impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
