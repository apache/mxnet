/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef COMMON_C_TYPES_MAP_HPP
#define COMMON_C_TYPES_MAP_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "gemm_types.hpp"
#include "internal_desc_types.hpp"

// These aliases should be in the global namespace as they are intended
// to give names that better reflects the meaning of the entities
using primitive_iface_t = dnnl_primitive;
using primitive_desc_iface_t = dnnl_primitive_desc;

namespace dnnl {
namespace impl {

// TODO: autogenerate this

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using stride_t = dnnl_dim_t;
using strides_t = dnnl_dims_t;

using status_t = dnnl_status_t;
namespace status {
const status_t success = dnnl_success;
const status_t out_of_memory = dnnl_out_of_memory;
const status_t invalid_arguments = dnnl_invalid_arguments;
const status_t unimplemented = dnnl_unimplemented;
const status_t iterator_ends = dnnl_iterator_ends;
const status_t runtime_error = dnnl_runtime_error;
const status_t not_required = dnnl_not_required;
} // namespace status

using prop_kind_t = dnnl_prop_kind_t;
namespace prop_kind {
const prop_kind_t undef = dnnl_prop_kind_undef;
const prop_kind_t forward_training = dnnl_forward_training;
const prop_kind_t forward_inference = dnnl_forward_inference;
const prop_kind_t forward_scoring = dnnl_forward_scoring;
const prop_kind_t forward = dnnl_forward;
const prop_kind_t backward = dnnl_backward;
const prop_kind_t backward_data = dnnl_backward_data;
const prop_kind_t backward_weights = dnnl_backward_weights;
const prop_kind_t backward_bias = dnnl_backward_bias;
} // namespace prop_kind

using alg_kind_t = dnnl_alg_kind_t;
namespace alg_kind {
const alg_kind_t undef = dnnl_alg_kind_undef;
const alg_kind_t convolution_auto = dnnl_convolution_auto;
const alg_kind_t convolution_direct = dnnl_convolution_direct;
const alg_kind_t convolution_winograd = dnnl_convolution_winograd;
const alg_kind_t deconvolution_direct = dnnl_deconvolution_direct;
const alg_kind_t deconvolution_winograd = dnnl_deconvolution_winograd;
const alg_kind_t eltwise_relu = dnnl_eltwise_relu;
const alg_kind_t eltwise_tanh = dnnl_eltwise_tanh;
const alg_kind_t eltwise_elu = dnnl_eltwise_elu;
const alg_kind_t eltwise_square = dnnl_eltwise_square;
const alg_kind_t eltwise_abs = dnnl_eltwise_abs;
const alg_kind_t eltwise_sqrt = dnnl_eltwise_sqrt;
const alg_kind_t eltwise_swish = dnnl_eltwise_swish;
const alg_kind_t eltwise_linear = dnnl_eltwise_linear;
const alg_kind_t eltwise_bounded_relu = dnnl_eltwise_bounded_relu;
const alg_kind_t eltwise_soft_relu = dnnl_eltwise_soft_relu;
const alg_kind_t eltwise_logistic = dnnl_eltwise_logistic;
const alg_kind_t eltwise_exp = dnnl_eltwise_exp;
const alg_kind_t eltwise_gelu = dnnl_eltwise_gelu;
const alg_kind_t eltwise_log = dnnl_eltwise_log;
const alg_kind_t eltwise_clip = dnnl_eltwise_clip;
const alg_kind_t eltwise_pow = dnnl_eltwise_pow;
const alg_kind_t eltwise_gelu_tanh = dnnl_eltwise_gelu_tanh;
const alg_kind_t eltwise_gelu_erf = dnnl_eltwise_gelu_erf;
const alg_kind_t eltwise_relu_use_dst_for_bwd
        = dnnl_eltwise_relu_use_dst_for_bwd;
const alg_kind_t eltwise_tanh_use_dst_for_bwd
        = dnnl_eltwise_tanh_use_dst_for_bwd;
const alg_kind_t eltwise_elu_use_dst_for_bwd = dnnl_eltwise_elu_use_dst_for_bwd;
const alg_kind_t eltwise_sqrt_use_dst_for_bwd
        = dnnl_eltwise_sqrt_use_dst_for_bwd;
const alg_kind_t eltwise_logistic_use_dst_for_bwd
        = dnnl_eltwise_logistic_use_dst_for_bwd;
const alg_kind_t eltwise_exp_use_dst_for_bwd = dnnl_eltwise_exp_use_dst_for_bwd;
const alg_kind_t eltwise_round = dnnl_eltwise_round;
const alg_kind_t pooling_max = dnnl_pooling_max;
const alg_kind_t pooling_avg = dnnl_pooling_avg;
const alg_kind_t pooling_avg_include_padding = dnnl_pooling_avg_include_padding;
const alg_kind_t pooling_avg_exclude_padding = dnnl_pooling_avg_exclude_padding;
const alg_kind_t lrn_across_channels = dnnl_lrn_across_channels;
const alg_kind_t lrn_within_channel = dnnl_lrn_within_channel;
const alg_kind_t vanilla_rnn = dnnl_vanilla_rnn;
const alg_kind_t vanilla_lstm = dnnl_vanilla_lstm;
const alg_kind_t vanilla_gru = dnnl_vanilla_gru;
const alg_kind_t lbr_gru = dnnl_lbr_gru;
const alg_kind_t binary_add = dnnl_binary_add;
const alg_kind_t binary_mul = dnnl_binary_mul;
const alg_kind_t binary_max = dnnl_binary_max;
const alg_kind_t binary_min = dnnl_binary_min;
const alg_kind_t binary_div = dnnl_binary_div;
const alg_kind_t resampling_nearest = dnnl_resampling_nearest;
const alg_kind_t resampling_linear = dnnl_resampling_linear;
const alg_kind_t reduction_max = dnnl_reduction_max;
const alg_kind_t reduction_min = dnnl_reduction_min;
const alg_kind_t reduction_sum = dnnl_reduction_sum;
const alg_kind_t reduction_mul = dnnl_reduction_mul;
const alg_kind_t reduction_mean = dnnl_reduction_mean;
const alg_kind_t reduction_norm_lp_max = dnnl_reduction_norm_lp_max;
const alg_kind_t reduction_norm_lp_sum = dnnl_reduction_norm_lp_sum;
const alg_kind_t reduction_norm_lp_power_p_max
        = dnnl_reduction_norm_lp_power_p_max;
const alg_kind_t reduction_norm_lp_power_p_sum
        = dnnl_reduction_norm_lp_power_p_sum;
} // namespace alg_kind

using data_type_t = dnnl_data_type_t;
namespace data_type {
const data_type_t undef = dnnl_data_type_undef;
const data_type_t f16 = dnnl_f16;
const data_type_t bf16 = dnnl_bf16;
const data_type_t f32 = dnnl_f32;
const data_type_t s32 = dnnl_s32;
const data_type_t s8 = dnnl_s8;
const data_type_t u8 = dnnl_u8;
} // namespace data_type

using scratchpad_mode_t = dnnl_scratchpad_mode_t;
namespace scratchpad_mode {
const scratchpad_mode_t library = dnnl_scratchpad_mode_library;
const scratchpad_mode_t user = dnnl_scratchpad_mode_user;
} // namespace scratchpad_mode

using rnn_packed_format_t = dnnl_rnn_packed_memory_format_t;
namespace rnn_packed_format {
const rnn_packed_format_t undef = dnnl_packed_format_undef;
const rnn_packed_format_t ldigo_p = dnnl_ldigo_p;
const rnn_packed_format_t ldgoi_p = dnnl_ldgoi_p;
} // namespace rnn_packed_format

using format_kind_t = dnnl_format_kind_t;
namespace format_kind {
const format_kind_t undef = dnnl_format_kind_undef;
const format_kind_t any = dnnl_format_kind_any;
const format_kind_t blocked = dnnl_blocked;
const format_kind_t wino = dnnl_format_kind_wino;
const format_kind_t rnn_packed = dnnl_format_kind_rnn_packed;
} // namespace format_kind

using format_tag_t = dnnl_format_tag_t;
namespace format_tag {
const format_tag_t undef = dnnl_format_tag_undef;
const format_tag_t any = dnnl_format_tag_any;
const format_tag_t a = dnnl_a;
const format_tag_t ab = dnnl_ab;
const format_tag_t abc = dnnl_abc;
const format_tag_t abcd = dnnl_abcd;
const format_tag_t abcde = dnnl_abcde;
const format_tag_t abcdef = dnnl_abcdef;
const format_tag_t abcdefg = dnnl_abcdefg;
const format_tag_t abcdefgh = dnnl_abcdefgh;
const format_tag_t abcdefghi = dnnl_abcdefghi;
const format_tag_t abcdefghij = dnnl_abcdefghij;
const format_tag_t abcdefghijk = dnnl_abcdefghijk;
const format_tag_t abcdefghijkl = dnnl_abcdefghijkl;
const format_tag_t abcdefghijlk = dnnl_abcdefghijlk;
const format_tag_t abcdefghikj = dnnl_abcdefghikj;
const format_tag_t abcdefghji = dnnl_abcdefghji;
const format_tag_t abcdefgih = dnnl_abcdefgih;
const format_tag_t abcdefhg = dnnl_abcdefhg;
const format_tag_t abcdegf = dnnl_abcdegf;
const format_tag_t abcdfe = dnnl_abcdfe;
const format_tag_t abced = dnnl_abced;
const format_tag_t abdc = dnnl_abdc;
const format_tag_t acbd = dnnl_acbd;
const format_tag_t abdec = dnnl_abdec;
const format_tag_t acb = dnnl_acb;
const format_tag_t acbde = dnnl_acbde;
const format_tag_t acbdef = dnnl_acbdef;
const format_tag_t acdb = dnnl_acdb;
const format_tag_t acdeb = dnnl_acdeb;
const format_tag_t ba = dnnl_ba;
const format_tag_t bac = dnnl_bac;
const format_tag_t bacd = dnnl_bacd;
const format_tag_t bca = dnnl_bca;
const format_tag_t bcda = dnnl_bcda;
const format_tag_t bcdea = dnnl_bcdea;
const format_tag_t bacde = dnnl_bacde;
const format_tag_t cba = dnnl_cba;
const format_tag_t cdba = dnnl_cdba;
const format_tag_t dcab = dnnl_dcab;
const format_tag_t cdeba = dnnl_cdeba;
const format_tag_t decab = dnnl_decab;
const format_tag_t defcab = dnnl_defcab;
const format_tag_t Abc16a = dnnl_Abc16a;
const format_tag_t ABc16a16b = dnnl_ABc16a16b;
const format_tag_t ABc4a4b = dnnl_ABc4a4b;
const format_tag_t aBc16b = dnnl_aBc16b;
const format_tag_t aBc32b = dnnl_aBc32b;
const format_tag_t ABc16b16a = dnnl_ABc16b16a;
const format_tag_t Abc4a = dnnl_Abc4a;
const format_tag_t aBc4b = dnnl_aBc4b;
const format_tag_t ABc4b16a4b = dnnl_ABc4b16a4b;
const format_tag_t ABc2b8a4b = dnnl_ABc2b8a4b;
const format_tag_t ABc16b16a4b = dnnl_ABc16b16a4b;
const format_tag_t ABc16b16a2b = dnnl_ABc16b16a2b;
const format_tag_t ABc4b4a = dnnl_ABc4b4a;
const format_tag_t ABc8a16b2a = dnnl_ABc8a16b2a;
const format_tag_t BAc8a16b2a = dnnl_BAc8a16b2a;
const format_tag_t ABc8a8b = dnnl_ABc8a8b;
const format_tag_t ABc8a4b = dnnl_ABc8a4b;
const format_tag_t aBc8b = dnnl_aBc8b;
const format_tag_t ABc8b16a2b = dnnl_ABc8b16a2b;
const format_tag_t ABc8b8a = dnnl_ABc8b8a;
const format_tag_t Abcd16a = dnnl_Abcd16a;
const format_tag_t Abcd8a = dnnl_Abcd8a;
const format_tag_t Abcd32a = dnnl_Abcd32a;
const format_tag_t ABcd16a16b = dnnl_ABcd16a16b;
const format_tag_t aBcd16b = dnnl_aBcd16b;
const format_tag_t aBcd32b = dnnl_aBcd32b;
const format_tag_t ABcd16b16a = dnnl_ABcd16b16a;
const format_tag_t aBCd16b16c = dnnl_aBCd16b16c;
const format_tag_t aBCd16c16b = dnnl_aBCd16c16b;
const format_tag_t Abcd4a = dnnl_Abcd4a;
const format_tag_t aBcd4b = dnnl_aBcd4b;
const format_tag_t ABcd4b16a4b = dnnl_ABcd4b16a4b;
const format_tag_t ABcd16b16a4b = dnnl_ABcd16b16a4b;
const format_tag_t ABcd16b16a2b = dnnl_ABcd16b16a2b;
const format_tag_t ABcd4b4a = dnnl_ABcd4b4a;
const format_tag_t ABcd4a4b = dnnl_ABcd4a4b;
const format_tag_t aBCd4c16b4c = dnnl_aBCd4c16b4c;
const format_tag_t aBCd2c8b4c = dnnl_aBCd2c8b4c;
const format_tag_t aBCd16c16b4c = dnnl_aBCd16c16b4c;
const format_tag_t aBCd16c16b2c = dnnl_aBCd16c16b2c;
const format_tag_t aBCd4c4b = dnnl_aBCd4c4b;
const format_tag_t aBCd4b4c = dnnl_aBCd4b4c;
const format_tag_t ABcd8a16b2a = dnnl_ABcd8a16b2a;
const format_tag_t BAcd8a16b2a = dnnl_BAcd8a16b2a;
const format_tag_t ABcd8a8b = dnnl_ABcd8a8b;
const format_tag_t ABcd8a4b = dnnl_ABcd8a4b;
const format_tag_t aBcd8b = dnnl_aBcd8b;
const format_tag_t ABcd8b16a2b = dnnl_ABcd8b16a2b;
const format_tag_t ABcd2b8a4b = dnnl_ABcd2b8a4b;
const format_tag_t aBCd8b16c2b = dnnl_aBCd8b16c2b;
const format_tag_t aCBd8b16c2b = dnnl_aCBd8b16c2b;
const format_tag_t ABcd8b8a = dnnl_ABcd8b8a;
const format_tag_t aBCd8b8c = dnnl_aBCd8b8c;
const format_tag_t aBCd8b4c = dnnl_aBCd8b4c;
const format_tag_t aBCd8c16b2c = dnnl_aBCd8c16b2c;
const format_tag_t aBCd8c8b = dnnl_aBCd8c8b;
const format_tag_t Abcde16a = dnnl_Abcde16a;
const format_tag_t Abcde32a = dnnl_Abcde32a;
const format_tag_t ABcde16a16b = dnnl_ABcde16a16b;
const format_tag_t aBcde16b = dnnl_aBcde16b;
const format_tag_t aBcde32b = dnnl_aBcde32b;
const format_tag_t ABcde16b16a = dnnl_ABcde16b16a;
const format_tag_t aBCde16b16c = dnnl_aBCde16b16c;
const format_tag_t aBCde16c16b = dnnl_aBCde16c16b;
const format_tag_t aBCde2c8b4c = dnnl_aBCde2c8b4c;
const format_tag_t Abcde4a = dnnl_Abcde4a;
const format_tag_t aBcde4b = dnnl_aBcde4b;
const format_tag_t ABcde4b4a = dnnl_ABcde4b4a;
const format_tag_t ABcde4a4b = dnnl_ABcde4a4b;
const format_tag_t aBCde4b4c = dnnl_aBCde4b4c;
const format_tag_t aBCde4c16b4c = dnnl_aBCde4c16b4c;
const format_tag_t aBCde16c16b4c = dnnl_aBCde16c16b4c;
const format_tag_t aBCde16c16b2c = dnnl_aBCde16c16b2c;
const format_tag_t aBCde4c4b = dnnl_aBCde4c4b;
const format_tag_t Abcde8a = dnnl_Abcde8a;
const format_tag_t ABcde8a8b = dnnl_ABcde8a8b;
const format_tag_t ABcde8a4b = dnnl_ABcde8a4b;
const format_tag_t aBcde8b = dnnl_aBcde8b;
const format_tag_t ABcde8b16a2b = dnnl_ABcde8b16a2b;
const format_tag_t ABcde8a16b2a = dnnl_ABcde8a16b2a;
const format_tag_t BAcde8a16b2a = dnnl_BAcde8a16b2a;
const format_tag_t ABcde4b16a4b = dnnl_ABcde4b16a4b;
const format_tag_t ABcde2b8a4b = dnnl_ABcde2b8a4b;
const format_tag_t aBCde8b16c2b = dnnl_aBCde8b16c2b;
const format_tag_t aCBde8b16c2b = dnnl_aCBde8b16c2b;
const format_tag_t ABcde8b8a = dnnl_ABcde8b8a;
const format_tag_t aBCde8b8c = dnnl_aBCde8b8c;
const format_tag_t aBCde8b4c = dnnl_aBCde8b4c;
const format_tag_t ABc4a8b8a4b = dnnl_ABc4a8b8a4b;
const format_tag_t ABcd4a8b8a4b = dnnl_ABcd4a8b8a4b;
const format_tag_t ABcde4a8b8a4b = dnnl_ABcde4a8b8a4b;
const format_tag_t ABcd2a8b8a2b = dnnl_ABcd2a8b8a2b;
const format_tag_t aBCd4b8c8b4c = dnnl_aBCd4b8c8b4c;
const format_tag_t aBCde4b8c8b4c = dnnl_aBCde4b8c8b4c;
const format_tag_t aBCdef4b8c8b4c = dnnl_aBCdef4b8c8b4c;
const format_tag_t BAc4b8a8b4a = dnnl_BAc4b8a8b4a;
const format_tag_t BAcd4b8a8b4a = dnnl_BAcd4b8a8b4a;
const format_tag_t BAcde4b8a8b4a = dnnl_BAcde4b8a8b4a;
const format_tag_t aCBd4c8b8c4b = dnnl_aCBd4c8b8c4b;
const format_tag_t aCBde4c8b8c4b = dnnl_aCBde4c8b8c4b;
const format_tag_t aCBdef4c8b8c4b = dnnl_aCBdef4c8b8c4b;
const format_tag_t aBCde2b8c8b2c = dnnl_aBCde2b8c8b2c;
const format_tag_t aBCde8c16b2c = dnnl_aBCde8c16b2c;
const format_tag_t aBCde8c8b = dnnl_aBCde8c8b;
const format_tag_t aBcdef16b = dnnl_aBcdef16b;
const format_tag_t aBCdef16b16c = dnnl_aBCdef16b16c;
const format_tag_t aBCdef16c16b = dnnl_aBCdef16c16b;
const format_tag_t aBCdef4c16b4c = dnnl_aBCdef4c16b4c;
const format_tag_t aBCdef2c8b4c = dnnl_aBCdef2c8b4c;
const format_tag_t aBcdef4b = dnnl_aBcdef4b;
const format_tag_t aBCdef4c4b = dnnl_aBCdef4c4b;
const format_tag_t aBCdef4b4c = dnnl_aBCdef4b4c;
const format_tag_t aBCdef8b8c = dnnl_aBCdef8b8c;
const format_tag_t aBCdef8b4c = dnnl_aBCdef8b4c;
const format_tag_t aBCdef8c16b2c = dnnl_aBCdef8c16b2c;
const format_tag_t aBCdef8b16c2b = dnnl_aBCdef8b16c2b;
const format_tag_t aCBdef8b16c2b = dnnl_aCBdef8b16c2b;
const format_tag_t aBCdef8c8b = dnnl_aBCdef8c8b;
const format_tag_t aBdc16b = dnnl_aBdc16b;
const format_tag_t aBdC16b2c = dnnl_aBdC16b2c;
const format_tag_t aBdC16b4c = dnnl_aBdC16b4c;
const format_tag_t aBdc4b = dnnl_aBdc4b;
const format_tag_t aBdc8b = dnnl_aBdc8b;
const format_tag_t aBdec16b = dnnl_aBdec16b;
const format_tag_t aBdeC16b2c = dnnl_aBdeC16b2c;
const format_tag_t aBdeC16b4c = dnnl_aBdeC16b4c;
const format_tag_t aBdec4b = dnnl_aBdec4b;
const format_tag_t aBdec8b = dnnl_aBdec8b;
const format_tag_t aBdefc16b = dnnl_aBdefc16b;
const format_tag_t aBdefC16b2c = dnnl_aBdefC16b2c;
const format_tag_t aCBdef16c16b = dnnl_aCBdef16c16b;
const format_tag_t aCBdef16b16c = dnnl_aCBdef16b16c;
const format_tag_t aBdefc4b = dnnl_aBdefc4b;
const format_tag_t aBdefc8b = dnnl_aBdefc8b;
const format_tag_t aBedc16b = dnnl_aBedc16b;
const format_tag_t Acb16a = dnnl_Acb16a;
const format_tag_t AcB16a2b = dnnl_AcB16a2b;
const format_tag_t AcB16a4b = dnnl_AcB16a4b;
const format_tag_t Acb4a = dnnl_Acb4a;
const format_tag_t Acb8a = dnnl_Acb8a;
const format_tag_t aCBd16b16c = dnnl_aCBd16b16c;
const format_tag_t aCBd16c16b = dnnl_aCBd16c16b;
const format_tag_t aCBde16b16c = dnnl_aCBde16b16c;
const format_tag_t aCBde16c16b = dnnl_aCBde16c16b;
const format_tag_t Acdb16a = dnnl_Acdb16a;
const format_tag_t AcdB16a2b = dnnl_AcdB16a2b;
const format_tag_t AcdB16a4b = dnnl_AcdB16a4b;
const format_tag_t Acdb4a = dnnl_Acdb4a;
const format_tag_t Acdb8a = dnnl_Acdb8a;
const format_tag_t Acdeb16a = dnnl_Acdeb16a;
const format_tag_t AcdeB16a2b = dnnl_AcdeB16a2b;
const format_tag_t Acdeb4a = dnnl_Acdeb4a;
const format_tag_t Acdeb8a = dnnl_Acdeb8a;
const format_tag_t Adcb16a = dnnl_Adcb16a;
const format_tag_t BAc16a16b = dnnl_BAc16a16b;
const format_tag_t BAcd16a16b = dnnl_BAcd16a16b;
const format_tag_t ABc32a32b = dnnl_ABc32a32b;
const format_tag_t BAcde16a16b = dnnl_BAcde16a16b;
const format_tag_t ABcd32a32b = dnnl_ABcd32a32b;
const format_tag_t ABcde32a32b = dnnl_ABcde32a32b;
const format_tag_t BAcde16b16a = dnnl_BAcde16b16a;
const format_tag_t aBdec32b = dnnl_aBdec32b;
const format_tag_t Abcdef16a = dnnl_Abcdef16a;
const format_tag_t Abcdef32a = dnnl_Abcdef32a;
const format_tag_t Acdb32a = dnnl_Acdb32a;
const format_tag_t BAc16b16a = dnnl_BAc16b16a;
const format_tag_t BAcd16b16a = dnnl_BAcd16b16a;
const format_tag_t aBCd2b4c2b = dnnl_aBCd2b4c2b;
const format_tag_t aBCde2b4c2b = dnnl_aBCde2b4c2b;
const format_tag_t aBCdef2b4c2b = dnnl_aBCdef2b4c2b;
const format_tag_t aBCd2c4b2c = dnnl_aBCd2c4b2c;
const format_tag_t aBCde2c4b2c = dnnl_aBCde2c4b2c;
const format_tag_t aBCdef2c4b2c = dnnl_aBCdef2c4b2c;
const format_tag_t aBCd4b8c2b = dnnl_aBCd4b8c2b;
const format_tag_t aBCde4b8c2b = dnnl_aBCde4b8c2b;
const format_tag_t aBCdef4b8c2b = dnnl_aBCdef4b8c2b;
const format_tag_t aBCd4c8b2c = dnnl_aBCd4c8b2c;
const format_tag_t aBCde4c8b2c = dnnl_aBCde4c8b2c;
const format_tag_t aBCdef4c8b2c = dnnl_aBCdef4c8b2c;
const format_tag_t last = dnnl_format_tag_last;

const format_tag_t x = dnnl_x;
const format_tag_t nc = dnnl_nc;
const format_tag_t cn = dnnl_cn;
const format_tag_t ncw = dnnl_ncw;
const format_tag_t nwc = dnnl_nwc;
const format_tag_t nchw = dnnl_nchw;
const format_tag_t nhwc = dnnl_nhwc;
const format_tag_t chwn = dnnl_chwn;
const format_tag_t ncdhw = dnnl_ncdhw;
const format_tag_t ndhwc = dnnl_ndhwc;
const format_tag_t oi = dnnl_oi;
const format_tag_t io = dnnl_io;
const format_tag_t oiw = dnnl_oiw;
const format_tag_t wio = dnnl_wio;
const format_tag_t owi = dnnl_owi;
const format_tag_t iwo = dnnl_iwo;
const format_tag_t oihw = dnnl_oihw;
const format_tag_t hwio = dnnl_hwio;
const format_tag_t ohwi = dnnl_ohwi;
const format_tag_t ihwo = dnnl_ihwo;
const format_tag_t iohw = dnnl_iohw;
const format_tag_t oidhw = dnnl_oidhw;
const format_tag_t dhwio = dnnl_dhwio;
const format_tag_t odhwi = dnnl_odhwi;
const format_tag_t idhwo = dnnl_idhwo;
const format_tag_t iodhw = dnnl_iodhw;
const format_tag_t goiw = dnnl_goiw;
const format_tag_t goihw = dnnl_goihw;
const format_tag_t wigo = dnnl_wigo;
const format_tag_t hwigo = dnnl_hwigo;
const format_tag_t dhwigo = dnnl_dhwigo;
const format_tag_t giohw = dnnl_giohw;
const format_tag_t goidhw = dnnl_goidhw;
const format_tag_t giodhw = dnnl_giodhw;
const format_tag_t tnc = dnnl_tnc;
const format_tag_t ntc = dnnl_ntc;
const format_tag_t ldnc = dnnl_ldnc;
const format_tag_t ldigo = dnnl_ldigo;
const format_tag_t ldgoi = dnnl_ldgoi;
const format_tag_t ldio = dnnl_ldio;
const format_tag_t ldoi = dnnl_ldoi;
const format_tag_t ldgo = dnnl_ldgo;
const format_tag_t nCdhw32c = dnnl_nCdhw32c;
const format_tag_t nCdhw16c = dnnl_nCdhw16c;
const format_tag_t nCdhw4c = dnnl_nCdhw4c;
const format_tag_t nCdhw8c = dnnl_nCdhw8c;
const format_tag_t nChw32c = dnnl_nChw32c;
const format_tag_t nChw16c = dnnl_nChw16c;
const format_tag_t nChw4c = dnnl_nChw4c;
const format_tag_t nChw8c = dnnl_nChw8c;
const format_tag_t nCw32c = dnnl_nCw32c;
const format_tag_t nCw16c = dnnl_nCw16c;
const format_tag_t nCw4c = dnnl_nCw4c;
const format_tag_t nCw8c = dnnl_nCw8c;
const format_tag_t NCw16n16c = dnnl_NCw16n16c;
const format_tag_t NChw16n16c = dnnl_NChw16n16c;
const format_tag_t NCdhw16n16c = dnnl_NCdhw16n16c;
const format_tag_t NCw32n32c = dnnl_NCw32n32c;
const format_tag_t NChw32n32c = dnnl_NChw32n32c;
const format_tag_t NCdhw32n32c = dnnl_NCdhw32n32c;
const format_tag_t IOdhw16i16o = dnnl_IOdhw16i16o;
const format_tag_t IOhw16i16o = dnnl_IOhw16i16o;
const format_tag_t Ohwi32o = dnnl_Ohwi32o;
const format_tag_t gIOhw16i16o = dnnl_gIOhw16i16o;
const format_tag_t gOhwi32o = dnnl_gOhwi32o;
const format_tag_t Goidhw16g = dnnl_Goidhw16g;
const format_tag_t IOw16o16i = dnnl_IOw16o16i;
const format_tag_t IOw16i16o = dnnl_IOw16i16o;
const format_tag_t gIOw16i16o = dnnl_gIOw16i16o;
const format_tag_t OIw16i16o = dnnl_OIw16i16o;
const format_tag_t OIw16o16i = dnnl_OIw16o16i;
const format_tag_t Oiw16o = dnnl_Oiw16o;
const format_tag_t OIw4i16o4i = dnnl_OIw4i16o4i;
const format_tag_t OIw2i8o4i = dnnl_OIw2i8o4i;
const format_tag_t OIw16i16o4i = dnnl_OIw16i16o4i;
const format_tag_t OIw16i16o2i = dnnl_OIw16i16o2i;
const format_tag_t OIw4i4o = dnnl_OIw4i4o;
const format_tag_t OIw4o4i = dnnl_OIw4o4i;
const format_tag_t Oiw4o = dnnl_Oiw4o;
const format_tag_t OIw8i16o2i = dnnl_OIw8i16o2i;
const format_tag_t OIw8i8o = dnnl_OIw8i8o;
const format_tag_t OIw8o16i2o = dnnl_OIw8o16i2o;
const format_tag_t IOw8o16i2o = dnnl_IOw8o16i2o;
const format_tag_t OIw8o8i = dnnl_OIw8o8i;
const format_tag_t OIw8o4i = dnnl_OIw8o4i;
const format_tag_t Owi16o = dnnl_Owi16o;
const format_tag_t OwI16o2i = dnnl_OwI16o2i;
const format_tag_t OwI16o4i = dnnl_OwI16o4i;
const format_tag_t Owi4o = dnnl_Owi4o;
const format_tag_t Owi8o = dnnl_Owi8o;
const format_tag_t IOdhw16o16i = dnnl_IOdhw16o16i;
const format_tag_t IOhw16o16i = dnnl_IOhw16o16i;
const format_tag_t Ohwi16o = dnnl_Ohwi16o;
const format_tag_t OhwI16o2i = dnnl_OhwI16o2i;
const format_tag_t OhwI16o4i = dnnl_OhwI16o4i;
const format_tag_t Ohwi4o = dnnl_Ohwi4o;
const format_tag_t Ohwi8o = dnnl_Ohwi8o;
const format_tag_t OIhw16i16o = dnnl_OIhw16i16o;
const format_tag_t OIhw16o16i = dnnl_OIhw16o16i;
const format_tag_t Oihw16o = dnnl_Oihw16o;
const format_tag_t OIhw4i16o4i = dnnl_OIhw4i16o4i;
const format_tag_t OIhw16i16o4i = dnnl_OIhw16i16o4i;
const format_tag_t OIhw16i16o2i = dnnl_OIhw16i16o2i;
const format_tag_t OIhw4i4o = dnnl_OIhw4i4o;
const format_tag_t OIhw4o4i = dnnl_OIhw4o4i;
const format_tag_t Oihw4o = dnnl_Oihw4o;
const format_tag_t OIhw8i16o2i = dnnl_OIhw8i16o2i;
const format_tag_t OIhw2i8o4i = dnnl_OIhw2i8o4i;
const format_tag_t OIhw8i8o = dnnl_OIhw8i8o;
const format_tag_t OIhw8o16i2o = dnnl_OIhw8o16i2o;
const format_tag_t IOhw8o16i2o = dnnl_IOhw8o16i2o;
const format_tag_t OIhw8o8i = dnnl_OIhw8o8i;
const format_tag_t OIhw8o4i = dnnl_OIhw8o4i;
const format_tag_t Owhi16o = dnnl_Owhi16o;
const format_tag_t Odhwi16o = dnnl_Odhwi16o;
const format_tag_t OdhwI16o2i = dnnl_OdhwI16o2i;
const format_tag_t Odhwi4o = dnnl_Odhwi4o;
const format_tag_t Odhwi8o = dnnl_Odhwi8o;
const format_tag_t OIdhw16i16o = dnnl_OIdhw16i16o;
const format_tag_t OIdhw16o16i = dnnl_OIdhw16o16i;
const format_tag_t Oidhw16o = dnnl_Oidhw16o;
const format_tag_t OIdhw4i4o = dnnl_OIdhw4i4o;
const format_tag_t OIdhw4o4i = dnnl_OIdhw4o4i;
const format_tag_t Oidhw4o = dnnl_Oidhw4o;
const format_tag_t OIdhw8i16o2i = dnnl_OIdhw8i16o2i;
const format_tag_t OIdhw4i16o4i = dnnl_OIdhw4i16o4i;
const format_tag_t OIdhw2i8o4i = dnnl_OIdhw2i8o4i;
const format_tag_t OIdhw8o16i2o = dnnl_OIdhw8o16i2o;
const format_tag_t IOdhw8o16i2o = dnnl_IOdhw8o16i2o;
const format_tag_t OIdhw8i8o = dnnl_OIdhw8i8o;
const format_tag_t OIdhw8o8i = dnnl_OIdhw8o8i;
const format_tag_t OIdhw8o4i = dnnl_OIdhw8o4i;
const format_tag_t gIOw16o16i = dnnl_gIOw16o16i;
const format_tag_t Goiw16g = dnnl_Goiw16g;
const format_tag_t Goiw8g = dnnl_Goiw8g;
const format_tag_t Goiw4g = dnnl_Goiw4g;
const format_tag_t gOIw16i16o = dnnl_gOIw16i16o;
const format_tag_t gOIw16o16i = dnnl_gOIw16o16i;
const format_tag_t gOiw16o = dnnl_gOiw16o;
const format_tag_t gOIw4i16o4i = dnnl_gOIw4i16o4i;
const format_tag_t gOIw2i8o4i = dnnl_gOIw2i8o4i;
const format_tag_t gOIw16i16o4i = dnnl_gOIw16i16o4i;
const format_tag_t gOIw16i16o2i = dnnl_gOIw16i16o2i;
const format_tag_t gOIw4i4o = dnnl_gOIw4i4o;
const format_tag_t gOIw4o4i = dnnl_gOIw4o4i;
const format_tag_t gOiw4o = dnnl_gOiw4o;
const format_tag_t gOIw8i16o2i = dnnl_gOIw8i16o2i;
const format_tag_t gOIw8i8o = dnnl_gOIw8i8o;
const format_tag_t gOIw8o16i2o = dnnl_gOIw8o16i2o;
const format_tag_t gIOw8o16i2o = dnnl_gIOw8o16i2o;
const format_tag_t gOIw8o8i = dnnl_gOIw8o8i;
const format_tag_t gOIw8o4i = dnnl_gOIw8o4i;
const format_tag_t gOwi16o = dnnl_gOwi16o;
const format_tag_t gOwI16o2i = dnnl_gOwI16o2i;
const format_tag_t gOwI16o4i = dnnl_gOwI16o4i;
const format_tag_t gOwi4o = dnnl_gOwi4o;
const format_tag_t gOwi8o = dnnl_gOwi8o;
const format_tag_t gIOdhw16o16i = dnnl_gIOdhw16o16i;
const format_tag_t gIOhw16o16i = dnnl_gIOhw16o16i;
const format_tag_t gOhwi16o = dnnl_gOhwi16o;
const format_tag_t gOhwI16o2i = dnnl_gOhwI16o2i;
const format_tag_t gOhwI16o4i = dnnl_gOhwI16o4i;
const format_tag_t gOhwi4o = dnnl_gOhwi4o;
const format_tag_t gOhwi8o = dnnl_gOhwi8o;
const format_tag_t Goihw16g = dnnl_Goihw16g;
const format_tag_t gOIhw16i16o = dnnl_gOIhw16i16o;
const format_tag_t gOIhw16o16i = dnnl_gOIhw16o16i;
const format_tag_t gOihw16o = dnnl_gOihw16o;
const format_tag_t gOIhw2i8o4i = dnnl_gOIhw2i8o4i;
const format_tag_t gOIhw4i16o4i = dnnl_gOIhw4i16o4i;
const format_tag_t gOIhw16i16o4i = dnnl_gOIhw16i16o4i;
const format_tag_t gOIhw16i16o2i = dnnl_gOIhw16i16o2i;
const format_tag_t gOIhw4i4o = dnnl_gOIhw4i4o;
const format_tag_t gOIhw4o4i = dnnl_gOIhw4o4i;
const format_tag_t gOihw4o = dnnl_gOihw4o;
const format_tag_t Goihw8g = dnnl_Goihw8g;
const format_tag_t Goihw4g = dnnl_Goihw4g;
const format_tag_t gOIhw8i16o2i = dnnl_gOIhw8i16o2i;
const format_tag_t gOIhw8i8o = dnnl_gOIhw8i8o;
const format_tag_t gOIhw8o16i2o = dnnl_gOIhw8o16i2o;
const format_tag_t OIw4o8i8o4i = dnnl_OIw4o8i8o4i;
const format_tag_t gIOhw8o16i2o = dnnl_gIOhw8o16i2o;
const format_tag_t OIhw4o8i8o4i = dnnl_OIhw4o8i8o4i;
const format_tag_t OIdhw4o8i8o4i = dnnl_OIdhw4o8i8o4i;
const format_tag_t IOw4i8o8i4o = dnnl_IOw4i8o8i4o;
const format_tag_t IOhw4i8o8i4o = dnnl_IOhw4i8o8i4o;
const format_tag_t IOdhw4i8o8i4o = dnnl_IOdhw4i8o8i4o;
const format_tag_t gIOw4i8o8i4o = dnnl_gIOw4i8o8i4o;
const format_tag_t gIOhw4i8o8i4o = dnnl_gIOhw4i8o8i4o;
const format_tag_t gIOdhw4i8o8i4o = dnnl_gIOdhw4i8o8i4o;
const format_tag_t OIhw2o8i8o2i = dnnl_OIhw2o8i8o2i;
const format_tag_t gOIw4o8i8o4i = dnnl_gOIw4o8i8o4i;
const format_tag_t gOIhw4o8i8o4i = dnnl_gOIhw4o8i8o4i;
const format_tag_t gOIdhw4o8i8o4i = dnnl_gOIdhw4o8i8o4i;
const format_tag_t gOIhw2o8i8o2i = dnnl_gOIhw2o8i8o2i;
const format_tag_t gOIhw8o8i = dnnl_gOIhw8o8i;
const format_tag_t gOIhw8o4i = dnnl_gOIhw8o4i;
const format_tag_t gOwhi16o = dnnl_gOwhi16o;
const format_tag_t gIOdhw16i16o = dnnl_gIOdhw16i16o;
const format_tag_t gOdhwi16o = dnnl_gOdhwi16o;
const format_tag_t gOdhwI16o2i = dnnl_gOdhwI16o2i;
const format_tag_t gOdhwi4o = dnnl_gOdhwi4o;
const format_tag_t gOdhwi8o = dnnl_gOdhwi8o;
const format_tag_t gOIdhw16i16o = dnnl_gOIdhw16i16o;
const format_tag_t gOIdhw16o16i = dnnl_gOIdhw16o16i;
const format_tag_t gOidhw16o = dnnl_gOidhw16o;
const format_tag_t gOIdhw4i4o = dnnl_gOIdhw4i4o;
const format_tag_t gOIdhw4o4i = dnnl_gOIdhw4o4i;
const format_tag_t gOidhw4o = dnnl_gOidhw4o;
const format_tag_t gOIdhw8i16o2i = dnnl_gOIdhw8i16o2i;
const format_tag_t gOIdhw4i16o4i = dnnl_gOIdhw4i16o4i;
const format_tag_t gOIdhw2i8o4i = dnnl_gOIdhw2i8o4i;
const format_tag_t gOIdhw8o16i2o = dnnl_gOIdhw8o16i2o;
const format_tag_t gIOdhw8o16i2o = dnnl_gIOdhw8o16i2o;
const format_tag_t gOIdhw8i8o = dnnl_gOIdhw8i8o;
const format_tag_t gOIdhw8o8i = dnnl_gOIdhw8o8i;
const format_tag_t gOIdhw8o4i = dnnl_gOIdhw8o4i;
const format_tag_t Goiw32g = dnnl_Goiw32g;
const format_tag_t Goihw32g = dnnl_Goihw32g;
const format_tag_t Goidhw32g = dnnl_Goidhw32g;
const format_tag_t gOIw2i4o2i = dnnl_gOIw2i4o2i;
const format_tag_t gOIhw2i4o2i = dnnl_gOIhw2i4o2i;
const format_tag_t gOIdhw2i4o2i = dnnl_gOIdhw2i4o2i;
const format_tag_t gOIw2o4i2o = dnnl_gOIw2o4i2o;
const format_tag_t gOIhw2o4i2o = dnnl_gOIhw2o4i2o;
const format_tag_t gOIdhw2o4i2o = dnnl_gOIdhw2o4i2o;
const format_tag_t gOIw4i8o2i = dnnl_gOIw4i8o2i;
const format_tag_t gOIhw4i8o2i = dnnl_gOIhw4i8o2i;
const format_tag_t gOIdhw4i8o2i = dnnl_gOIdhw4i8o2i;
const format_tag_t gOIw4o8i2o = dnnl_gOIw4o8i2o;
const format_tag_t gOIhw4o8i2o = dnnl_gOIhw4o8i2o;
const format_tag_t gOIdhw4o8i2o = dnnl_gOIdhw4o8i2o;
} // namespace format_tag

using memory_extra_flags_t = dnnl_memory_extra_flags_t;
namespace memory_extra_flags {
const memory_extra_flags_t none = dnnl_memory_extra_flag_none;
const memory_extra_flags_t compensation_conv_s8s8
        = dnnl_memory_extra_flag_compensation_conv_s8s8;
const memory_extra_flags_t scale_adjust = dnnl_memory_extra_flag_scale_adjust;
const memory_extra_flags_t gpu_rnn_u8s8_compensation
        = dnnl_memory_extra_flag_gpu_rnn_u8s8_compensation;
const memory_extra_flags_t compensation_conv_asymmetric_src
        = dnnl_memory_extra_flag_compensation_conv_asymmetric_src;
} // namespace memory_extra_flags

using engine_kind_t = dnnl_engine_kind_t;
namespace engine_kind {
const engine_kind_t any_engine = dnnl_any_engine;
const engine_kind_t cpu = dnnl_cpu;
const engine_kind_t gpu = dnnl_gpu;
} // namespace engine_kind

enum runtime_kind_t {
    dnnl_runtime_none,
    dnnl_runtime_seq,
    dnnl_runtime_omp,
    dnnl_runtime_tbb,
    dnnl_runtime_threadpool,
    dnnl_runtime_ocl,
    dnnl_runtime_sycl,
};

namespace runtime_kind {
const runtime_kind_t none = dnnl_runtime_none;
const runtime_kind_t seq = dnnl_runtime_seq;
const runtime_kind_t omp = dnnl_runtime_omp;
const runtime_kind_t tbb = dnnl_runtime_tbb;
const runtime_kind_t threadpool = dnnl_runtime_threadpool;
const runtime_kind_t ocl = dnnl_runtime_ocl;
const runtime_kind_t sycl = dnnl_runtime_sycl;
} // namespace runtime_kind

using primitive_kind_t = dnnl_primitive_kind_t;
namespace primitive_kind {
const primitive_kind_t undefined = dnnl_undefined_primitive;
const primitive_kind_t reorder = dnnl_reorder;
const primitive_kind_t concat = dnnl_concat;
const primitive_kind_t sum = dnnl_sum;
const primitive_kind_t convolution = dnnl_convolution;
const primitive_kind_t deconvolution = dnnl_deconvolution;
const primitive_kind_t shuffle = dnnl_shuffle;
const primitive_kind_t eltwise = dnnl_eltwise;
const primitive_kind_t softmax = dnnl_softmax;
const primitive_kind_t pooling = dnnl_pooling;
const primitive_kind_t pooling_v2 = dnnl_pooling_v2;
const primitive_kind_t lrn = dnnl_lrn;
const primitive_kind_t batch_normalization = dnnl_batch_normalization;
const primitive_kind_t layer_normalization = dnnl_layer_normalization;
const primitive_kind_t inner_product = dnnl_inner_product;
const primitive_kind_t rnn = dnnl_rnn;
const primitive_kind_t gemm = dnnl_gemm;
const primitive_kind_t binary = dnnl_binary;
const primitive_kind_t logsoftmax = dnnl_logsoftmax;
const primitive_kind_t matmul = dnnl_matmul;
const primitive_kind_t resampling = dnnl_resampling;
const primitive_kind_t reduction = dnnl_reduction;

// Internal only primitive kinds.
const primitive_kind_t internal_only_start = (primitive_kind_t)(1 << 12);
const primitive_kind_t zero_pad = internal_only_start;
} // namespace primitive_kind

using query_t = dnnl_query_t;
namespace query {
const query_t undef = dnnl_query_undef;

const query_t engine = dnnl_query_engine;
const query_t primitive_kind = dnnl_query_primitive_kind;

const query_t num_of_inputs_s32 = dnnl_query_num_of_inputs_s32;
const query_t num_of_outputs_s32 = dnnl_query_num_of_outputs_s32;

const query_t time_estimate_f64 = dnnl_query_time_estimate_f64;
const query_t memory_consumption_s64 = dnnl_query_memory_consumption_s64;

const query_t scratchpad_engine = dnnl_query_scratchpad_engine;

const query_t impl_info_str = dnnl_query_impl_info_str;

const query_t reorder_src_engine = dnnl_query_reorder_src_engine;
const query_t reorder_dst_engine = dnnl_query_reorder_dst_engine;

const query_t prop_kind = dnnl_query_prop_kind;

const query_t some_d = dnnl_query_some_d;
const query_t op_d = dnnl_query_op_d;
const query_t convolution_d = dnnl_query_convolution_d;
const query_t deconvolution_d = dnnl_query_deconvolution_d;
const query_t shuffle_d = dnnl_query_shuffle_d;
const query_t eltwise_d = dnnl_query_eltwise_d;
const query_t softmax_d = dnnl_query_softmax_d;
const query_t pooling_d = dnnl_query_pooling_d;
const query_t pooling_v2_d = dnnl_query_pooling_v2_d;
const query_t lrn_d = dnnl_query_lrn_d;
const query_t batch_normalization_d = dnnl_query_batch_normalization_d;
const query_t layer_normalization_d = dnnl_query_layer_normalization_d;
const query_t inner_product_d = dnnl_query_inner_product_d;
const query_t rnn_d = dnnl_query_rnn_d;
const query_t gemm_d = dnnl_query_gemm_d;
const query_t binary_d = dnnl_query_binary_d;
const query_t logsoftmax_d = dnnl_query_logsoftmax_d;
const query_t matmul_d = dnnl_query_matmul_d;
const query_t resampling_d = dnnl_query_resampling_d;
const query_t reduction_d = dnnl_query_reduction_d;

const query_t some_md = dnnl_query_some_md;
const query_t src_md = dnnl_query_src_md;
const query_t diff_src_md = dnnl_query_diff_src_md;
const query_t weights_md = dnnl_query_weights_md;
const query_t diff_weights_md = dnnl_query_diff_weights_md;
const query_t dst_md = dnnl_query_dst_md;
const query_t diff_dst_md = dnnl_query_diff_dst_md;
const query_t exec_arg_md = dnnl_query_exec_arg_md;

const query_t workspace_md = dnnl_query_workspace_md;
const query_t scratchpad_md = dnnl_query_scratchpad_md;

// Internal only query kinds.
const query_t internal_only_start = (query_t)(1 << 12);
const query_t zero_pad_d = internal_only_start;
} // namespace query

using blocking_desc_t = dnnl_blocking_desc_t;
using rnn_packed_desc_t = dnnl_rnn_packed_desc_t;
using wino_desc_t = dnnl_wino_desc_t;
using memory_extra_desc_t = dnnl_memory_extra_desc_t;
using memory_desc_t = dnnl_memory_desc_t;
using convolution_desc_t = dnnl_convolution_desc_t;
using deconvolution_desc_t = dnnl_deconvolution_desc_t;
using shuffle_desc_t = dnnl_shuffle_desc_t;
using pooling_desc_t = dnnl_pooling_desc_t;
using pooling_v2_desc_t = dnnl_pooling_v2_desc_t;
using eltwise_desc_t = dnnl_eltwise_desc_t;
using softmax_desc_t = dnnl_softmax_desc_t;
using lrn_desc_t = dnnl_lrn_desc_t;
using batch_normalization_desc_t = dnnl_batch_normalization_desc_t;
using layer_normalization_desc_t = dnnl_layer_normalization_desc_t;
using inner_product_desc_t = dnnl_inner_product_desc_t;
using binary_desc_t = dnnl_binary_desc_t;
using logsoftmax_desc_t = dnnl_logsoftmax_desc_t;
using matmul_desc_t = dnnl_matmul_desc_t;
using resampling_desc_t = dnnl_resampling_desc_t;
using reduction_desc_t = dnnl_reduction_desc_t;

using rnn_direction_t = dnnl_rnn_direction_t;
using rnn_desc_t = dnnl_rnn_desc_t;

/* Internal type, declared in gemm_types.hpp */
using gemm_desc_t = dnnl_gemm_desc_t;

/* Internal types, for the primitives which don't have descs */
using concat_desc_t = dnnl_concat_desc_t;
using reorder_desc_t = dnnl_reorder_desc_t;
using sum_desc_t = dnnl_sum_desc_t;
using zero_pad_desc_t = dnnl_zero_pad_desc_t;

/* C op_desc_t, which eventually are just (void*) */
using c_op_desc_t = dnnl_op_desc_t;
using const_c_op_desc_t = const_dnnl_op_desc_t;

struct op_desc_t {
    union {
        primitive_kind_t kind;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        shuffle_desc_t shuffle;
        pooling_desc_t pooling;
        pooling_v2_desc_t pooling_v2;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        layer_normalization_desc_t layer_normalization;
        inner_product_desc_t inner_product;
        rnn_desc_t rnn;
        gemm_desc_t gemm;
        concat_desc_t concat;
        reorder_desc_t reorder;
        sum_desc_t sum;
        binary_desc_t binary;
        matmul_desc_t matmul;
        resampling_desc_t resampling;
        zero_pad_desc_t zero_pad;
        reduction_desc_t reduction;
    };

#define DECL_CTOR_AND_CONVERTERS(c_type) \
    op_desc_t(const c_type &) = delete; \
    static op_desc_t *convert_from_c(c_type *_) { \
        return reinterpret_cast<op_desc_t *>(_); \
    } \
    static const op_desc_t *convert_from_c(const c_type *_) { \
        return reinterpret_cast<const op_desc_t *>(_); \
    }

    DECL_CTOR_AND_CONVERTERS(convolution_desc_t);
    DECL_CTOR_AND_CONVERTERS(shuffle_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_v2_desc_t);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(layer_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t);
    DECL_CTOR_AND_CONVERTERS(rnn_desc_t);
    DECL_CTOR_AND_CONVERTERS(gemm_desc_t);
    DECL_CTOR_AND_CONVERTERS(concat_desc_t);
    DECL_CTOR_AND_CONVERTERS(reorder_desc_t);
    DECL_CTOR_AND_CONVERTERS(sum_desc_t);
    DECL_CTOR_AND_CONVERTERS(binary_desc_t);
    DECL_CTOR_AND_CONVERTERS(matmul_desc_t);
    DECL_CTOR_AND_CONVERTERS(resampling_desc_t);
    DECL_CTOR_AND_CONVERTERS(zero_pad_desc_t);
    DECL_CTOR_AND_CONVERTERS(reduction_desc_t);

    // concat_desc_t and sum_desc_t have data members which have non-trivial
    // special member functions hence the default destructor is implicitly
    // deleted by the compiler which causes a warning on Windows so we should
    // delete the destructor explicitly.
    ~op_desc_t() = delete;

#undef DECL_CTOR_AND_CONVERTERS
};

using engine_t = dnnl_engine;
using primitive_desc_iterator_t = dnnl_primitive_desc_iterator;
using primitive_attr_t = dnnl_primitive_attr;
using post_ops_t = dnnl_post_ops;
using memory_t = dnnl_memory;

using stream_flags_t = dnnl_stream_flags_t;
namespace stream_flags {
const stream_flags_t in_order = dnnl_stream_in_order;
const stream_flags_t out_of_order = dnnl_stream_out_of_order;
const stream_flags_t default_flags = dnnl_stream_default_flags;
} // namespace stream_flags
using stream_t = dnnl_stream;

struct memory_storage_t;

/* forward declaration of the internal primitive_desc types */
struct batch_normalization_bwd_pd_t;
struct batch_normalization_fwd_pd_t;
struct batch_normalization_pd_t;
struct binary_pd_t;
struct concat_pd_t;
struct convolution_bwd_data_pd_t;
struct convolution_bwd_weights_pd_t;
struct convolution_fwd_pd_t;
struct convolution_pd_t;
struct deconvolution_bwd_data_pd_t;
struct deconvolution_bwd_weights_pd_t;
struct deconvolution_fwd_pd_t;
struct deconvolution_pd_t;
struct eltwise_bwd_pd_t;
struct eltwise_fwd_pd_t;
struct eltwise_pd_t;
struct gemm_pd_t;
struct inner_product_bwd_data_pd_t;
struct inner_product_bwd_weights_pd_t;
struct inner_product_fwd_pd_t;
struct inner_product_pd_t;
struct layer_normalization_bwd_pd_t;
struct layer_normalization_fwd_pd_t;
struct layer_normalization_pd_t;
struct lrn_bwd_pd_t;
struct lrn_fwd_pd_t;
struct lrn_pd_t;
struct matmul_pd_t;
struct pooling_bwd_pd_t;
struct pooling_fwd_pd_t;
struct pooling_pd_t;
struct reduction_pd_t;
struct reorder_pd_t;
struct resampling_pd_t;
struct rnn_bwd_pd_t;
struct rnn_fwd_pd_t;
struct rnn_pd_t;
struct shuffle_pd_t;
struct softmax_bwd_pd_t;
struct softmax_fwd_pd_t;
struct softmax_pd_t;
struct sum_pd_t;

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
