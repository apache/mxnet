/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
#define MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_

#include <string>
#include <map>
#include <vector>

#if MXNET_USE_CUDA

namespace mxnet {

namespace fusion {

const std::map<std::string, std::vector<std::vector<std::string>>> ops_desc = {
    {"elemwise_add", {{"op::add(%, %)", "_0", "_1"}}},
    {"_plus", {{"op::add(%, %)", "_0", "_1"}}},
    {"_Plus", {{"op::add(%, %)", "_0", "_1"}}},
    {"_add", {{"op::add(%, %)", "_0", "_1"}}},
    {"elemwise_sub", {{"op::sub(%, %)", "_0", "_1"}}},
    {"_minus", {{"op::sub(%, %)", "_0", "_1"}}},
    {"_Minus", {{"op::sub(%, %)", "_0", "_1"}}},
    {"_sub", {{"op::sub(%, %)", "_0", "_1"}}},
    {"elemwise_mul", {{"op::mul(%, %)", "_0", "_1"}}},
    {"_mul", {{"op::mul(%, %)", "_0", "_1"}}},
    {"_Mul", {{"op::mul(%, %)", "_0", "_1"}}},
    {"elemwise_div", {{"op::div(%, %)", "_0", "_1"}}},
    {"_div", {{"op::div(%, %)", "_0", "_1"}}},
    {"_Div", {{"op::div(%, %)", "_0", "_1"}}},
    {"_Power", {{"op::power(%, %)", "_0", "_1"}}},
    {"_power", {{"op::power(%, %)", "_0", "_1"}}},
    {"_Maximum", {{"op::max(%, %)", "_0", "_1"}}},
    {"_maximum", {{"op::max(%, %)", "_0", "_1"}}},
    {"_Minimum", {{"op::min(%, %)", "_0", "_1"}}},
    {"_minimum", {{"op::min(%, %)", "_0", "_1"}}},
    {"_mod", {{"op::mod(%, %)", "_0", "_1"}}},
    {"amp_cast", {{"op::identity(%)", "_0"}}},
    {"_backward_amp_cast", {{"op::identity(%)", "_0"}}},
    {"relu", {{"op::relu(%)", "_0"}}},
    {"sigmoid", {{"op::sigmoid(%)", "_0"}}},
    {"log_sigmoid", {{"op::log_sigmoid(%)", "_0"}}},
    {"mish", {{"op::mish(%)", "_0"}}},
    {"softsign", {{"op::softsign(%)", "_0"}}},
    {"exp", {{"op::exp(%)", "_0"}}},
    {"expm1", {{"op::expm1(%)", "_0"}}},
    {"log", {{"op::log(%)", "_0"}}},
    {"log10", {{"op::log10(%)", "_0"}}},
    {"log2", {{"op::log2(%)", "_0"}}},
    {"log1p", {{"op::log1p(%)", "_0"}}},
    {"degrees", {{"op::degrees(%)", "_0"}}},
    {"radians", {{"op::radians(%)", "_0"}}},
    {"sin", {{"op::sin(%)", "_0"}}},
    {"cos", {{"op::cos(%)", "_0"}}},
    {"tan", {{"op::tan(%)", "_0"}}},
    {"arcsin", {{"op::arcsin(%)", "_0"}}},
    {"arccos", {{"op::arccos(%)", "_0"}}},
    {"arctan", {{"op::arctan(%)", "_0"}}},
    {"sinh", {{"op::sinh(%)", "_0"}}},
    {"cosh", {{"op::cosh(%)", "_0"}}},
    {"tanh", {{"op::tanh(%)", "_0"}}},
    {"arcsinh", {{"op::arcsinh(%)", "_0"}}},
    {"arccosh", {{"op::arccosh(%)", "_0"}}},
    {"arctanh", {{"op::arctanh(%)", "_0"}}},
    {"sqrt", {{"op::sqrt(%)", "_0"}}},
    {"rsqrt", {{"op::rsqrt(%)", "_0"}}},
    {"cbrt", {{"op::cbrt(%)", "_0"}}},
    {"rcbrt", {{"op::rcbrt(%)", "_0"}}},
    {"square", {{"op::square(%)", "_0"}}},
    {"squeeze", {{"op::identity(%)", "_0"}}},
    {"zeros_like", {{"op::zero(%)", "_0"}}},
    {"ones_like", {{"op::one(%)", "_0"}}},
    {"flatten", {{"op::identity(%)", "_0"}}},
    {"Reshape", {{"op::identity(%)", "_0"}}},
    {"reshape", {{"op::identity(%)", "_0"}}},
    {"_backward_reshape", {{"op::identity(%)", "_0"}}},
    {"expand_dims", {{"op::identity(%)", "_0"}}},
    {"round", {{"op::round(%)", "_0"}}},
    {"rint", {{"op::rint(%)", "_0"}}},
    {"fix", {{"op::fix(%)", "_0"}}},
    {"floor", {{"op::floor(%)", "_0"}}},
    {"ceil", {{"op::ceil(%)", "_0"}}},
    {"trunc", {{"op::trunc(%)", "_0"}}},
    {"sign", {{"op::sign(%)", "_0"}}},
    {"reciprocal", {{"op::reciprocal(%)", "_0"}}},
    {"abs", {{"op::abs(%)", "_0"}}},
    {"gamma", {{"op::gamma(%)", "_0"}}},
    {"gammaln", {{"op::gammaln(%)", "_0"}}},
    {"erf", {{"op::erf(%)", "_0"}}},
    {"erfinv", {{"op::erfinv(%)", "_0"}}},
    {"_copy", {{"op::identity(%)", "_0"}}},
    {"_identity_with_attr_like_rhs", {{"op::identity(%)", "_0"}}},
    {"_plus_scalar", {{"op::add(%, float(%))", "_0", "scalar"}}},
    {"_PlusScalar", {{"op::add(%, float(%))", "_0", "scalar"}}},
    {"_minus_scalar", {{"op::sub(%, float(%))", "_0", "scalar"}}},
    {"_MinusScalar", {{"op::sub(%, float(%))", "_0", "scalar"}}},
    {"_rminus_scalar", {{"(-op::sub(%, float(%)))", "_0", "scalar"}}},
    {"_RMinusScalar", {{"(-op::sub(%, float(%)))", "_0", "scalar"}}},
    {"_mul_scalar", {{"op::mul(%, float(%))", "_0", "scalar"}}},
    {"_MulScalar", {{"op::mul(%, float(%))", "_0", "scalar"}}},
    {"_div_scalar", {{"op::mul(%, 1.0f/float(%))", "_0", "scalar"}}},
    {"_DivScalar", {{"op::mul(%, 1.0f/float(%))", "_0", "scalar"}}},
    {"_rdiv_scalar", {{"op::rdiv(%, float(%))", "_0", "scalar"}}},
    {"_power_scalar", {{"op::power(%, float(%))", "_0", "scalar"}}},
    {"_PowerScalar", {{"op::power(%, float(%))", "_0", "scalar"}}},
    {"_rpower_scalar", {{"op::rpow(%, float(%))", "_0", "scalar"}}},
    {"_RPowerScalar", {{"op::rpow(%, float(%))", "_0", "scalar"}}},
    {"_RDivScalar", {{"op::rdiv(%, float(%))", "_0", "scalar"}}},
    {"_mod_scalar", {{"op::mod(%, float(%))", "_0", "scalar"}}},
    {"_rmod_scalar", {{"op::rmod(%, float(%))", "_0", "scalar"}}},
    {"Cast", {{"op::cast<%>(%)", "dtype", "_0"}}},
    {"cast", {{"op::cast<%>(%)", "dtype", "_0"}}},
    {"Activation", {{"op::%(%)", "act_type", "_0"}}},
    {"clip", {{"op::clip(%, %, %)", "_0", "a_min", "a_max"}}},
    {"_zeros", {{"op::zero<%>()", "dtype"}}},
    {"_ones", {{"op::one<%>()", "dtype"}}},
    {"negative", {{"(-%)", "_0"}}},
    {"_hypot", {{"op::hypot(%, %)", "_0", "_1"}}},
    {"_hypot_scalar", {{"op::hypot(%, float(%))", "_0", "scalar"}}},
    {"logical_not", {{"op::logical_not(%)", "_0"}}},
    {"_backward_relu", {{"op::backward_relu(%, %)", "_0", "_1"}}},
    {"_backward_sigmoid", {{"op::backward_sigmoid(%, %)", "_0", "_1"}}},
    {"_backward_log_sigmoid", {{"op::backward_log_sigmoid(%, %)", "_0", "_1"}}},
    {"_backward_mish", {{"op::backward_mish(%, %)", "_0", "_1"}}},
    {"_backward_expm1", {{"op::backward_expm1(%, %)", "_0", "_1"}}},
    {"_backward_log", {{"op::backward_log(%, %)", "_0", "_1"}}},
    {"_backward_log10", {{"op::backward_log10(%, %)", "_0", "_1"}}},
    {"_backward_log2", {{"op::backward_log2(%, %)", "_0", "_1"}}},
    {"_backward_log1p", {{"op::backward_log1p(%, %)", "_0", "_1"}}},
    {"_backward_sin", {{"op::backward_sin(%, %)", "_0", "_1"}}},
    {"_backward_cos", {{"op::backward_cos(%, %)", "_0", "_1"}}},
    {"_backward_tan", {{"op::backward_tan(%, %)", "_0", "_1"}}},
    {"_backward_arcsin", {{"op::backward_arcsin(%, %)", "_0", "_1"}}},
    {"_backward_arccos", {{"op::backward_arccos(%, %)", "_0", "_1"}}},
    {"_backward_arctan", {{"op::backward_arctan(%, %)", "_0", "_1"}}},
    {"_backward_sinh", {{"op::backward_sinh(%, %)", "_0", "_1"}}},
    {"_backward_cosh", {{"op::backward_cosh(%, %)", "_0", "_1"}}},
    {"_backward_tanh", {{"op::backward_tanh(%, %)", "_0", "_1"}}},
    {"_backward_arcsinh", {{"op::backward_arcsinh(%, %)", "_0", "_1"}}},
    {"_backward_arccosh", {{"op::backward_arccosh(%, %)", "_0", "_1"}}},
    {"_backward_arctanh", {{"op::backward_arctanh(%, %)", "_0", "_1"}}},
    {"_backward_sqrt", {{"op::backward_sqrt(%, %)", "_0", "_1"}}},
    {"_backward_rsqrt", {{"op::backward_rsqrt(%, %)", "_0", "_1"}}},
    {"_backward_cbrt", {{"op::backward_cbrt(%, %)", "_0", "_1"}}},
    {"_backward_rcbrt", {{"op::backward_rcbrt(%, %)", "_0", "_1"}}},
    {"_backward_square", {{"op::backward_square(%, %)", "_0", "_1"}}},
    {"_backward_div_scalar", {{"(% * 1.0f/float(%))", "_0", "scalar"}}},
    {"_backward_div_scalar", {{"(% * 1.0f/float(%))", "_0", "scalar"}}},
    {"_backward_rdiv_scalar", {{"op::rdiv_grad(%, %) * %", "_1", "scalar", "_0"}}},
    {"_backward_hypot_scalar", {{"(% * % / op::hypot(%, float(%)))", "_0", "_1", "_1", "scalar"}}},
    {"_backward_radians", {{"op::radians(%)", "_0"}}},
    {"_backward_erf", {{"op::backward_erf(%, %)", "_0", "_1"}}},
    {"_backward_erfinv", {{"op::backward_erfinv(%, %)", "_0", "_1"}}},
    {"_backward_reciprocal", {{"op::backward_reciprocal(%, %)", "_0", "_1"}}},
    {"_backward_abs", {{"(op::backward_abs(%, %))", "_0", "_1"}}},
    {"_backward_degrees", {{"op::degrees(%)", "_0"}}},
    {"_backward_clip", {{"op::backward_clip(%, %, %, %)", "_0", "_1", "a_min", "a_max"}}},
    {"smooth_l1", {{"op::smooth_l1(%, float(%))", "_0", "scalar"}}},
    {"_backward_smooth_l1", {{"op::smooth_l1_grad(%, float(%)) * %", "_1", "scalar", "_0"}}},
    // TODO(ptredak): arange
    // TODO(ptredak): LeakyRelu
    {"_backward_sub", {{"(%)", "_0"}, {"(-(%))", "_0"}}},
    {"_backward_mul", {{"(% * %)", "_0", "_2"}, {"(% * %)", "_0", "_1"}}},
    {"_backward_mul_scalar", {{"(% * float(%))", "_0", "scalar"}}},
    {"_backward_div", {{"(% / %)", "_0", "_2"}, {"(-% * % / (% * %))", "_0", "_1", "_2", "_2"}}},
    {"_backward_power",
     {{"(% * % * powf(%, % - 1))", "_0", "_2", "_1", "_2"},
      {"(% * powf(%, %) * logf(%))", "_0", "_1", "_2", "_1"}}},
    {"_backward_power_scalar",
     {{"(% * float(%) * powf(%, float(%) - 1))", "_0", "scalar", "_1", "scalar"}}},
    {"_backward_rpower_scalar", {{"(% * % * logf(float(%)))", "_0", "_1", "scalar"}}},
    {"_backward_maximum",
     {{"((% >= %) ? % : 0)", "_1", "_2", "_0"}, {"((% >= %) ? 0 : %)", "_1", "_2", "_0"}}},
    {"_backward_minimum",
     {{"((% <= %) ? % : 0)", "_1", "_2", "_0"}, {"((% <= %) ? 0 : %)", "_1", "_2", "_0"}}},
    {"_backward_hypot",
     {{"(% * % / op::hypot(%, %))", "_0", "_1", "_1", "_2"},
      {"(% * % / op::hypot(%, %))", "_0", "_2", "_1", "_2"}}}};

// LeakyReLU ops: based on "act_type" attribute
const std::map<std::string, std::vector<std::vector<std::string>>> LeakyReLU_ops = {
    {"gelu_erf", {{"op::gelu_erf(%)", "_0"}}},
};
const std::map<std::string, std::vector<std::vector<std::string>>> LeakyReLU_bwd_ops = {
    {"gelu_erf", {{"op::backward_gelu_erf(%, %)", "_0", "_1"}}},
};

const std::map<std::string, std::string> slice_ops = {
    {"slice_axis", ""},
    {"slice", ""},
    {"slice_like", ""},
    {"broadcast_like", ""},
};

const std::vector<std::string> variable_io_ops = {"add_n",
                                                  "_backward_Activation",
                                                  "amp_multicast",
                                                  "_backward_amp_multicast",
                                                  "_backward_cast"};

const char kernel_begin[] = R"code(
const int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < N; i+= gridDim.x * blockDim.x) {
    int offset = i*nvec;
)code";

const char kernel_end[] = R"code(}
}
)code";

}  // namespace fusion

}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
