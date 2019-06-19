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

namespace detail {

const char fp16_support_string[] = R"code(
struct __align__(2) __half {
  __host__ __device__ __half() { }
  unsigned short __x;
};
/* Definitions of intrinsics */
__device__ inline __half __float2half(const float f) {
  __half val;
 asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.__x) : "f"(f));
  return val;
}
__device__ inline float __half2float(const __half h) {
  float val;
 asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h.__x));
  return val;
}
typedef __half half;
)code";

const char type_support_string[] = R"code(
using float32 = float;
using float64 = double;
using float16 = half;
using uint8 = unsigned char;
using int8 = char;
using int32 = int;
using int64 = long long;
)code";

const std::map<std::string, std::string> fused_op_binary_ops = {
  {"elemwise_add", "add"},
  {"_plus"       , "add"},
  {"_Plus"       , "add"},
  {"_add"        , "add"},
  {"elemwise_sub", "sub"},
  {"_minus"      , "sub"},
  {"_Minus"      , "sub"},
  {"_sub"        , "sub"},
  {"elemwise_mul", "mul"},
  {"_mul"        , "mul"},
  {"_Mul"        , "mul"},
  {"elemwise_div", "div"},
  {"_div"        , "div"},
  {"_Div"        , "div"},
  {"_Power"      , "power"},
  {"_power"      , "power"},
  {"_Maximum"    , "max"},
  {"_maximum"    , "max"},
  {"_Minimum"    , "min"},
  {"_minimum"    , "min"}
};

const std::map<std::string, std::string> fused_op_unary_ops = {
  {"amp_cast"                          , "identity"},
  {"relu"                              , "relu"},
  {"sigmoid"                           , "sigmoid"},
  {"softsign"                          , "softsign"},
  {"exp"                               , "exp"},
  {"expm1"                             , "expm1"},
  {"log"                               , "log"},
  {"log10"                             , "log10"},
  {"log2"                              , "log2"},
  {"log1p"                             , "log1p"},
  {"degrees"                           , "degrees"},
  {"radians"                           , "radians"},
  {"sin"                               , "sin"},
  {"cos"                               , "cos"},
  {"tan"                               , "tan"},
  {"arcsin"                            , "arcsin"},
  {"arccos"                            , "arccos"},
  {"arccos"                            , "arccos"},
  {"arctan"                            , "arctan"},
  {"sinh"                              , "sinh"},
  {"cosh"                              , "cosh"},
  {"tanh"                              , "tanh"},
  {"arcsinh"                           , "arcsinh"},
  {"arccosh"                           , "arccosh"},
  {"arctanh"                           , "arctanh"},
  {"sqrt"                              , "sqrt"},
  {"rsqrt"                             , "rsqrt"},
  {"cbrt"                              , "cbrt"},
  {"rcbrt"                             , "rcbrt"},
  {"square"                            , "square"},
  {"squeeze"                           , "identity"},
  {"zeros_like"                        , "zero"},
  {"ones_like"                         , "one"},
  {"flatten"                           , "identity"},
  {"Reshape"                           , "identity"},
  {"reshape"                           , "identity"},
  {"expand_dims"                       , "identity"},
  {"round"                             , "round"},
  {"rint"                              , "rint"},
  {"fix"                               , "fix"},
  {"floor"                             , "floor"},
  {"ceil"                              , "ceil"},
  {"trunc"                             , "trunc"},
  {"sign"                              , "sign"},
  {"reciprocal"                        , "reciprocal"},
  {"abs"                               , "abs"},
  {"gamma"                             , "gamma"},
  {"gammaln"                           , "gammaln"},
  {"erf"                               , "erf"},
  {"erfinv"                            , "erfinv"},
  {"_copy"                             , "identity"},
  {"_identity_with_attr_like_rhs"      , "identity"}
};

const std::map<std::string, std::vector<std::string>> fused_op_special_ops = {
  {"_plus_scalar", {"add(%, %)", "_0", "scalar"}},
  {"_PlusScalar", {"add(%, %)", "_0", "scalar"}},
  {"_minus_scalar", {"sub(%, %)", "_0", "scalar"}},
  {"_MinusScalar", {"sub(%, %)", "_0", "scalar"}},
  {"_rminus_scalar", {"(-sub(%, %))", "_0", "scalar"}},
  {"_RMinusScalar", {"(-sub(%, %))", "_0", "scalar"}},
  {"_mul_scalar", {"mul(%, %)", "_0", "scalar"}},
  {"_MulScalar", {"mul(%, %)", "_0", "scalar"}},
  {"_div_scalar", {"div(%, %)", "_0", "scalar"}},
  {"_DivScalar", {"div(%, %)", "_0", "scalar"}},
  {"_rdiv_scalar", {"rdiv(%, %)", "_0", "scalar"}},
  {"_power_scalar", {"power(%, %)", "_0", "scalar"}},
  {"_PowerScalar", {"power(%, %)", "_0", "scalar"}},
  {"_rpower_scalar", {"rpow(%, %)", "_0", "scalar"}},
  {"_RPowerScalar", {"rpow(%, %)", "_0", "scalar"}},
  {"_RDivScalar", {"rdiv(%, %)", "_0", "scalar"}},
  {"Cast", {"cast<%>(%)", "dtype", "_0"}},
  {"cast", {"cast<%>(%)", "dtype", "_0"}},
  {"Activation", {"%(%)", "act_type", "_0"}},
  {"clip", {"clip(%, %, %)", "_0", "a_min", "a_max"}},
  {"_zeros", {"zero<%>(0)", "dtype"}},
  {"_ones", {"one<%>(0)", "dtype"}},
  {"negative", {"(-%)", "_0"}},
  {"_hypot", {"hypot(%, %)", "_0", "_1"}},
  {"_hypot_scalar", {"hypot(%, %)", "_0", "scalar"}},
  {"_backward_relu", {"backward_relu(%, %)", "_1", "_0"}},
  {"_backward_sigmoid", {"backward_sigmoid(%, %)", "_1", "_0"}},
  {"_backward_expm1", {"backward_expm1(%, %)", "_1", "_0"}},
  {"_backward_log", {"backward_log(%, %)", "_1", "_0"}},
  {"_backward_log10", {"backward_log10(%, %)", "_1", "_0"}},
  {"_backward_log2", {"backward_log2(%, %)", "_1", "_0"}},
  {"_backward_log1p", {"backward_log1p(%, %)", "_1", "_0"}},
  {"_backward_sin", {"backward_sin(%, %)", "_1", "_0"}},
  {"_backward_cos", {"backward_cos(%, %)", "_1", "_0"}},
  {"_backward_tan", {"backward_tan(%, %)", "_1", "_0"}},
  {"_backward_arcsin", {"backward_arcsin(%, %)", "_1", "_0"}},
  {"_backward_arccos", {"backward_arccos(%, %)", "_1", "_0"}},
  {"_backward_arctan", {"backward_arctan(%, %)", "_1", "_0"}},
  {"_backward_sinh", {"backward_sinh(%, %)", "_1", "_0"}},
  {"_backward_cosh", {"backward_cosh(%, %)", "_1", "_0"}},
  {"_backward_tanh", {"backward_tanh(%, %)", "_1", "_0"}},
  {"_backward_arcsinh", {"backward_arcsinh(%, %)", "_1", "_0"}},
  {"_backward_arccosh", {"backward_arccosh(%, %)", "_1", "_0"}},
  {"_backward_arctanh", {"backward_arctanh(%, %)", "_1", "_0"}},
  {"_backward_sqrt", {"backward_sqrt(%, %)", "_1", "_0"}},
  {"_backward_rsqrt", {"backward_rsqrt(%, %)", "_1", "_0"}},
  {"_backward_cbrt", {"backward_cbrt(%, %)", "_1", "_0"}},
  {"_backward_rcbrt", {"backward_rcbrt(%, %)", "_1", "_0"}},
  {"_backward_square", {"backward_square(%, %)", "_1", "_0"}},
  {"_backward_div_scalar", {"(% / %)", "_0", "scalar"}},
  {"_backward_div_scalar", {"(% / %)", "_0", "scalar"}},
  {"_backward_rdiv_scalar", {"(-% * % / (% * %))", "_0", "scalar", "_1", "_1"}},
  {"_backward_hypot_scalar", {"(% * % / hypot(%, %))", "_0", "_1", "_1", "scalar"}},
  {"_backward_radians", {"radians(%)", "_0"}},
  {"_backward_erf", {"backward_erf(%, %)", "_1", "_0"}},
  {"_backward_erfinv", {"backward_erfinv(%, %)", "_1", "_0"}}
  // TODO(ptredak): arange
};

// Multiple inputs/multiple outputs
const std::map<std::string, std::vector<std::vector<std::string>>> fused_op_mimo_ops = {
  {"_backward_sub", {{"(%)", "_0"},
                     {"(-(%))", "_0"}}},
  {"_backward_mul", {{"(% * %)", "_0", "_2"},
                     {"(% * %)", "_0", "_1"}}},
  {"_backward_mul_scalar", {{"(% * %)", "_0", "scalar"}}},
  {"_backward_div", {{"(% / %)", "_0", "_2"},
                     {"(-% * % / (% * %))", "_0", "_1", "_2", "_2"}}},
  {"_backward_power", {{"(% * % * powf(%, % - 1))", "_0", "_2", "_1", "_2"},
                       {"(% * powf(%, %) * logf(%))", "_0", "_1", "_2", "_1"}}},
  {"_backward_power_scalar", {{"(% * % * powf(%, % - 1))", "_0", "scalar", "_1", "scalar"}}},
  {"_backward_rpower_scalar", {{"(% * % * logf(%))", "_0", "_1", "scalar"}}},
  {"_backward_maximum", {{"((% >= %) ? % : 0)", "_1", "_2", "_0"},
                         {"((% >= %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_minimum", {{"((% <= %) ? % : 0)", "_1", "_2", "_0"},
                         {"((% <= %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_hypot", {{"(% * % / hypot(%, %))", "_0", "_1", "_1", "_2"},
                       {"(% * % / hypot(%, %))", "_0", "_2", "_1", "_2"}}}
};

const std::map<std::string, std::string> fused_op_slice_ops = {
  {"slice_axis"   , ""},
};

const std::vector<std::string> fused_op_variable_io_ops = {
  "add_n",
  "_backward_Activation"
};

const char fused_op_function_definitions[] = R"code(

template <class T>
struct remove_pointer;

template <class U>
struct remove_pointer<U*>
{
  typedef U type;
};

template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
inline typename LoadType<DType>::Type load(const DType input) {
  return input;
}

template <>
inline float load(const half input) {
  return __half2float(input);
}

template <typename DType1, typename DType2>
inline DType1 store(const DType2 input, DType1* ref) {
  return input;
}

template<>
inline half store(const float input, half* ref) {
  return __float2half(input);
}



template <int size>
struct VectorConfig {
    static_assert(size >= 4, "Error");
    using IndexType = float;
};

template <>
struct VectorConfig<8> {
    using IndexType = double;
};

template <>
struct VectorConfig<16> {
    using IndexType = double2;
};

template <typename DType, int nvec>
union VectorType {
    typename VectorConfig<sizeof(DType)*nvec>::IndexType y;
    DType x[nvec];
    VectorType () {};
    VectorType (const VectorType<DType, nvec>& y2) {
        y = y2.y;
    }
    VectorType (const decltype(y) &y2) {
        y = y2;
    }
}; 

template <int ndim>
struct Strides {
   int x[ndim];
};

template <int nvec, typename DType>
inline VectorType<DType, nvec> load_index(const DType * input, int i) {
  const auto* vector_input = reinterpret_cast<const typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(input + i);
  VectorType<DType, nvec> ret = {*vector_input};
  return ret;
}

template <int nvec, int axis, typename DType, int ndim>
inline VectorType<DType, nvec> load_slice(const DType * input, const Strides<ndim> strides, int begin, int end, int offset) {
  int idx[nvec];
  bool mem_aligned = true;

  Strides<ndim> ref_strides;
  if (axis > 0) {
      int shape = strides.x[axis-1]/strides.x[axis];
      if (begin < 0) begin = shape - begin;
      if (end < 0) begin = shape - begin;
      if (end > shape) end = shape;
      #pragma unroll
      for (int dim = 0; dim < axis; dim++) {
          ref_strides.x[dim] = (strides.x[dim] / shape) * (end-begin);
      }
  }
  #pragma unroll
  for (int dim = axis; dim < ndim; dim++) {
      ref_strides.x[dim] = strides.x[dim];
  }

  #pragma unroll
  for (int j = 0; j < nvec; j++) {
    idx[j] = 0;
    int ref_idx = offset + j;
    #pragma unroll
    for (int dim = 0; dim < ndim; dim++) {
       int stride = ref_strides.x[dim];
       idx[j] += (ref_idx / stride) * strides.x[dim];
       ref_idx = ref_idx % stride;
    }
    idx[j] += begin * strides.x[axis];
    if (j > 0 && (idx[j] != (idx[j-1] + 1))) {
        mem_aligned = false;
    }
  }
  mem_aligned = mem_aligned && ((idx[0] % nvec) == 0);
  if (!mem_aligned) {
    VectorType<DType, nvec> ret;
    #pragma unroll
    for (int j = 0; j < nvec; j++) {
        ret.x[j] = *(input + idx[j]);
    }
    return ret;
  }
  return load_index<nvec>(input, idx[0]);
}



template <int nvec, typename DType>
inline void store_index(const VectorType<DType, nvec> value, int i, DType * output) {
  auto vector_output = reinterpret_cast<typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
  vector_output[i] = value.y;
}

template <int nvec, typename DType>
inline void store_add_index(const VectorType<DType, nvec> value, int i, DType * output) {
  auto vector_output = reinterpret_cast<typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
  vector_output[i] += value.y;
}

template <typename DType>
inline DType identity(const DType val) {
  return val;
}

template <typename DType, typename DType2>
inline DType add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
inline DType sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
inline DType mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
inline DType div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
inline DType rdiv(const DType a, const DType2 b) {
  return b / a;
}

template <typename DType, typename DType2>
inline DType power(const DType a, const DType2 b) {
  return powf(a, b);
}

template <typename DType, typename DType2>
inline DType rpow(const DType a, const DType2 b) {
  return powf(b, a);
}

template <typename DType, typename DType2>
inline DType max(const DType a, const DType2 b) {
  return a > b ? a : b;
}

template <typename DType, typename DType2>
inline DType min(const DType a, const DType2 b) {
  return a < b ? a : b;
}

template <typename DType, typename DType2>
inline DType hypot(const DType a, const DType2 b) {
  return hypotf(a, b);
}

template <typename OutType, typename DType>
inline typename LoadType<OutType>::Type cast(const DType val) {
  return static_cast<typename LoadType<OutType>::Type>(val);
}

// TODO(ptredak): this is not exactly identity, needs type inference
// in the middle of the graph to do it right
template <typename DType>
inline DType amp_multicast(const DType val) {
  return val;
}

// activations

template <typename DType>
inline DType relu(const DType val) {
  return val > 0 ? val : 0;
}

template <typename DType>
inline DType backward_relu(const DType val, const DType grad) {
  return val > 0 ? grad : 0;
}

template <typename DType>
inline DType sigmoid(const DType val) {
  return 1.f/(1 + expf(-val));
}

template <typename DType>
inline DType backward_sigmoid(const DType out, const DType grad) {
  return grad * out * (1 - out);
}

template <typename DType>
inline DType softrelu(const DType val) {
  return logf(1 + expf(val));
}

template <typename DType>
inline DType backward_softrelu(const DType val, const DType grad) {
  return grad * sigmoid(val);
}

template <typename DType>
inline DType softsign(const DType val) {
  return val / (1 + fabsf(val));
}

template <typename DType>
inline DType backward_softsign(const DType val, const DType grad) {
  const DType ap1 = 1 + fabsf(val);
  return grad / (ap1 * ap1);
}

// exp and log

template <typename DType>
inline DType exp(const DType val) {
  return expf(val);
}

template <typename DType>
inline DType backward_exp(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType expm1(const DType val) {
  return expm1f(val);
}

template <typename DType>
inline DType backward_expm1(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType log(const DType val) {
  return logf(val);
}

template <typename DType>
inline DType backward_log(const DType val, const DType grad) {
  return grad / val;
}

template <typename DType>
inline DType log10(const DType val) {
  return log10f(val);
}

template <typename DType>
inline DType backward_log10(const DType val, const DType grad) {
  return grad / (val * logf(10));
}

template <typename DType>
inline DType log2(const DType val) {
  return log2f(val);
}

template <typename DType>
inline DType backward_log2(const DType val, const DType grad) {
  return grad / (val * logf(2));
}

template <typename DType>
inline DType log1p(const DType val) {
  return log1pf(val);
}

template <typename DType>
inline DType backward_log1p(const DType val, const DType grad) {
  return grad / (1 + val);
}

// trigonometric

constexpr double pi = 3.14159265358979323846;

template <typename DType>
inline DType degrees(const DType val) {
  return (val / pi) * 180;
}

template <typename DType>
inline DType radians(const DType val) {
  return (val / 180.0) * pi;
}

template <typename DType>
inline DType sin(const DType val) {
  return sinf(val);
}

template <typename DType>
inline DType backward_sin(const DType val, const DType grad) {
  return grad * cosf(val);
}

template <typename DType>
inline DType cos(const DType val) {
  return cosf(val);
}

template <typename DType>
inline DType backward_cos(const DType val, const DType grad) {
  return -grad * sinf(val);
}

template <typename DType>
inline DType tan(const DType val) {
  return tanf(val);
}

// Uses output from tan
template <typename DType>
inline DType backward_tan(const DType out, const DType grad) {
  return grad * (out * out + 1);
}

template <typename DType>
inline DType arcsin(const DType val) {
  return asinf(val);
}

template <typename DType>
inline DType backward_arcsin(const DType val, const DType grad) {
  return grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arccos(const DType val) {
  return acosf(val);
}

template <typename DType>
inline DType backward_arccos(const DType val, const DType grad) {
  return -grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arctan(const DType val) {
  return atanf(val);
}

template <typename DType>
inline DType backward_arctan(const DType val, const DType grad) {
  return grad / (1 + val*val);
}

template <typename DType>
inline DType sinh(const DType val) {
  return sinhf(val);
}

template <typename DType>
inline DType backward_sinh(const DType val, const DType grad) {
  return grad * coshf(val);
}

template <typename DType>
inline DType cosh(const DType val) {
  return coshf(val);
}

template <typename DType>
inline DType backward_cosh(const DType val, const DType grad) {
  return grad * sinhf(val);
}

template <typename DType>
inline DType tanh(const DType val) {
  return tanhf(val);
}

// Uses tanh output
template <typename DType>
inline DType backward_tanh(const DType out, const DType grad) {
  return grad * (1 - out * out);
}

template <typename DType>
inline DType arcsinh(const DType val) {
  return asinhf(val);
}

template <typename DType>
inline DType backward_arcsinh(const DType val, const DType grad) {
  return grad / sqrtf(val * val + 1);
}

template <typename DType>
inline DType arccosh(const DType val) {
  return acoshf(val);
}

template <typename DType>
inline DType backward_arccosh(const DType val, const DType grad) {
  return grad / sqrtf(val * val - 1);
}

template <typename DType>
inline DType arctanh(const DType val) {
  return atanhf(val);
}

template <typename DType>
inline DType backward_arctanh(const DType val, const DType grad) {
  return grad / (1 - val * val);
}

// sqrt

template <typename DType>
inline DType sqrt(const DType val) {
  return sqrtf(val);
}

template <typename DType>
inline DType backward_sqrt(const DType out, const DType grad) {
  return 0.5 * grad / out;
}

template <typename DType>
inline DType rsqrt(const DType val) {
  return rsqrtf(val);
}

template <typename DType>
inline DType backward_rsqrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -0.5 * grad * sqrtf(inv) * inv;
}

template <typename DType>
inline DType cbrt(const DType val) {
  return cbrtf(val);
}

template <typename DType>
inline DType backward_cbrt(const DType out, const DType grad) {
  return grad / (3.0f * out * out);
}

template <typename DType>
inline DType rcbrt(const DType val) {
  return rcbrtf(val);
}

template <typename DType>
inline DType backward_rcbrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * cbrtf(inv) * inv;
}

template <typename DType>
inline DType square(const DType val) {
  return val * val;
}

template <typename DType>
inline DType backward_square(const DType val, const DType grad) {
  return 2 * val * grad;
}

template <typename DType>
inline typename LoadType<DType>::Type zero(const DType val) {
  return 0;
}

template <typename DType>
inline typename LoadType<DType>::Type one(const DType val) {
  return 1;
}

template <typename DType>
inline DType round(const DType val) {
  return roundf(val);
}

template <typename DType>
inline DType rint(const DType val) {
  return rintf(val);
}

template <typename DType>
inline DType fix(const DType val) {
    const auto floor = floorf(val);
    const auto ceil = ceilf(val);
    return (floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil;
}

template <typename DType>
inline DType floor(const DType val) {
    return floorf(val);
}

template <typename DType>
inline DType ceil(const DType val) {
    return ceilf(val);
}

template <typename DType>
inline DType trunc(const DType val) {
    return truncf(val);
}

template <typename DType>
inline DType clip(const DType val, const float a_min, const float a_max) {
  return max(min(val, a_max), a_min);
}

template <typename DType>
inline DType sign(const DType val) {
  if (val < 0) return -1;
  return val > 0 ? 1 : 0;
}

template <typename DType>
inline DType reciprocal(const DType val) {
  return 1.0f / val;
}

template <typename DType>
inline DType abs(const DType val) {
  return fabsf(val);
}

template <typename DType>
inline DType gamma(const DType val) {
  return tgammaf(val);
}

template <typename DType>
inline DType gammaln(const DType val) {
  return lgammaf(val);
}

template <typename DType>
inline DType erf(const DType val) {
  return erff(val);
}

template <typename DType>
inline DType backward_erf(const DType val, const DType grad) {
  return 2.0f / sqrt(pi) * exp(-(val*val)) * grad;
}

template <typename DType>
inline DType erfinv(const DType val) {
  return erfinvf(val);
}

template <typename DType>
inline DType backward_erfinv(const DType val, const DType grad) {
  return 0.5f * sqrt(pi) * exp(val * val) * grad;
}

)code";

const char fused_op_kernel_begin[] = R"code(
const int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < N; i+= gridDim.x * blockDim.x) {
    int offset = i*nvec;

)code";

const char fused_op_kernel_end[] = R"code(
}
}
)code";

}  // namespace detail

}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
