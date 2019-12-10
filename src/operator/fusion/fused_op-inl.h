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

#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

namespace mxnet {

namespace fusion {

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

const std::map<std::string, std::vector<std::vector<std::string>>> ops_desc = {
  {"elemwise_add"                      , {{"op::add(%, %)", "_0", "_1"}}},
  {"_plus"                             , {{"op::add(%, %)", "_0", "_1"}}},
  {"_Plus"                             , {{"op::add(%, %)", "_0", "_1"}}},
  {"_add"                              , {{"op::add(%, %)", "_0", "_1"}}},
  {"elemwise_sub"                      , {{"op::sub(%, %)", "_0", "_1"}}},
  {"_minus"                            , {{"op::sub(%, %)", "_0", "_1"}}},
  {"_Minus"                            , {{"op::sub(%, %)", "_0", "_1"}}},
  {"_sub"                              , {{"op::sub(%, %)", "_0", "_1"}}},
  {"elemwise_mul"                      , {{"op::mul(%, %)", "_0", "_1"}}},
  {"_mul"                              , {{"op::mul(%, %)", "_0", "_1"}}},
  {"_Mul"                              , {{"op::mul(%, %)", "_0", "_1"}}},
  {"elemwise_div"                      , {{"op::div(%, %)", "_0", "_1"}}},
  {"_div"                              , {{"op::div(%, %)", "_0", "_1"}}},
  {"_Div"                              , {{"op::div(%, %)", "_0", "_1"}}},
  {"_Power"                            , {{"op::power(%, %)", "_0", "_1"}}},
  {"_power"                            , {{"op::power(%, %)", "_0", "_1"}}},
  {"_Maximum"                          , {{"op::max(%, %)", "_0", "_1"}}},
  {"_maximum"                          , {{"op::max(%, %)", "_0", "_1"}}},
  {"_Minimum"                          , {{"op::min(%, %)", "_0", "_1"}}},
  {"_minimum"                          , {{"op::min(%, %)", "_0", "_1"}}},
  {"amp_cast"                          , {{"op::identity(%)", "_0"}}},
  {"_backward_amp_cast"                , {{"op::identity(%)", "_0"}}},
  {"relu"                              , {{"op::relu(%)", "_0"}}},
  {"sigmoid"                           , {{"op::sigmoid(%)", "_0"}}},
  {"softsign"                          , {{"op::softsign(%)", "_0"}}},
  {"exp"                               , {{"op::exp(%)", "_0"}}},
  {"expm1"                             , {{"op::expm1(%)", "_0"}}},
  {"log"                               , {{"op::log(%)", "_0"}}},
  {"log10"                             , {{"op::log10(%)", "_0"}}},
  {"log2"                              , {{"op::log2(%)", "_0"}}},
  {"log1p"                             , {{"op::log1p(%)", "_0"}}},
  {"degrees"                           , {{"op::degrees(%)", "_0"}}},
  {"radians"                           , {{"op::radians(%)", "_0"}}},
  {"sin"                               , {{"op::sin(%)", "_0"}}},
  {"cos"                               , {{"op::cos(%)", "_0"}}},
  {"tan"                               , {{"op::tan(%)", "_0"}}},
  {"arcsin"                            , {{"op::arcsin(%)", "_0"}}},
  {"arccos"                            , {{"op::arccos(%)", "_0"}}},
  {"arctan"                            , {{"op::arctan(%)", "_0"}}},
  {"sinh"                              , {{"op::sinh(%)", "_0"}}},
  {"cosh"                              , {{"op::cosh(%)", "_0"}}},
  {"tanh"                              , {{"op::tanh(%)", "_0"}}},
  {"arcsinh"                           , {{"op::arcsinh(%)", "_0"}}},
  {"arccosh"                           , {{"op::arccosh(%)", "_0"}}},
  {"arctanh"                           , {{"op::arctanh(%)", "_0"}}},
  {"sqrt"                              , {{"op::sqrt(%)", "_0"}}},
  {"rsqrt"                             , {{"op::rsqrt(%)", "_0"}}},
  {"cbrt"                              , {{"op::cbrt(%)", "_0"}}},
  {"rcbrt"                             , {{"op::rcbrt(%)", "_0"}}},
  {"square"                            , {{"op::square(%)", "_0"}}},
  {"squeeze"                           , {{"op::identity(%)", "_0"}}},
  {"zeros_like"                        , {{"op::zero(%)", "_0"}}},
  {"ones_like"                         , {{"op::one(%)", "_0"}}},
  {"flatten"                           , {{"op::identity(%)", "_0"}}},
  {"Reshape"                           , {{"op::identity(%)", "_0"}}},
  {"reshape"                           , {{"op::identity(%)", "_0"}}},
  {"_backward_reshape"                 , {{"op::identity(%)", "_0"}}},
  {"expand_dims"                       , {{"op::identity(%)", "_0"}}},
  {"round"                             , {{"op::round(%)", "_0"}}},
  {"rint"                              , {{"op::rint(%)", "_0"}}},
  {"fix"                               , {{"op::fix(%)", "_0"}}},
  {"floor"                             , {{"op::floor(%)", "_0"}}},
  {"ceil"                              , {{"op::ceil(%)", "_0"}}},
  {"trunc"                             , {{"op::trunc(%)", "_0"}}},
  {"sign"                              , {{"op::sign(%)", "_0"}}},
  {"reciprocal"                        , {{"op::reciprocal(%)", "_0"}}},
  {"abs"                               , {{"op::abs(%)", "_0"}}},
  {"gamma"                             , {{"op::gamma(%)", "_0"}}},
  {"gammaln"                           , {{"op::gammaln(%)", "_0"}}},
  {"erf"                               , {{"op::erf(%)", "_0"}}},
  {"erfinv"                            , {{"op::erfinv(%)", "_0"}}},
  {"_copy"                             , {{"op::identity(%)", "_0"}}},
  {"_identity_with_attr_like_rhs"      , {{"op::identity(%)", "_0"}}},
  {"_plus_scalar"                      , {{"op::add(%, float(%))", "_0", "scalar"}}},
  {"_PlusScalar"                       , {{"op::add(%, float(%))", "_0", "scalar"}}},
  {"_minus_scalar"                     , {{"op::sub(%, float(%))", "_0", "scalar"}}},
  {"_MinusScalar"                      , {{"op::sub(%, float(%))", "_0", "scalar"}}},
  {"_rminus_scalar"                    , {{"(-op::sub(%, float(%)))", "_0", "scalar"}}},
  {"_RMinusScalar"                     , {{"(-op::sub(%, float(%)))", "_0", "scalar"}}},
  {"_mul_scalar"                       , {{"op::mul(%, float(%))", "_0", "scalar"}}},
  {"_MulScalar"                        , {{"op::mul(%, float(%))", "_0", "scalar"}}},
  {"_div_scalar"                       , {{"op::mul(%, 1.0f/float(%))", "_0", "scalar"}}},
  {"_DivScalar"                        , {{"op::mul(%, 1.0f/float(%))", "_0", "scalar"}}},
  {"_rdiv_scalar"                      , {{"op::rdiv(%, float(%))", "_0", "scalar"}}},
  {"_power_scalar"                     , {{"op::power(%, float(%))", "_0", "scalar"}}},
  {"_PowerScalar"                      , {{"op::power(%, float(%))", "_0", "scalar"}}},
  {"_rpower_scalar"                    , {{"op::rpow(%, float(%))", "_0", "scalar"}}},
  {"_RPowerScalar"                     , {{"op::rpow(%, float(%))", "_0", "scalar"}}},
  {"_RDivScalar"                       , {{"op::rdiv(%, float(%))", "_0", "scalar"}}},
  {"Cast"                              , {{"op::cast<%>(%)", "dtype", "_0"}}},
  {"cast"                              , {{"op::cast<%>(%)", "dtype", "_0"}}},
  {"Activation"                        , {{"op::%(%)", "act_type", "_0"}}},
  {"clip"                              , {{"op::clip(%, %, %)", "_0", "a_min", "a_max"}}},
  {"_zeros"                            , {{"op::zero<%>()", "dtype"}}},
  {"_ones"                             , {{"op::one<%>()", "dtype"}}},
  {"negative"                          , {{"(-%)", "_0"}}},
  {"_hypot"                            , {{"op::hypot(%, %)", "_0", "_1"}}},
  {"_hypot_scalar"                     , {{"op::hypot(%, float(%))", "_0", "scalar"}}},
  {"_backward_relu"                    , {{"op::backward_relu(%, %)", "_1", "_0"}}},
  {"_backward_sigmoid"                 , {{"op::backward_sigmoid(%, %)", "_1", "_0"}}},
  {"_backward_expm1"                   , {{"op::backward_expm1(%, %)", "_1", "_0"}}},
  {"_backward_log"                     , {{"op::backward_log(%, %)", "_1", "_0"}}},
  {"_backward_log10"                   , {{"op::backward_log10(%, %)", "_1", "_0"}}},
  {"_backward_log2"                    , {{"op::backward_log2(%, %)", "_1", "_0"}}},
  {"_backward_log1p"                   , {{"op::backward_log1p(%, %)", "_1", "_0"}}},
  {"_backward_sin"                     , {{"op::backward_sin(%, %)", "_1", "_0"}}},
  {"_backward_cos"                     , {{"op::backward_cos(%, %)", "_1", "_0"}}},
  {"_backward_tan"                     , {{"op::backward_tan(%, %)", "_1", "_0"}}},
  {"_backward_arcsin"                  , {{"op::backward_arcsin(%, %)", "_1", "_0"}}},
  {"_backward_arccos"                  , {{"op::backward_arccos(%, %)", "_1", "_0"}}},
  {"_backward_arctan"                  , {{"op::backward_arctan(%, %)", "_1", "_0"}}},
  {"_backward_sinh"                    , {{"op::backward_sinh(%, %)", "_1", "_0"}}},
  {"_backward_cosh"                    , {{"op::backward_cosh(%, %)", "_1", "_0"}}},
  {"_backward_tanh"                    , {{"op::backward_tanh(%, %)", "_1", "_0"}}},
  {"_backward_arcsinh"                 , {{"op::backward_arcsinh(%, %)", "_1", "_0"}}},
  {"_backward_arccosh"                 , {{"op::backward_arccosh(%, %)", "_1", "_0"}}},
  {"_backward_arctanh"                 , {{"op::backward_arctanh(%, %)", "_1", "_0"}}},
  {"_backward_sqrt"                    , {{"op::backward_sqrt(%, %)", "_1", "_0"}}},
  {"_backward_rsqrt"                   , {{"op::backward_rsqrt(%, %)", "_1", "_0"}}},
  {"_backward_cbrt"                    , {{"op::backward_cbrt(%, %)", "_1", "_0"}}},
  {"_backward_rcbrt"                   , {{"op::backward_rcbrt(%, %)", "_1", "_0"}}},
  {"_backward_square"                  , {{"op::backward_square(%, %)", "_1", "_0"}}},
  {"_backward_div_scalar"              , {{"(% * 1.0f/float(%))", "_0", "scalar"}}},
  {"_backward_div_scalar"              , {{"(% * 1.0f/float(%))", "_0", "scalar"}}},
  {"_backward_rdiv_scalar"             , {{"(-% * float(%) / (% * %))", "_0",
                                           "scalar", "_1", "_1"}}},
  {"_backward_hypot_scalar"            , {{"(% * % / op::hypot(%, float(%)))",
                                           "_0", "_1", "_1", "scalar"}}},
  {"_backward_radians"                 , {{"op::radians(%)", "_0"}}},
  {"_backward_erf"                     , {{"op::backward_erf(%, %)", "_1", "_0"}}},
  {"_backward_erfinv"                  , {{"op::backward_erfinv(%, %)", "_1", "_0"}}},
  {"_backward_reciprocal"              , {{"op::backward_reciprocal(%, %)", "_1", "_0"}}},
  {"_backward_abs"                     , {{"(% * op::sign(%))", "_0", "_1"}}},
  {"_backward_degrees"                 , {{"op::degrees(%)", "_0"}}},
  {"_backward_sign"                    , {{"op::zero(%)", "_0"}}},
  {"_backward_clip"                    , {{"op::backward_clip(%, %, %, %)", "_1", "_0",
                                                                            "a_min", "a_max"}}},
  {"smooth_l1"                         , {{"op::smooth_l1(%, float(%))", "_0", "scalar"}}},
  {"_backward_smooth_l1"               , {{"op::backward_smooth_l1(%, float(%), %)",
                                           "_1", "scalar", "_0"}}},
  // TODO(ptredak): arange
  // TODO(ptredak): LeakyRelu
  // TODO(ptredak): mod and rmod
  {"_backward_sub"                     , {{"(%)", "_0"},
                                          {"(-(%))", "_0"}}},
  {"_backward_mul"                     , {{"(% * %)", "_0", "_2"},
                                          {"(% * %)", "_0", "_1"}}},
  {"_backward_mul_scalar"              , {{"(% * float(%))", "_0", "scalar"}}},
  {"_backward_div"                     , {{"(% / %)", "_0", "_2"},
                                          {"(-% * % / (% * %))", "_0", "_1", "_2", "_2"}}},
  {"_backward_power"                   , {{"(% * % * powf(%, % - 1))", "_0", "_2", "_1", "_2"},
                                          {"(% * powf(%, %) * logf(%))", "_0", "_1", "_2", "_1"}}},
  {"_backward_power_scalar"            , {{"(% * float(%) * powf(%, float(%) - 1))",
                                           "_0", "scalar", "_1", "scalar"}}},
  {"_backward_rpower_scalar"           , {{"(% * % * logf(float(%)))", "_0", "_1", "scalar"}}},
  {"_backward_maximum"                 , {{"((% >= %) ? % : 0)", "_1", "_2", "_0"},
                                          {"((% >= %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_minimum"                 , {{"((% <= %) ? % : 0)", "_1", "_2", "_0"},
                                          {"((% <= %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_hypot"                   , {{"(% * % / op::hypot(%, %))", "_0", "_1", "_1", "_2"},
                                          {"(% * % / op::hypot(%, %))", "_0", "_2", "_1", "_2"}}}
};

const std::map<std::string, std::string> slice_ops = {
  {"slice_axis"   , ""},
  {"slice"   , ""},
  {"slice_like"   , ""},
  {"broadcast_like"   , ""},
};

const std::vector<std::string> variable_io_ops = {
  "add_n",
  "_backward_Activation",
  "amp_multicast",
  "_backward_amp_multicast",
  "_backward_cast"
};

const char function_definitions[] = R"code(

#define INT_MAX (2147483647)

namespace op {

template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
__device__ inline typename LoadType<DType>::Type load(const DType input) {
  return input;
}

template <>
__device__ inline float load(const half input) {
  return __half2float(input);
}

template <typename DType1, typename DType2>
__device__ inline DType1 store(const DType2 input, DType1* ref) {
  return input;
}

template <typename DType>
__device__ inline half store(const DType input, half* ref) {
  return __float2half(input);
}

template <int size>
struct VectorConfig {
    static_assert(size >= 4, "VectorConfig needs to have size of at least 4B");
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

template <>
struct VectorConfig<32> {
    using IndexType = double4;
};

template <typename DType>
__device__ inline DType add_elem(const DType& x, const DType& y) {
  return x + y;
}

template <>
__device__ inline half add_elem(const half& x, const half& y) {
  return __float2half(__half2float(x) + __half2float(y));
}

template <typename DType, int nvec>
union VectorType {
    typename VectorConfig<sizeof(DType)*nvec>::IndexType y;
    DType x[nvec];
    __device__ VectorType () {};
    __device__ VectorType (const VectorType<DType, nvec>& y2) {
        y = y2.y;
    }
    __device__ VectorType (const decltype(y) &y2) {
        y = y2;
    }
    __device__ inline VectorType<DType, nvec>& operator+=(const VectorType<DType, nvec>& rhs) {
      #pragma unroll
      for (int i = 0; i < nvec; ++i) {
        x[i] = add_elem(x[i], rhs.x[i]);
      }
      return *this;
    }
};

template <int ndim>
struct Shape {
   int x[ndim];
   size_t size;
   __device__ inline const int& operator [](const int i) const {
       return x[i];
   }
   __device__ inline int& operator [](const int i) {
       return x[i];
   }
   __device__ inline void set(const int def) {
       #pragma unroll
       for (int i = 0; i < ndim; i++) {
           x[i] = def;
       }
   }
};

template <>
struct Shape<0> {
   size_t size;
};

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> load_index(const DType * input, int i,
                                                     const Shape<ndim> &shape) {
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<
                                const typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(
                                    input + i);
    VectorType<DType, nvec> ret = {*vector_input};
    return ret;
  } else {
    VectorType<DType, nvec> ret({0});
    return ret;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> global_load_index(const DType * input, int i,
                                                            const Shape<ndim> &shape) {
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<
                                const typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(
                                    input + i);
    VectorType<DType, nvec> ret = {__ldg(vector_input)};
    return ret;
  } else {
    VectorType<DType, nvec> ret({0});
    return ret;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> load_slice(const DType * input, const Shape<ndim>& shape,
                                                     Shape<ndim> begin, Shape<ndim> end,
                                                     int offset) {
  int idx[nvec];

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] - begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] - end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
    idx[j] = 0;
    int ref_idx = offset + j;
    #pragma unroll
    for (int dim = 0; dim < ndim; dim++) {
       int stride = ref_strides[dim];
       if (shape[dim] > 1) {
         idx[j] += (ref_idx / stride + begin[dim]) * strides[dim];
       }
       ref_idx = ref_idx % stride;
    }
  }
  VectorType<DType, nvec> ret;
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
      ret.x[j] = *(input + idx[j]);
  }
  return ret;
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> fast_load_slice(const DType * input,
                                                          const Shape<ndim>& shape,
                                                          Shape<ndim> begin,
                                                          Shape<ndim> end,
                                                          int offset) {
  int idx = 0;

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] - begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] - end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  int ref_idx = offset;
  #pragma unroll
  for (int dim = 0; dim < ndim; dim++) {
     int stride = ref_strides[dim];
     if (shape[dim] > 1) {
       idx += (ref_idx / stride + begin[dim]) * strides[dim];
     }
     ref_idx = ref_idx % stride;
  }
  return global_load_index<nvec>(input, idx, shape);
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_index(const VectorType<DType, nvec> value, int i,
                        DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
    vector_output[i] = value.y;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_add_index(const VectorType<DType, nvec> value, int i,
                            DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
    VectorType<DType, nvec> ret(vector_output[i]);
    ret += value;
    vector_output[i] = ret.y;
  }
}

template <typename DType>
__device__ inline DType identity(const DType val) {
  return val;
}

template <typename DType, typename DType2>
__device__ inline DType add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
__device__ inline DType sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
__device__ inline DType mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
__device__ inline DType div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
__device__ inline DType rdiv(const DType a, const DType2 b) {
  return b / a;
}

template <typename DType, typename DType2>
__device__ inline DType power(const DType a, const DType2 b) {
  return powf(a, b);
}

template <typename DType, typename DType2>
__device__ inline DType rpow(const DType a, const DType2 b) {
  return powf(b, a);
}

template <typename DType, typename DType2>
__device__ inline DType max(const DType a, const DType2 b) {
  return a > b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline DType min(const DType a, const DType2 b) {
  return a < b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline DType hypot(const DType a, const DType2 b) {
  return hypotf(a, b);
}

template <typename OutType, typename DType>
__device__ inline typename LoadType<OutType>::Type cast(const DType val) {
  return static_cast<typename LoadType<OutType>::Type>(val);
}

// activations

template <typename DType>
__device__ inline DType relu(const DType val) {
  return val > 0 ? val : 0;
}

template <typename DType>
__device__ inline DType sigmoid(const DType val) {
  return 1.f/(1 + expf(-val));
}

template <typename DType>
__device__ inline DType softrelu(const DType val) {
  return logf(1 + expf(val));
}

template <typename DType>
__device__ inline DType softsign(const DType val) {
  return val / (1 + fabsf(val));
}

// exp and log

template <typename DType>
__device__ inline DType exp(const DType val) {
  return expf(val);
}

template <typename DType>
__device__ inline DType expm1(const DType val) {
  return expm1f(val);
}

template <typename DType>
__device__ inline DType log(const DType val) {
  return logf(val);
}

template <typename DType>
__device__ inline DType log10(const DType val) {
  return log10f(val);
}

template <typename DType>
__device__ inline DType log2(const DType val) {
  return log2f(val);
}

template <typename DType>
__device__ inline DType log1p(const DType val) {
  return log1pf(val);
}

// trigonometric

constexpr double pi = 3.14159265358979323846;

template <typename DType>
__device__ inline DType degrees(const DType val) {
  return (val / pi) * 180;
}

template <typename DType>
__device__ inline DType radians(const DType val) {
  return (val / 180.0) * pi;
}

template <typename DType>
__device__ inline DType sin(const DType val) {
  return sinf(val);
}

template <typename DType>
__device__ inline DType cos(const DType val) {
  return cosf(val);
}

template <typename DType>
__device__ inline DType tan(const DType val) {
  return tanf(val);
}

template <typename DType>
__device__ inline DType arcsin(const DType val) {
  return asinf(val);
}

template <typename DType>
__device__ inline DType arccos(const DType val) {
  return acosf(val);
}

template <typename DType>
__device__ inline DType arctan(const DType val) {
  return atanf(val);
}

template <typename DType>
__device__ inline DType sinh(const DType val) {
  return sinhf(val);
}

template <typename DType>
__device__ inline DType cosh(const DType val) {
  return coshf(val);
}

template <typename DType>
__device__ inline DType tanh(const DType val) {
  return tanhf(val);
}

template <typename DType>
__device__ inline DType arcsinh(const DType val) {
  return asinhf(val);
}

template <typename DType>
__device__ inline DType arccosh(const DType val) {
  return acoshf(val);
}

template <typename DType>
__device__ inline DType arctanh(const DType val) {
  return atanhf(val);
}

// sqrt

template <typename DType>
__device__ inline DType sqrt(const DType val) {
  return sqrtf(val);
}

template <typename DType>
__device__ inline DType rsqrt(const DType val) {
  return rsqrtf(val);
}

template <typename DType>
__device__ inline DType cbrt(const DType val) {
  return cbrtf(val);
}

template <typename DType>
__device__ inline DType rcbrt(const DType val) {
  return rcbrtf(val);
}

template <typename DType>
__device__ inline DType square(const DType val) {
  return val * val;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type zero(const DType val) {
  return 0;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type zero() {
  return 0;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type one(const DType val) {
  return 1;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type one() {
  return 1;
}

template <typename DType>
__device__ inline DType round(const DType val) {
  return roundf(val);
}

template <typename DType>
__device__ inline DType rint(const DType val) {
  return rintf(val);
}

template <typename DType>
__device__ inline DType fix(const DType val) {
    const auto floor = floorf(val);
    const auto ceil = ceilf(val);
    return (floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil;
}

template <typename DType>
__device__ inline DType floor(const DType val) {
    return floorf(val);
}

template <typename DType>
__device__ inline DType ceil(const DType val) {
    return ceilf(val);
}

template <typename DType>
__device__ inline DType trunc(const DType val) {
    return truncf(val);
}

template <typename DType>
__device__ inline DType clip(const DType val, const float a_min, const float a_max) {
  return max(min(val, a_max), a_min);
}

template <typename DType>
__device__ inline DType sign(const DType val) {
  if (val < 0) return -1;
  return val > 0 ? 1 : 0;
}

template <typename DType>
__device__ inline DType reciprocal(const DType val) {
  return 1.0f / val;
}

template <typename DType>
__device__ inline DType abs(const DType val) {
  return fabsf(val);
}

template <typename DType>
__device__ inline DType gamma(const DType val) {
  return tgammaf(val);
}

template <typename DType>
__device__ inline DType gammaln(const DType val) {
  return lgammaf(val);
}

template <typename DType>
__device__ inline DType erf(const DType val) {
  return erff(val);
}

template <typename DType>
__device__ inline DType erfinv(const DType val) {
  return erfinvf(val);
}

template <typename DType1, typename DType2>
__device__ inline DType1 smooth_l1(const DType1 val, const DType2 scalar) {
  const auto bsq = scalar * scalar;
  const auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return val - 0.5f * ibsq;
  } else if (val < -ibsq) {
    return -val - 0.5f * ibsq;
  } else {
    return 0.5f * val * val * bsq;
  }
}

}  // namespace op

)code";

const char backward_function_definitions[] = R"code(

namespace op {

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_relu(const DType val, const DTypeGrad grad) {
  return val > 0 ? grad : 0;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sigmoid(const DType out, const DTypeGrad grad) {
  return grad * out * (1 - out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softrelu(const DType val, const DTypeGrad grad) {
  return grad * sigmoid(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softsign(const DType val, const DTypeGrad grad) {
  const DType ap1 = 1 + fabsf(val);
  return grad / (ap1 * ap1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_exp(const DType val, const DTypeGrad grad) {
  return grad * expf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_expm1(const DType val, const DTypeGrad grad) {
  return grad * expf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log(const DType val, const DTypeGrad grad) {
  return grad / val;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log10(const DType val, const DTypeGrad grad) {
  return grad / (val * logf(10));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log2(const DType val, const DTypeGrad grad) {
  return grad / (val * logf(2));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log1p(const DType val, const DTypeGrad grad) {
  return grad / (1 + val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sin(const DType val, const DTypeGrad grad) {
  return grad * cosf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cos(const DType val, const DTypeGrad grad) {
  return -grad * sinf(val);
}

// Uses output from tan
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tan(const DType out, const DTypeGrad grad) {
  return grad * (out * out + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsin(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccos(const DType val, const DTypeGrad grad) {
  return -grad / sqrtf(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctan(const DType val, const DTypeGrad grad) {
  return grad / (1 + val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sinh(const DType val, const DTypeGrad grad) {
  return grad * coshf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cosh(const DType val, const DTypeGrad grad) {
  return grad * sinhf(val);
}

// Uses tanh output
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tanh(const DType out, const DTypeGrad grad) {
  return grad * (1 - out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsinh(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(val * val + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccosh(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(val * val - 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctanh(const DType val, const DTypeGrad grad) {
  return grad / (1 - val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sqrt(const DType out, const DTypeGrad grad) {
  return 0.5 * grad / out;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rsqrt(const DType val, const DTypeGrad grad) {
  const DType inv = 1 / val;
  return -0.5 * grad * sqrtf(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cbrt(const DType out, const DTypeGrad grad) {
  return grad / (3.0f * out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rcbrt(const DType val, const DTypeGrad grad) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * cbrtf(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_square(const DType val, const DTypeGrad grad) {
  return 2 * val * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_clip(const DType val, const DTypeGrad grad,
                                          const float a_min, const float a_max) {
  if (val > a_max || val < a_min) {
    return 0;
  } else {
    return grad;
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_reciprocal(const DType val, const DTypeGrad grad) {
  return -grad / (val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erf(const DType val, const DTypeGrad grad) {
  return 2.0f / sqrt(pi) * exp(-(val*val)) * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erfinv(const DType val, const DTypeGrad grad) {
  return 0.5f * sqrt(pi) * exp(val * val) * grad;
}

template <typename DType, typename DType2, typename DTypeGrad>
__device__ inline DTypeGrad backward_smooth_l1(const DType val, const DType2 scalar,
                                               const DTypeGrad grad) {
  auto bsq = scalar * scalar;
  auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return grad;
  } else if (val < -ibsq) {
    return -grad;
  } else {
    return bsq * val * grad;
  }
}

}  // namespace op

)code";

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

#endif  // MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
