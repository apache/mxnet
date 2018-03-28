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

/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantization_utils-inl.h
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_

#include <mxnet/base.h>
#include <algorithm>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {


template<typename T>
MSHADOW_XINLINE int Sign(T val) {
  return (val > T(0)) - (val < T(0));
}

template<typename T>
MSHADOW_XINLINE T Abs(T a) {
#ifdef __CUDACC__
  return ::abs(a);
#else
  return std::abs(a);
#endif
}

template<typename T>
MSHADOW_XINLINE T Max(T a, T b) {
#ifdef __CUDACC__
  return ::max(a, b);
#else
  return std::max(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE T Min(T a, T b) {
#ifdef __CUDACC__
  return ::min(a, b);
#else
  return std::min(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE float MaxAbs(T a, T b) {
  return Max(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

template<typename T>
MSHADOW_XINLINE float MinAbs(T a, T b) {
  return Min(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

template<typename T>
MSHADOW_XINLINE T FloatToQuantized(float input, float min_range, float max_range) {
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  float real_range = MaxAbs(min_range, max_range);
  float quantized_range = MinAbs(MaxValue<T>(), MinValue<T>());
  float scale = quantized_range / real_range;
  return Sign(input) * Min(Abs(input) * scale + 0.5f, quantized_range);
}

template <typename T>
MSHADOW_XINLINE float QuantizedToFloat(T input, float min_range, float max_range) {
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  float quantized_range = MinAbs(MinValue<T>(), MaxValue<T>());
  float real_range = MaxAbs(min_range, max_range);
  float scale = real_range / quantized_range;
  return input * scale;
}

struct QuantizedToFloatStruct {
  template<typename T>
  MSHADOW_XINLINE static void Map(int i, float *output, const T *input,
                                  const float *range_min, const float *range_max) {
    output[i] = QuantizedToFloat(input[i], *range_min, *range_max);
  }
};

template <class T1, class T2>
MSHADOW_XINLINE T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                                        float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
MSHADOW_XINLINE void RequantizeManyInNewRange(size_t count, T2* output, const T1 *input,
                                              float input_min, float input_max,
                                              float actual_min, float actual_max) {
  for (size_t i = 0; i < count; ++i) {
    const float input_float =
        QuantizedToFloat<T1>(input[i], input_min, input_max);
    output[i] = FloatToQuantized<T2>(input_float, actual_min, actual_max);
  }
}

/*!
 * \brief Get the scaling factor for converting type T to float.
 */
template<typename T>
MSHADOW_XINLINE float FloatForOneQuantizedLevel(float range_min, float range_max) {
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  const int64_t highest = static_cast<int64_t>(MaxValue<T>());
  const int64_t lowest  = static_cast<int64_t>(MinValue<T>());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <typename TA, typename TB, typename TC>
MSHADOW_XINLINE void QuantizationRangeForMultiplication(float min_a, float max_a,
                                                        float min_b, float max_b,
                                                        float* min_c, float* max_c) {
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  const float a_float_for_one_quant_level =
    FloatForOneQuantizedLevel<TA>(min_a, max_a);
  const float b_float_for_one_quant_level =
    FloatForOneQuantizedLevel<TB>(min_b, max_b);

  const int64_t c_highest =
    static_cast<int64_t>(MaxValue<TC>());
  const int64_t c_lowest  =
    static_cast<int64_t>(MinValue<TC>());
  const float c_float_for_one_quant_level =
    a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

struct QuantizationRangeForMultiplicationStruct {
  MSHADOW_XINLINE static void Map(int i,
                                  float *min_c,
                                  float *max_c,
                                  const float *min_a,
                                  const float *max_a,
                                  const float *min_b,
                                  const float *max_b) {
  QuantizationRangeForMultiplication<int8_t, int8_t, int32_t>(
    min_a[i], max_a[i], min_b[i], max_b[i], min_c, max_c);
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_
