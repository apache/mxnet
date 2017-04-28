/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantization_utils-inl.h
 * \brief (TODO)
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_UTILS_H_

#include <mxnet/base.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

using mshadow::red::limits::MinValue;
using mshadow::red::limits::MaxValue;

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
MSHADOW_XINLINE float FloatToQuantizedUnclamped(
    float input, float min_range, float max_range) {
  const float real_range     = MaxAbs(min_range, max_range);
  const float quantize_range = MinAbs(MinValue<T>(), MaxValue<T>());
  float scale = quantize_range / real_range;
  return Sign(input) * (Abs(input) * scale + 0.5f);
}

template<typename T>
MSHADOW_XINLINE T FloatToQuantized(
    float input, float min_range, float max_range) {
  float real_range = MaxAbs(min_range, max_range);
  float quantized_range = MinAbs(MaxValue<T>(), MinValue<T>());
  float scale = quantized_range / real_range;
  return Sign(input) * Min(Abs(input) * scale + 0.5f, quantized_range);
}

template <typename T>
MSHADOW_XINLINE float QuantizedToFloat(T input, float min_range, float max_range) {
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
MSHADOW_XINLINE void RequantizeManyInNewRange(size_t count,
    T2* output, const T1 *input, float input_min,
    float input_max, float actual_min, float actual_max) {
  for (size_t i = 0; i < count; ++i) {
    const float input_float =
        QuantizedToFloat<T1>(input[i], input_min, input_max);
    output[i] = FloatToQuantized<T2>(input_float, actual_min, actual_max);
  }
}

struct RequantizeManyInNewRangeStruct {
  template<typename T1, typename T2>
  MSHADOW_XINLINE static void Map(int i, T2 *output, float *omin_range, float *omax_range,
      const T1 *input, const float *imin_range, const float *imax_range,
      const float *actual_min, const float *actual_max) {

    const float input_float = QuantizedToFloat<T1>(input[i], *imin_range, *imax_range);
    float real_range = MaxAbs(*actual_min, *actual_max);
    *omin_range = -real_range;
    *omax_range =  real_range;
    output[i] = FloatToQuantized<T2>(input_float, -real_range, real_range);
  }
};

template<typename T>
MSHADOW_XINLINE float FloatForOneQuantizedLevel(
    float range_min, float range_max) {
  const int64_t highest = static_cast<int64_t>(MaxValue<T>());
  const int64_t lowest  = static_cast<int64_t>(MinValue<T>());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <typename TA, typename TB, typename TC>
MSHADOW_XINLINE void QuantizationRangeForMultiplication(
    float min_a, float max_a, float min_b, float max_b,
    float* min_c, float* max_c) {
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



// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by MXNet, and implements
// C = A * B, like the standard BLAS GEMM interface. If the tranpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

template <class T1, class T2, class T3>
void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, const T1* a, int32_t offset_a,
                   size_t lda, const T2* b, int32_t offset_b, size_t ldb, T3* c,
                   int32_t shift_c, int32_t offset_c, int32_t mult_c, size_t ldc) {
  int a_i_stride;
  int a_l_stride;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  } else {
    a_i_stride = lda;
    a_l_stride = 1;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  } else {
    b_j_stride = 1;
    b_l_stride = ldb;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  } else {
    c_i_stride = ldc;
    c_j_stride = 1;
  }

  const int32_t highest =
    static_cast<int32_t>(std::numeric_limits<T3>::max());
  const int32_t lowest  =
    static_cast<int32_t>(std::numeric_limits<T3>::min());
  const int32_t rounding =
    (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const int32_t a_value = static_cast<int32_t>(a[a_index]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const int32_t b_value = static_cast<int32_t>(b[b_index]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }
      if (output < lowest) {
        output = lowest;
      }
      c[c_index] = static_cast<T3>(output);
    }
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_UTILS_H_
