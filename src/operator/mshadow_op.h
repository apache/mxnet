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
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_

#include <mxnet/base.h>
#include "math.h"
#include "math_functions-inl.h"
#include "special_functions-inl.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace mxnet {
namespace op {
namespace mshadow_op {

#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif
using std::enable_if;
using std::is_unsigned;

#define MXNET_UNARY_MATH_OP(name) \
struct name { \
  template<typename DType> \
  MSHADOW_XINLINE static DType Map(DType a) { \
    return math::name(a); \
  } \
}

#define MXNET_BINARY_MATH_OP(name) \
struct name { \
  template<typename DType> \
  MSHADOW_XINLINE static DType Map(DType a, DType b) { \
    return math::name(a, b); \
  } \
}

/*! \brief identity Operation */
struct identity {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a;
  }
};

struct identity_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f);
  }
};

struct left {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a;
  }
};

struct right {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return b;
  }
};

struct negation {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-a);
  }
};

struct reciprocal {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f/a);
  }
};

struct reciprocal_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-(DType(1.0f) / (a * a)));
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(1.0f / (1.0f + math::exp(-af)));
  }
};

template<>
MSHADOW_XINLINE double sigmoid::Map<double>(double a) {
  return 1.0 / (1.0 + math::exp(-a));
}

struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * (DType(1.0f) - a));
  }
};

/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a > DType(0.0f) ? a : DType(0.0f);
  }
};

struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a > DType(0.0f) ? DType(1.0f) : DType(0.0f);
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > DType(0.0f) ? a : a * b);
  }
};

struct xelu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > DType(0.0f) ? DType(1.0f) : b);
  }
};

/*! \brief Exponential Linear Unit */
struct elu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType a) {
    float af(static_cast<float>(a));
    return x > DType(0.0f) ? x : DType(af * math::expm1f(x));
  }
};

struct elu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType a) {
    return x > DType(0.0f) ? DType(1.0f) : DType(a + x);
  }
};

MXNET_UNARY_MATH_OP(tanh);

struct tanh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) - a * a);
  }
};

/*! \brief SoftReLU, also known as softplus activation */
struct softrelu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // Avoid overflow of exp for large inputs.
    // Thresholds 20.0 is chosen such that softrelu(a) = a
    // for a > 20 using floating precision.
    if (a > DType(20.0f)) {
      return a;
    } else {
      return DType(math::log1p(math::expf(a)));
    }
  }
};

template<>
MSHADOW_XINLINE double softrelu::Map<double>(double a) {
  if (a > 20.0) {
    return a;
  } else {
    return math::log1p(math::exp(a));
  }
}

struct softrelu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return -math::expm1(-a);
  }
};

MXNET_UNARY_MATH_OP(exp);

MXNET_UNARY_MATH_OP(expm1);

MXNET_UNARY_MATH_OP(log);

struct log_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / a);
  }
};

MXNET_UNARY_MATH_OP(log10);

// Constant is 1 / log(10)
struct log10_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.43429448190325182765f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log10_grad::Map<double>(double a) {
  return 0.43429448190325182765 / a;
}

MXNET_UNARY_MATH_OP(log2);

// Constant is 1 / log(2)
struct log2_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.44269504088896340737f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log2_grad::Map<double>(double a) {
  return 1.44269504088896340737 / a;
}

MXNET_UNARY_MATH_OP(sin);

struct sin_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::cos(a);
  }
};

MXNET_UNARY_MATH_OP(log1p);

struct log1p_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (DType(1.0f) + a));
  }
};

MXNET_UNARY_MATH_OP(cos);

struct cos_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return -math::sin(a);
  }
};

MXNET_UNARY_MATH_OP(tan);

struct tan_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * a + DType(1.0f));
  }
};

struct arcsin {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::asin(a);
  }
};

struct arcsin_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(1.0f / math::sqrt(1.0f - af * af));
  }
};

template<>
MSHADOW_XINLINE double arcsin_grad::Map<double>(double a) {
  return 1.0 / math::sqrt(1.0 - a * a);
}

struct arccos {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::acos(a);
  }
};

struct arccos_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(-1.0f / math::sqrt(1.0f - af * af));
  }
};

template<>
MSHADOW_XINLINE double arccos_grad::Map<double>(double a) {
  return -1.0 / math::sqrt(1.0 - a * a);
}

struct arctan {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::atan(a);
  }
};

struct arctan_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (a * a + DType(1.0f)));
  }
};

MXNET_BINARY_MATH_OP(hypot);

struct hypot_grad_left {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    float af(static_cast<float>(a));
    return DType(af / math::hypotf(a, b));
  }
};

template<>
MSHADOW_XINLINE double hypot_grad_left::Map<double>(double a, double b) {
  return a / math::hypot(a, b);
}

struct hypot_grad_right {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    float bf(static_cast<float>(b));
    return DType(bf / math::hypotf(a, b));
  }
};

template<>
MSHADOW_XINLINE double hypot_grad_right::Map<double>(double a, double b) {
  return b / math::hypot(a, b);
}

struct degrees {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(180. / PI * a);
  }
};

struct degrees_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(180. / PI);
  }
};

struct radians {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(PI /180. * a);
  }
};

struct radians_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(PI / 180.);
  }
};

MXNET_UNARY_MATH_OP(sinh);

struct sinh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::cosh(a);
  }
};

MXNET_UNARY_MATH_OP(cosh);

struct cosh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::sinh(a);
  }
};

struct arcsinh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::asinh(a);
  }
};

struct arcsinh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(1.0f / math::sqrt(1.0f + af * af));
  }
};

template<>
MSHADOW_XINLINE double arcsinh_grad::Map<double>(double a) {
  return 1.0 / math::sqrt(1.0 + a * a);
}

struct arccosh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::acosh(a);
  }
};

struct arccosh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(1.0f / math::sqrt(af * af - 1.0f));
  }
};

template<>
MSHADOW_XINLINE double arccosh_grad::Map<double>(double a) {
  return 1.0 / math::sqrt(a * a - 1.0);
}

struct arctanh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::atanh(a);
  }
};

struct arctanh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (DType(1.0f) - a * a));
  }
};

struct square {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * a);
  }
};

struct square_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(2.0f) * a);
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? DType(1.0f) : DType(0.0f);
  }
};

/*! \brief used for generate element of abs */
struct abs {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::fabs(a);  // NOLINT(*)
  }
};

/*! \brief used for generate element of sign */
struct sign {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a < DType(0.0f)) return DType(-DType(1.0f));
    if (a > DType(0.0f)) return DType(1.0f);
    return DType(0.0f);
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a > DType(0.0f)) return DType(1.0f);
    return DType(0.0f);
  }
};

struct sign_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.0f);
  }
};

/*! \brief used for generate element of power */
struct power {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return math::pow(a, b);
  }
};

struct power_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    float bf(static_cast<float>(b));
    return DType(math::pow(static_cast<float>(a), bf - 1.0f) * bf);
  }
};

template<>
MSHADOW_XINLINE double power_grad::Map<double>(double a, double b) {
  return math::pow(a, b - 1.0) * b;
}

struct power_rgrad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(math::powf(a, b) * math::logf(a));
  }
};

template<>
MSHADOW_XINLINE double power_rgrad::Map<double>(double a, double b) {
  return math::pow(a, b) * math::log(a);
}

struct rpower {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return math::pow(b, a);
  }
};

struct rpower_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(static_cast<float>(a) * math::logf(b));
  }
};

template<>
MSHADOW_XINLINE double rpower_grad::Map<double>(double a, double b) {
  return a * math::log(b);
}

/*! \brief used for generate element of maximum */
struct maximum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a > b ? a : b;
  }
};

/*! \brief used for generate element of minimum */
struct minimum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? a : b;
  }
};

struct ge {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a >= b ? DType(1.0f) : DType(0.0f);
  }
};

struct gt {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a > b ? DType(1.0f) : DType(0.0f);
  }
};

struct lt {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? DType(1.0f) : DType(0.0f);
  }
};

struct le {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a <= b ? DType(1.0f) : DType(0.0f);
  }
};

struct eq {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a == b ? DType(1.0f) : DType(0.0f);
  }
};

struct ne {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a != b ? DType(1.0f) : DType(0.0f);
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::sqrt(a);
  }
};

struct square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(0.5f) / a);
  }
};

/*!\ \brief used for generate element rsqrt */
struct reciprocal_square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f / math::sqrtf(a));
  }
};

template<>
MSHADOW_XINLINE double reciprocal_square_root::Map<double>(double a) {
  return 1.0 / math::sqrt(a);
}

struct reciprocal_square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(-0.5f / (af * math::sqrt(af)));
  }
};

template<>
MSHADOW_XINLINE double reciprocal_square_root_grad::Map<double>(double a) {
  return -0.5 / (a * math::sqrt(a));
}

/*!\ \brief used for generate element cbrt */
struct cube_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::cbrt(a);
  }
};

struct cube_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (DType(3.0f) * a * a));
  }
};

/*!\ \brief used for generate element rcbrt */
struct reciprocal_cube_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0f / math::cbrtf(a));
  }
};

template<>
MSHADOW_XINLINE double reciprocal_cube_root::Map<double>(double a) {
  return 1.0 / math::cbrt(a);
}

struct reciprocal_cube_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float af(static_cast<float>(a));
    return DType(-1.0f / (3.0f * af * math::cbrt(af)));
  }
};

template<>
MSHADOW_XINLINE double reciprocal_cube_root_grad::Map<double>(double a) {
  return -1.0 / (3.0 * a * math::cbrt(a));
}

/*! \brief used for generate element of round */
MXNET_UNARY_MATH_OP(round);

/*! \brief used for generate element of ceil */
MXNET_UNARY_MATH_OP(ceil);

/*! \brief used for generate element of floor */
MXNET_UNARY_MATH_OP(floor);

/*! \brief used to round towards zero */
MXNET_UNARY_MATH_OP(trunc);

/*! \brief used to round number to nearest integer */
struct rint {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float floor = math::floorf(a);
    float ceil = math::ceilf(a);
    return DType((a - floor) <= (ceil - a) ? floor : ceil);
  }
};

/*! \brief used to round number to integer nearest to 0 */
struct fix {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float floor = math::floorf(a);
    float ceil = math::ceilf(a);
    return DType((floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil);
  }
};

/*! \brief used for generate gradient of MAE loss*/
struct minus_sign {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a-b > DType(0.0f) ? DType(1.0f) : -DType(1.0f));
  }
};

struct rminus {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(b - a);
  }
};

struct div_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(DType(1.0f) / b);
  }
};

struct div_rgrad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(-a / (b * b));
  }
};

struct rdiv {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(b / a);
  }
};

struct rdiv_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(-b / (a * a));
  }
};

struct mod {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (b == DType(0)) {
      return DType(0);
    } else if (b < DType(0)) {
      if (a < DType(0)) {
        return DType(-::fmod(-static_cast<double>(a), -static_cast<double>(b)));
      } else {
        return DType(::fmod(static_cast<double>(a), -static_cast<double>(b)) +
                     (::fmod(static_cast<double>(a), -static_cast<double>(b)) != DType(0)
                      ? b : DType(0)));
      }
    } else {
      if (a < DType(0)) {
        return DType(-::fmod(-static_cast<double>(a), static_cast<double>(b)) +
                     (::fmod(-static_cast<double>(a), static_cast<double>(b)) != DType(0)
                      ? b : DType(0)));
      } else {
        return DType(::fmod(static_cast<double>(a), static_cast<double>(b)));
      }
    }
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (b == DType(0)) {
      return DType(0);
    } else {
      return DType(::fmod(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};
#ifdef __CUDACC__
template<>
MSHADOW_XINLINE mshadow::half::half2_t mod::Map<mshadow::half::half2_t>
                                               (mshadow::half::half2_t a,
                                                mshadow::half::half2_t b) {
  return a%b;
}
#endif

struct mod_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double mod_grad::Map<double>(double a, double b) {
  return 1.0;
}
template<>
MSHADOW_XINLINE float mod_grad::Map<float>(float a, float b) {
  return 1.0f;
}
#ifdef __CUDACC__
template<>
MSHADOW_XINLINE mshadow::half::half_t mod_grad::Map<mshadow::half::half_t>
                                                   (mshadow::half::half_t a,
                                                    mshadow::half::half_t b) {
  return mshadow::half::half_t(1.0f);
}
template<>
MSHADOW_XINLINE mshadow::half::half2_t mod_grad::Map<mshadow::half::half2_t>
                                                    (mshadow::half::half2_t a,
                                                     mshadow::half::half2_t b) {
  mshadow::half::half2_t result = mshadow::half::half2_t();
#if MSHADOW_CUDA_HALF2
  result.half2_ = ::__float2half2_rn(1.0f);
#else
  result.half_t2[0] = mshadow::half::half_t(0.0f);
  result.half_t2[1] = mshadow::half::half_t(1.0f);
#endif
  return result;
}
#endif

struct mod_rgrad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double mod_rgrad::Map<double>(double a, double b) {
  return -::floor(a/b);
}
template<>
MSHADOW_XINLINE float mod_rgrad::Map<float>(float a, float b) {
  return -::floorf(a/b);
}
#ifdef __CUDACC__
template<>
MSHADOW_XINLINE mshadow::half::half_t mod_rgrad::Map<mshadow::half::half_t>
                                                    (mshadow::half::half_t a,
                                                     mshadow::half::half_t b) {
  return mshadow::half::half_t(-::floorf(static_cast<float>(a/b)));
}
template<>
MSHADOW_XINLINE mshadow::half::half2_t mod_rgrad::Map<mshadow::half::half2_t>
                                                     (mshadow::half::half2_t a,
                                                      mshadow::half::half2_t b) {
#if MSHADOW_CUDA_HALF2
  return mshadow::half::half2_t(__hneg2(::h2floor((a/b).half2_)));
#else
  return mshadow::half::half2_t(mshadow::half::half_t(-::floorf(
                                  static_cast<float>(a.half_t2[0]/b.half_t2[0]))),
                                mshadow::half::half_t(-::floorf(
                                  static_cast<float>(a.half_t2[1]/b.half_t2[1]))));
#endif
}
#endif

struct rmod {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (a == DType(0)) {
      return DType(0);
    } else if (a < DType(0)) {
      if (b < DType(0)) {
        return DType(-::fmod(-static_cast<double>(b), -static_cast<double>(a)));
      } else {
        return DType(::fmod(static_cast<double>(b), -static_cast<double>(a)) +
                     (::fmod(static_cast<double>(b), -static_cast<double>(a)) != DType(0)
                      ? a : DType(0)));
      }
    } else {
      if (b < DType(0)) {
        return DType(-::fmod(-static_cast<double>(b), static_cast<double>(a)) +
                     (::fmod(-static_cast<double>(b), static_cast<double>(a)) != DType(0)
                      ? a : DType(0)));
      } else {
        return DType(::fmod(static_cast<double>(b), static_cast<double>(a)));
      }
    }
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (a == DType(0)) {
      return DType(0);
    } else {
      return DType(::fmod(static_cast<double>(b), static_cast<double>(a)));
    }
  }
};
#ifdef __CUDACC__
template<>
MSHADOW_XINLINE mshadow::half::half2_t rmod::Map<mshadow::half::half2_t>
                                                (mshadow::half::half2_t a,
                                                 mshadow::half::half2_t b) {
  return b%a;
}
#endif

struct rmod_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double rmod_grad::Map<double>(double a, double b) {
  return -::floor(b/a);
}
template<>
MSHADOW_XINLINE float rmod_grad::Map<float>(float a, float b) {
  return -::floorf(b/a);
}
#ifdef __CUDACC__
template<>
MSHADOW_XINLINE mshadow::half::half_t rmod_grad::Map<mshadow::half::half_t>
                                                   (mshadow::half::half_t a,
                                                    mshadow::half::half_t b) {
  return mshadow::half::half_t(-::floorf(static_cast<float>(b/a)));
}
template<>
MSHADOW_XINLINE mshadow::half::half2_t rmod_grad::Map<mshadow::half::half2_t>
                                                     (mshadow::half::half2_t a,
                                                      mshadow::half::half2_t b) {
#if MSHADOW_CUDA_HALF2
  return mshadow::half::half2_t(::__hneg2(::h2floor((b/a).half2_)));
#else
  return mshadow::half::half2_t(mshadow::half::half_t(-::floorf(
                                  static_cast<float>(b.half_t2[0]/a.half_t2[0]))),
                                mshadow::half::half_t(-::floorf(
                                  static_cast<float>(b.half_t2[1]/a.half_t2[1]))));
#endif
}
#endif

struct clip {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType bound) {
    if (x > bound) {
      return bound;
    } else if (x < -bound) {
      return -bound;
    } else {
      return x;
    }
  }
};

/***** gamma ******/

struct gamma {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::tgamma(a);
  }
};

struct gamma_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    float af(static_cast<float>(a));
    return DType(math::tgamma(af) * special_functions::cephes::psi<float>(af));
  }
};

template<>
MSHADOW_XINLINE double gamma_grad::Map<double>(double a) {
  return math::tgamma(a) * special_functions::cephes::psi<double>(a);
}

/***** gammaln ******/

struct gammaln {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return math::lgamma(a);
  }
};

struct gammaln_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(special_functions::cephes::psi<float>(a));
  }
};

template<>
MSHADOW_XINLINE double gammaln_grad::Map<double>(double a) {
  return special_functions::cephes::psi<double>(a);
}

/* Smooth L1 Loss is a loss specific for R-CNN franchise training
 * Smooth L1 Loss function
 * f(x) = 0.5 * (sigma * x) ^ 2,     |x| < 1 / sigma^2
 *      = |x| - 0.5 / sigma / sigma, otherwise
 * When sigma = 1, it is equivalent to Huber Loss evaluated at
 * delta = 1.
 * smooth_l1_loss = w_out * f(w_in * x)
 * with w_in, w_out provided by input_data.
 */
struct smooth_l1_loss {
  // a is x, b is sigma2
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    b *= b;
    if (a > DType(1.0f) / b) {
      return a - DType(0.5f) / b;
    } else if (a < DType(-1.0f) / b) {
      return -a - DType(0.5f) / b;
    } else {
      return DType(0.5f) * a * a * b;
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, |x| < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_gradient {
  // a is x, b is sigma2
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    b *= b;
    if (a > DType(1.0f) / b) {
      return DType(1.0f);
    } else if (a < DType(-1.0f) / b) {
      return DType(-1.0f);
    } else {
      return b * a;
    }
  }
};  // struct smooth_l1_derivative

/*! \brief product reducer */
struct product {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    dst *= src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& none) { // NOLINT(*)
    Reduce(dst, src);
  }
  /*!
  *\brief calculate gradient of redres with respect to redsrc,
  * redres: reduced result, redsrc: one of reduction element
  */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres / redsrc;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 1;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};

namespace isnan_typed {
  template<typename DType>
  MSHADOW_XINLINE bool IsNan(volatile DType val) {
    return false;
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile float val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile double val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile long double val) {
    return isnan(val);
  }

  template<>
  MSHADOW_XINLINE bool IsNan(volatile mshadow::half::half_t val) {
    return (val.half_ & 0x7fff) > 0x7c00;
  }
};  // namespace isnan_typed

/*! \brief sum reducer that ignores NaN values in the input */
struct nansum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    if (isnan_typed::IsNan(src)) return;
    dst += src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& residual) { // NOLINT(*)
    if (isnan_typed::IsNan(src)) return;
    DType y = src - residual;
    DType t = dst + y;
    residual = (t - dst) - y;
    dst = t;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
      initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    SetInitValue(initv);
    residual = 0;
  }
};

struct nansum_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return isnan_typed::IsNan(a) ? DType(0) : DType(1);
  }
};

/*! \brief product reducer that ignores NaN values in the input */
struct nanprod {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    if (isnan_typed::IsNan(src)) return;
    dst *= src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& none) { // NOLINT(*)
    Reduce(dst, src);
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
    initv = 1;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};

struct nanprod_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return isnan_typed::IsNan(a) ? DType(0) : b / a;
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
