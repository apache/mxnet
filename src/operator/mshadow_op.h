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
 * Copyright (c) 2015 by Contributors
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
#include "./operator_tune.h"

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

#define MXNET_UNARY_MATH_OP(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a) { \
      return DType(expr); \
    } \
  }

#define MXNET_UNARY_MATH_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a) { \
      return (expr); \
    } \
  }

#define MXNET_BINARY_MATH_OP(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a, DType b) { \
      return DType(expr); \
    } \
  }

#define MXNET_BINARY_MATH_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable  { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a, DType b) { \
      return (expr); \
    } \
  }

#define MXNET_SIMPLE_UNARY_MATH_OP(name) MXNET_UNARY_MATH_OP(name, math::name(a))

#define MXNET_SIMPLE_BINARY_MATH_OP(name) MXNET_BINARY_MATH_OP(name, math::name(a, b))

MXNET_UNARY_MATH_OP_NC(identity, a);

MXNET_UNARY_MATH_OP(identity_grad, 1);

MXNET_BINARY_MATH_OP_NC(left, a);

MXNET_BINARY_MATH_OP_NC(right, b);

MXNET_BINARY_MATH_OP_NC(mul, a * b);

MXNET_BINARY_MATH_OP_NC(div, a / b);

MXNET_BINARY_MATH_OP_NC(plus, a + b);

MXNET_BINARY_MATH_OP_NC(minus, a - b);

MXNET_UNARY_MATH_OP(negation, -a);

MXNET_UNARY_MATH_OP(reciprocal, 1.0f / math::id(a));

MXNET_UNARY_MATH_OP(reciprocal_grad, -1.0f / math::sqr(a));

MXNET_UNARY_MATH_OP(sigmoid, 1.0f / (1.0f + math::exp(-a)));

MXNET_UNARY_MATH_OP(sigmoid_grad, math::id(a) * (1.0f - math::id(a)));

MXNET_UNARY_MATH_OP_NC(relu, a > DType(0) ? a : DType(0));

MXNET_UNARY_MATH_OP_NC(relu_grad, a > DType(0) ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP(xelu, a > DType(0) ? math::id(a) :
                     math::id(a) * math::id(b));

MXNET_BINARY_MATH_OP_NC(xelu_grad, a > DType(0) ? DType(1) : b);

MXNET_BINARY_MATH_OP(elu, a > DType(0) ? math::id(a) :
                     math::id(b) * math::expm1(a));

MXNET_BINARY_MATH_OP_NC(elu_grad, a > DType(0) ? DType(1) : DType(b + a));

MXNET_SIMPLE_UNARY_MATH_OP(tanh);

MXNET_UNARY_MATH_OP(tanh_grad, 1.0f - math::sqr(a));

/*! \brief SoftReLU, also known as softplus activation */
struct softrelu : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // Avoid overflow of exp for large inputs.
    // Thresholds 20.0 is chosen such that softrelu(a) = a
    // for a > 20 using floating precision
    if (a > DType(20.0f)) {
      return a;
    } else {
      return DType(math::log1p(math::exp(a)));
    }
  }
};

MXNET_UNARY_MATH_OP(softrelu_grad, -math::expm1(-a));

MXNET_SIMPLE_UNARY_MATH_OP(exp);

MXNET_SIMPLE_UNARY_MATH_OP(expm1);

MXNET_SIMPLE_UNARY_MATH_OP(log);

MXNET_UNARY_MATH_OP(log_grad, 1.0f / math::id(a));

MXNET_SIMPLE_UNARY_MATH_OP(log10);

// Constant is 1 / log(10)
struct log10_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.4342944819f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log10_grad::Map<double>(double a) {
  return 0.43429448190325182765 / a;
}

MXNET_SIMPLE_UNARY_MATH_OP(log2);

// Constant is 1 / log(2)
struct log2_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.442695041f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log2_grad::Map<double>(double a) {
  return 1.44269504088896340737 / a;
}

MXNET_SIMPLE_UNARY_MATH_OP(sin);

MXNET_UNARY_MATH_OP(sin_grad, math::cos(a));

MXNET_SIMPLE_UNARY_MATH_OP(log1p);

MXNET_UNARY_MATH_OP(log1p_grad, 1.0f / (1.0f + math::id(a)));

MXNET_SIMPLE_UNARY_MATH_OP(cos);

MXNET_UNARY_MATH_OP(cos_grad, -math::sin(a));

MXNET_SIMPLE_UNARY_MATH_OP(tan);

MXNET_UNARY_MATH_OP(tan_grad, math::sqr(a) + 1.0f);

MXNET_UNARY_MATH_OP(arcsin, math::asin(a));

MXNET_UNARY_MATH_OP(arcsin_grad, 1.0f / math::sqrt(1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(arccos, math::acos(a));

MXNET_UNARY_MATH_OP(arccos_grad, -1.0f / math::sqrt(1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(arctan, math::atan(a));

MXNET_UNARY_MATH_OP(arctan_grad, 1.0f / (math::sqr(a) + 1.0f));

MXNET_SIMPLE_BINARY_MATH_OP(hypot);

MXNET_BINARY_MATH_OP(hypot_grad_left, math::id(a) / math::hypot(a, b));

MXNET_BINARY_MATH_OP(hypot_grad_right, math::id(b) / math::hypot(a, b));

MXNET_UNARY_MATH_OP(degrees, 180.0f / PI * math::id(a));

MXNET_UNARY_MATH_OP(degrees_grad, 180.0f / PI);

MXNET_UNARY_MATH_OP(radians, PI / 180.0f * math::id(a));

MXNET_UNARY_MATH_OP(radians_grad, PI / 180.0f);

MXNET_SIMPLE_UNARY_MATH_OP(sinh);

MXNET_UNARY_MATH_OP(sinh_grad, math::cosh(a));

MXNET_SIMPLE_UNARY_MATH_OP(cosh);

MXNET_UNARY_MATH_OP(cosh_grad, math::sinh(a));

MXNET_UNARY_MATH_OP(arcsinh, math::asinh(a));

MXNET_UNARY_MATH_OP(arcsinh_grad, 1.0f / math::hypot(a, DType(1)));

MXNET_UNARY_MATH_OP(arccosh, math::acosh(a));

MXNET_UNARY_MATH_OP(arccosh_grad, 1.0f / math::sqrt(math::sqr(a) - 1.0f));

MXNET_UNARY_MATH_OP(arctanh, math::atanh(a));

MXNET_UNARY_MATH_OP(arctanh_grad, 1.0f / (1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(square, math::sqr(a));

MXNET_UNARY_MATH_OP(square_grad, 2.0f * math::id(a));

/*! \brief used for generate Bernoulli mask */
MXNET_BINARY_MATH_OP_NC(threshold, a < b ? DType(1) : DType(0));

/*! \brief used for generate element of abs */
MXNET_UNARY_MATH_OP(abs, math::fabs(a)); // NOLINT(*)

/*! \brief used for generate element of sign */
struct sign : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a < DType(0)) return DType(-DType(1));
    if (a > DType(0)) return DType(1);
    return DType(0);
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a > DType(0)) return DType(1);
    return DType(0);
  }
};

MXNET_UNARY_MATH_OP_NC(sign_grad, DType(0));

/*! \brief used for generate element of power */
MXNET_BINARY_MATH_OP(power, math::pow(a, b));

MXNET_BINARY_MATH_OP(power_grad, math::pow(a, b - DType(1)) * math::id(b));

MXNET_BINARY_MATH_OP(power_rgrad, math::pow(a, b) * math::log(a));

MXNET_BINARY_MATH_OP(rpower, math::pow(b, a));

MXNET_BINARY_MATH_OP(rpower_grad, math::id(a) * math::log(b));

/*! \brief used for generate element of maximum */
MXNET_BINARY_MATH_OP(maximum, a > b ? a : b);

/*! \brief used for generate element of minimum */
MXNET_BINARY_MATH_OP_NC(minimum, a < b ? a : b);

MXNET_BINARY_MATH_OP_NC(ge, a >= b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(gt, a > b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(lt, a < b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(le, a <= b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(eq, a == b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(ne, a != b ? DType(1) : DType(0));

MXNET_UNARY_MATH_OP(square_root, math::sqrt(a));

MXNET_UNARY_MATH_OP(square_root_grad, 0.5f / math::id(a));

MXNET_UNARY_MATH_OP(reciprocal_square_root, 1.0f / math::sqrt(a));

MXNET_UNARY_MATH_OP(reciprocal_square_root_grad, -0.5f / (math::sqrt(a) * math::id(a)));

MXNET_UNARY_MATH_OP(cube_root, math::cbrt(a));

MXNET_UNARY_MATH_OP(cube_root_grad, 1.0f / (3.0f * math::sqr(a)));

MXNET_UNARY_MATH_OP(reciprocal_cube_root, 1.0f / math::cbrt(a));

MXNET_UNARY_MATH_OP(reciprocal_cube_root_grad, -1.0f / (3.0f * math::cbrt(a) * math::id(a)));

/*! \brief used for generate element of round */
MXNET_SIMPLE_UNARY_MATH_OP(round);

/*! \brief used for generate element of ceil */
MXNET_SIMPLE_UNARY_MATH_OP(ceil);

/*! \brief used for generate element of floor */
MXNET_SIMPLE_UNARY_MATH_OP(floor);

/*! \brief used to round towards zero */
MXNET_SIMPLE_UNARY_MATH_OP(trunc);

/*! \brief used to round number to nearest integer */
struct rint : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    auto floor = math::floor(a);
    auto ceil = math::ceil(a);
    auto af = math::id(a);
    return DType((af - floor) <= (ceil - af) ? floor : ceil);
  }
};

/*! \brief used to round number to integer nearest to 0 */
struct fix : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    auto floor = math::floor(a);
    auto ceil = math::ceil(a);
    return DType((floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil);
  }
};

/*! \brief used for generate gradient of MAE loss*/
MXNET_BINARY_MATH_OP_NC(minus_sign, a - b > DType(0) ? DType(1) : -DType(1));

MXNET_BINARY_MATH_OP(rminus, b - a);

MXNET_BINARY_MATH_OP(div_grad, 1.0f / math::id(b));

template<>
MSHADOW_XINLINE mshadow::half::half2_t div_grad::Map<mshadow::half::half2_t>
                                               (mshadow::half::half2_t a,
                                                mshadow::half::half2_t b) {
  return mshadow::half::half2_t(1) / b;
}

MXNET_BINARY_MATH_OP(div_rgrad, -math::id(a) / math::sqr(b));

template<>
MSHADOW_XINLINE mshadow::half::half2_t div_rgrad::Map<mshadow::half::half2_t>
                                               (mshadow::half::half2_t a,
                                                mshadow::half::half2_t b) {
  return -a / (b * b);
}

MXNET_BINARY_MATH_OP(rdiv, math::id(b) / math::id(a));

MXNET_BINARY_MATH_OP(rdiv_grad, -math::id(b) / math::sqr(a));

struct mod : public mxnet_op::tunable {
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

template<>
MSHADOW_XINLINE mshadow::half::half2_t mod::Map<mshadow::half::half2_t>
                                               (mshadow::half::half2_t a,
                                                mshadow::half::half2_t b) {
  return a%b;
}

struct mod_grad : public mxnet_op::tunable  {
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
#if (defined(__CUDACC__) && MSHADOW_CUDA_HALF2)
  result.half2_ = ::__float2half2_rn(1.0f);
#else
  result.half_t2[0] = mshadow::half::half_t(0.0f);
  result.half_t2[1] = mshadow::half::half_t(1.0f);
#endif
  return result;
}

struct mod_rgrad : public mxnet_op::tunable {
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
#if (defined(__CUDACC__) && MSHADOW_CUDA_HALF2)
  return mshadow::half::half2_t(__hneg2(::h2floor((a/b).half2_)));
#else
  return mshadow::half::half2_t(mshadow::half::half_t(-::floorf(
                                  static_cast<float>(a.half_t2[0]/b.half_t2[0]))),
                                mshadow::half::half_t(-::floorf(
                                  static_cast<float>(a.half_t2[1]/b.half_t2[1]))));
#endif
}

struct rmod : public mxnet_op::tunable {
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

template<>
MSHADOW_XINLINE mshadow::half::half2_t rmod::Map<mshadow::half::half2_t>
                                                (mshadow::half::half2_t a,
                                                 mshadow::half::half2_t b) {
  return b%a;
}

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
#if (defined(__CUDACC__) && MSHADOW_CUDA_HALF2)
  return mshadow::half::half2_t(::__hneg2(::h2floor((b/a).half2_)));
#else
  return mshadow::half::half2_t(mshadow::half::half_t(-::floorf(
                                  static_cast<float>(b.half_t2[0]/a.half_t2[0]))),
                                mshadow::half::half_t(-::floorf(
                                  static_cast<float>(b.half_t2[1]/a.half_t2[1]))));
#endif
}

struct clip : public mxnet_op::tunable {
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

MXNET_UNARY_MATH_OP(gamma, math::tgamma(a));

struct gamma_grad : public mxnet_op::tunable {
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

MXNET_UNARY_MATH_OP(gammaln, math::lgamma(a));

struct gammaln_grad : public mxnet_op::tunable {
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
 * Smooth L1 Loss function:
 * f(x) = 0.5 * (sigma * x) ^ 2,     |x| < 1 / sigma^2
 *      = |x| - 0.5 / sigma / sigma, otherwise
 * When sigma = 1, it is equivalent to the Huber loss, evaluated at
 * delta = 1.
 * smooth_l1_loss = w_out * f(w_in * x)
 * with w_in, w_out provided by input_data.
 */
struct smooth_l1_loss : public mxnet_op::tunable {
  // a is x, b is sigma
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    auto bsq = math::sqr(b);
    auto ibsq = 1.0f / bsq;
    auto af = math::id(a);
    if (af > ibsq) {
      return DType(af - 0.5f * ibsq);
    } else if (af < -ibsq) {
      return DType(-af - 0.5f * ibsq);
    } else {
      return DType(0.5f * af * af * bsq);
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, |x| < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_gradient : public mxnet_op::tunable {
  // a is x, b is sigma2
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    auto bsq = math::sqr(b);
    auto ibsq = 1.0f / bsq;
    auto af = math::id(a);
    if (af > ibsq) {
      return DType(1);
    } else if (af < -ibsq) {
      return DType(-1);
    } else {
      return DType(bsq * af);
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

struct nansum_grad : public mxnet_op::tunable {
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

struct nanprod_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return isnan_typed::IsNan(a) ? DType(0) : b / a;
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
