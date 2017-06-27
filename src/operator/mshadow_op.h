/*!
 * Copyright (c) 2015 by Contributors
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_

#include <mxnet/base.h>
#include <math.h>
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

/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (DType(1.0f) + expf(-a)));
  }
};
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
    return DType(a > DType(0.0f) ? a : DType(0.0f));
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a > DType(0.0f) ? DType(1.0f) : DType(0.0f));
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
    return DType(x > DType(0.0f) ? x : a * (expf(x) - DType(1.0f)));
  }
};

struct elu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType a) {
    return DType(x > DType(0.0f) ? DType(1.0f) : a + x);
  }
};

struct tanh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(tanhf( a ));
  }
};

struct tanh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) - a * a);
  }
};

/*! \brief SoftReLU, also known as softplus activation. */
struct softrelu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(log1pf(expf(a)));
  }
};
struct softrelu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) - expf(-a));
  }
};

struct exp {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(expf(a));
  }
};

struct expm1 {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(expm1f(a));
  }
};

struct log {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(logf(a));
  }
};

struct log10 {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(log10f(a));
  }
};

struct log2 {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(log2f(a));
  }
};

struct log_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / a);
  }
};

struct sin {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sinf(a));
  }
};

struct sin_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(cosf(a));
  }
};

struct log1p {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(log1pf(a));
  }
};

struct log1p_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / (DType(1.0f) + a));
  }
};

struct cos {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(cosf(a));
  }
};

struct cos_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-sinf(a));
  }
};

struct tan {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(tanf(a));
  }
};

struct tan_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a * a + 1);
  }
};

struct arcsin {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(asinf(a));
  }
};

struct arcsin_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0 / (sqrtf(1 - a*a)));
  }
};

struct arccos {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(acosf(a));
  }
};

struct arccos_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-1.0 / (sqrtf(1 - a*a)));
  }
};

struct arctan {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(atanf(a));
  }
};

struct arctan_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1 / (a*a + 1));
  }
};

struct hypot {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(sqrtf(a * a + b * b));
  }
};

struct hypot_grad_left {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a/sqrtf(a * a + b * b));
  }
};

struct hypot_grad_right {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(b/sqrtf(a * a + b * b));
  }
};

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

struct sinh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sinhf(a));
  }
};

struct sinh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(coshf(a));
  }
};

struct cosh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(coshf(a));
  }
};

struct cosh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sinhf(a));
  }
};

struct arcsinh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(asinhf(a));
  }
};

struct arcsinh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0 / (sqrtf(1 + a*a)));
  }
};

struct arccosh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(acoshf(a));
  }
};

struct arccosh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.0 / (sqrtf(a*a - 1.0)));
  }
};

struct arctanh {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(atanhf(a));
  }
};

struct arctanh_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-1.0 / (a*a - 1.0));
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
    return DType(a < b ? DType(1.0f) : DType(0.0f));
  }
};

/*! \brief used for generate element of abs */
struct abs {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(fabsf(float(a)));  // NOLINT(*)
  }
};

/*! \brief used for generate element of sign */
struct sign {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (a < 0.0f) return DType(-1.0f);
    if (a > 0.0f) return DType(1.0f);
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
    return DType(powf( a, b ));
  }
};

struct power_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(powf( a, b - 1 )*b);
  }
};

struct power_rgrad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(powf( a, b )*logf(a));
  }
};

struct rpower {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(powf( b, a ));
  }
};

struct rpower_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a*logf(b));
  }
};

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
    return a >= b ? DType(1) : DType(0);
  }
};

struct gt {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a > b ? DType(1) : DType(0);
  }
};

struct lt {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? DType(1) : DType(0);
  }
};

struct le {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a <= b ? DType(1) : DType(0);
  }
};

struct eq {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a == b ? DType(1) : DType(0);
  }
};

struct ne {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a != b ? DType(1) : DType(0);
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(sqrtf(a));
  }
};

struct square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(0.5f) / a);
  }
};

/*!\ \brief used for generate element sqrt */
struct reciprocal_square_root {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f)/sqrtf(a));
  }
};

struct reciprocal_square_root_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(-(DType(1.0f) / (DType(2.0f) * a * sqrtf(a))));
  }
};

/*! \brief used for generate element of round */
struct round {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(roundf(a));
  }
};

/*! \brief used for generate element of ceil */
struct ceil {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(ceilf(a));
  }
};

/*! \brief used for generate element of floor */
struct floor {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(floorf(a));
  }
};

/*! \brief used to round towards zero */
struct trunc {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(truncf(a));
  }
};

/*! \brief used to round number to nearest integer */
struct rint {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float floor = floorf(a);
    float ceil = ceilf(a);
    return DType((a - floor) <= (ceil - a) ? floor : ceil);
  }
};

/*! \brief used to round number to integer nearest to 0 */
struct fix {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    float floor = floorf(a);
    float ceil = ceilf(a);
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
    return DType(b-a);
  }
};

struct div_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(DType(1)/b);
  }
};

struct div_rgrad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(-a/(b*b));
  }
};

struct rdiv {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(b/a);
  }
};

struct rdiv_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(-b/(a*a));
  }
};

struct mod {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
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
  return 1.0f;
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
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
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
    // default implementation using floating precision
    return DType(tgammaf(a));
  }
};

template<>
MSHADOW_XINLINE double gamma::Map<double>(double a) {
  return tgamma(a);
}

struct gamma_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(tgammaf(a) * special_functions::cephes::psi<float>(a));
  }
};

template<>
MSHADOW_XINLINE double gamma_grad::Map<double>(double a) {
  return tgamma(a) * special_functions::cephes::psi<double>(a);
}

/***** gammaln ******/

struct gammaln {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(lgammaf(a));
  }
};

template<>
MSHADOW_XINLINE double gammaln::Map<double>(double a) {
  return lgamma(a);
}

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
 * f(x) = 0.5 * (sigma * x) ^ 2,     x < 1 / sigma^2
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
    if (a > 1.0f / b) {
      return a - 0.5f / b;
    } else if (a < -1.0f / b) {
      return -a - 0.5f / b;
    } else {
      return 0.5f * a * a * b;
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, x < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_gradient {
  // a is x, b is sigma2
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    b *= b;
    if (a > 1.0f / b) {
      return 1.0f;
    } else if (a < -1.0f / b) {
      return DType(-1);
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
    if (isnan_typed::IsNan(dst)) {
      if (isnan_typed::IsNan(src)) {
        dst = DType(0);
      } else {
        dst = src;
      }
    } else {
      if (isnan_typed::IsNan(src)) {
        dst = dst;
      } else {
        dst += src;
      }
    }
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
      initv = 0;
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
    if (isnan_typed::IsNan(dst)) {
      if (isnan_typed::IsNan(src)) {
        dst = DType(1);
      } else {
        dst = src;
      }
    } else {
      if (isnan_typed::IsNan(src)) {
        dst = dst;
      } else {
        dst *= src;
      }
    }
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
    initv = 1;
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
