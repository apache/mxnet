/*!
 * Copyright (c) 2015 by Contributors
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_

#include <mxnet/base.h>

namespace mxnet {
namespace op {
namespace mshadow_op {
/*! \brief identity Operation */
struct identity {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a);
  }
};

struct identity_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f));
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

struct log {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(logf(a));
  }
};

struct log_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(1.0f) / a);
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
    if (a < 0.0f) return DType(-DType(1.0f));
    if (a > 0.0f) return DType(DType(1.0f));
    return DType(DType(0.0f));
  }
};
struct sign_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(DType(0.0f));
  }
};
/*! \brief used for generate element of power */
struct power {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(powf( a, b ));
  }
};

/*! \brief used for generate element of maximum */
struct maximum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > b ? a : b);
  }
};

struct maximum_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a > b ? DType(1) : DType(0));
  }
};

/*! \brief used for generate element of minimum */
struct minimum {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a < b ? a : b);
  }
};
struct minimum_grad  {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a < b ? DType(1) : DType(0));
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

/*! \brief used for generate gradient of MAE loss*/
struct minus_sign {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a-b > DType(0.0f) ? DType(1.0f) : -DType(1.0f));
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
