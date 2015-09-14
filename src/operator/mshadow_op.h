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
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a;
  }
};

struct identity_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f;
  }
};


struct negation {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return -a;
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? a : 0.0f;
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0.0f ? a : a * b;
  }
};

struct xelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0.0f ? 1.0f : b;
  }
};

struct tanh {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return tanhf( a );
  }
};

struct tanh_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f - a * a;
  }
};


struct square {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a;
  }
};

/*! \brief used for generate Bernoulli mask */
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

/*! \brief used for generate element of power */
struct power {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return powf( a, b );
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return sqrt(a);
  }
};
}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
