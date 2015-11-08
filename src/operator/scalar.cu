/*!
 * Copyright (c) 2015 by Contributors
 * \file Scalar.cu
 * \brief
 * \author Bing Xu
*/
#include "./scalar-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ScalarParam param) {
    return new ScalarOp<gpu>(param);
}
}  // op
}  // namespace mxnet

