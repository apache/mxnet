/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op.cu
 * \brief elementwise binary operator
*/
#include "./elementwise_binary_scalar_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateElementwiseBinaryScalarOp<gpu>(elembinary::ElementwiseBinaryScalarOpType type, ScalarOpParam param) {
    return CreateElementwiseBinaryScalarOp_<gpu>(type, param);
}
}  // namespace op
}  // namespace mxnet
