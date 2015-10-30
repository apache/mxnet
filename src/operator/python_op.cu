/*!
 * Copyright (c) 2015 by Contributors
 * \file python_op.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./python_op-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(PythonOpParam param) {
  return new PythonOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
