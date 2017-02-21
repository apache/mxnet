/*!
 * Copyright (c) 2017 by Contributors
 * \file control_flow_op.cu
 * \brief
 */
#include "./control_flow_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(where)
.set_attr<FCompute>("FCompute<gpu>", WhereOpForward<gpu>);

NNVM_REGISTER_OP(_backward_where)
.set_attr<FCompute>("FCompute<gpu>", WhereOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
