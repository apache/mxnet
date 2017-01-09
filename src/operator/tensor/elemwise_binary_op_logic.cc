/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY(_equal)
.add_alias("_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_BINARY(_not_equal)
.add_alias("_Not_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::ne>);

MXNET_OPERATOR_REGISTER_BINARY(_greater)
.add_alias("_Greater")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::gt>);

MXNET_OPERATOR_REGISTER_BINARY(_greater_equal)
.add_alias("_Greater_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::ge>);

MXNET_OPERATOR_REGISTER_BINARY(_lesser)
.add_alias("_Lesser")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::lt>);

MXNET_OPERATOR_REGISTER_BINARY(_lesser_equal)
.add_alias("_Lesser_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::le>);

}  // namespace op
}  // namespace mxnet
