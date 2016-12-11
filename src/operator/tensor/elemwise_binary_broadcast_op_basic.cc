/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_add)
.add_alias("broadcast_plus").add_alias("_plus").add_alias("_Plus")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_plus"});

// specialized to elementwise add, currently only used for gradient aggregation
MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_plus"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY(_grad_add)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_plus)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_sub)
.add_alias("broadcast_minus").add_alias("_minus").add_alias("_Minus")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_minus"});

NNVM_REGISTER_OP(_backward_minus)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_mul)
.add_alias("_mul").add_alias("_Mul")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::right,
                                                              mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_div)
.add_alias("_div").add_alias("_Div")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::div_grad,
                                                              mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_equal)
.add_alias("broadcast_equal").add_alias("_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::eq>);
    
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_not_equal)
.add_alias("broadcast_not_equal").add_alias("_Not_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ne>);
    
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_greater)
.add_alias("broadcast_greater").add_alias("_Greater")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::gt>);
    
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_greater_equal)
.add_alias("broadcast_greater_equal").add_alias("_Greater_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ge>);
    
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_lesser)
.add_alias("broadcast_lesser").add_alias("_Lesser")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::lt>);
    
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_lesser_equal)
.add_alias("broadcast_lesser_equal").add_alias("_Lesser_Equal")
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::le>);

}  // namespace op
}  // namespace mxnet
