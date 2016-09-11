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
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_plus)
.add_alias("broadcast_plus")
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::plus>)
.attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_plus"});

NNVM_REGISTER_OP(_backward_plus)
.set_num_inputs(1)
.set_num_outputs(2)
.attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity, mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_minus)
.add_alias("broadcast_minus")
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::minus>)
.attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_minus"});

NNVM_REGISTER_OP(_backward_minus)
.set_num_inputs(1)
.set_num_outputs(2)
.attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity, mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_mul)
.add_alias("broadcast_mul")
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::mul>)
.attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::left,
                                                              mshadow_op::right>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_div)
.add_alias("broadcast_div")
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow::op::div>)
.attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::div_grad,
                                                              mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_power)
.add_alias("broadcast_power")
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power"});

NNVM_REGISTER_OP(_backward_power)
.set_num_inputs(3)
.set_num_outputs(2)
.attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::power_grad,
                                                              mshadow_op::power_rgrad>);

}  // namespace op
}  // namespace mxnet

//



// MXNET_REGISTER_SIMPLE_OP(broadcast_plus, XPU)
// .set_shape_function(BinaryBroadcastShape_)
// .set_function(XPU::kDevMask, BinaryBroadcastForward_<
//               XPU, mshadow::op::plus>, kNoInplace, kRegisterSymbolic)
// .set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
//               XPU, mshadow_op::identity, mshadow_op::identity>, kNoInplace)
// .describe("lhs add rhs with broadcast");

// MXNET_REGISTER_SIMPLE_OP(broadcast_minus, XPU)
// .set_shape_function(BinaryBroadcastShape_)
// .set_function(XPU::kDevMask, BinaryBroadcastForward_<
//               XPU, mshadow::op::minus>, kNoInplace, kRegisterSymbolic)
// .set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
//               XPU, mshadow_op::identity, mshadow_op::negation>, kNoInplace)
// .describe("lhs minus rhs with broadcast");

// MXNET_REGISTER_SIMPLE_OP(broadcast_mul, XPU)
// .set_shape_function(BinaryBroadcastShape_)
// .set_function(XPU::kDevMask, BinaryBroadcastForward_<
//               XPU, mshadow::op::mul>, kNoInplace, kRegisterSymbolic)
// .set_gradient(XPU::kDevMask, BroadcastMulBackward_<XPU>, kNoInplace)
// .describe("lhs multiple rhs with broadcast");

// MXNET_REGISTER_SIMPLE_OP(broadcast_div, XPU)
// .set_shape_function(BinaryBroadcastShape_)
// .set_function(XPU::kDevMask, BinaryBroadcastForward_<
//               XPU, mshadow::op::div>, kNoInplace, kRegisterSymbolic)
// .set_gradient(XPU::kDevMask, BroadcastDivBackward_<XPU>, kNoInplace)
// .describe("lhs divide rhs with broadcast");

// MXNET_REGISTER_SIMPLE_OP(broadcast_power, XPU)
// .set_shape_function(BinaryBroadcastShape_)
// .set_function(XPU::kDevMask, BinaryBroadcastForward_<
//               XPU, mshadow_op::power>, kNoInplace, kRegisterSymbolic)
// .set_gradient(XPU::kDevMask, BroadcastPowerBackward_<XPU>, kNoInplace)
// .describe("lhs power rhs with broadcast");
