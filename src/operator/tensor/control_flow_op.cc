/*!
 * Copyright (c) 2015 by Contributors
 * \file control_flow_op.cc
 * \brief CPU Implementation of flow control
 */
#include "./control_flow_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(where)
.MXNET_DESCRIBE("Given three ndarrays, condition, x, and y, return an ndarray"
                " with the elements from x or y, depending on the elements"
                " from condition are true or false. x and y must have the same"
                " shape. If condition has the same shape as x, each element"
                " in the output array is from x if the corresponding element"
                " in the condition is true, and from y if false. If condtion"
                " does not have the same shape as x, it must be a 1D array"
                " whose size is the same as x's first dimension size. Each"
                " row of the output array is from x's row if the corresponding"
                " element from condition is true, and from y's row if false.")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
      return std::vector<std::string>{"condition", "x", "y"};
    })
.set_attr<nnvm::FInferShape>("FInferShape", WhereOpShape)
.set_attr<nnvm::FInferType>("FInferType", WhereOpType)
.set_attr<FCompute>("FCompute<cpu>", WhereOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("condition", "NDArray", "condition array")
.add_argument("x", "NDArray", "")
.add_argument("y", "NDArray", "");

}  // namespace op
}  // namespace mxnet
