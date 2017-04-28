/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_relu.cc
 * \brief
 * \author Ziheng Jiang
*/
#include <mxnet/op_attr_types.h>
#include "./quantized_relu-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(int dtype) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedReluOp<DType>();
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QuantizedReluProp::CreateOperatorEx(Context ctx,
                                           std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(quantized_relu, QuantizedReluProp)
.describe(R"code(Applies an activation function element-wise to the input.
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.")
.add_argument("min_data", "NDArray-or-Symbol", "")
.add_argument("max_data", "NDArray-or-Symbol", "");


NNVM_REGISTER_OP(relu)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const NodeAttrs& attrs = n->attrs;
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("quantized_relu");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });


}  // namespace op
}  // namespace mxnet
