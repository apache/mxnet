/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv2d.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_conv2d-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizedConv2DParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedConv2DParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedConv2DOp<DType>();
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QuantizedConv2DProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(quantized_conv2d, QuantizedConv2DProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("filter", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_filter", "NDArray-or-Symbol", "Minimum value of filter.")
.add_argument("max_filter", "NDArray-or-Symbol", "Maximum value of filter.")
.add_arguments(QuantizedConv2DParam::__FIELDS__());

NNVM_REGISTER_OP(quantized_conv2d)
.set_attr<TQuantizationNeedShrink>("TQuantizationNeedShrink", true);

NNVM_REGISTER_OP(Convolution)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const nnvm::NodeAttrs& attrs = n->attrs;
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("quantized_conv2d");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
