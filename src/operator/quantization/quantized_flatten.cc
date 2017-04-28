/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantized_flatten.cc
 * \brief
 */
#include <mxnet/op_attr_types.h>
#include "./quantized_flatten-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(quantized_flatten)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedFlattenShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedFlattenType)
.set_attr<FCompute>("FCompute<cpu>", QuantizedFlattenCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_quantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("min_range", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the input")
.add_argument("max_range", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the input");

NNVM_REGISTER_OP(Flatten)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const NodeAttrs& attrs = n->attrs;
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("quantized_flatten");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });


}  // namespace op
}  // namespace mxnet
