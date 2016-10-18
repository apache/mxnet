/*!
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.cc
 * \brief elementwise sum operator
*/
#include "./elemwise_sum.h"

namespace mxnet {
namespace op {

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
        .describe("Number of inputs to be summed.");
  }
};

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

std::vector<nnvm::NodeEntry> ElementWiseSumGrad(
    const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& ograds) {
  // identity constraints in the beginning for easier shape inference.
  const nnvm::Op* backward_copy_op =
      nnvm::Op::Get("_backward_copy");
  CHECK_EQ(ograds.size(), 1);
  std::vector<nnvm::NodeEntry> ret;
  nnvm::NodeEntry n_out{n, 0, 0};
  for (auto& h : n->inputs) {
    nnvm::NodePtr id_node = nnvm::Node::Create();
    id_node->attrs.op = backward_copy_op;
    id_node->control_deps.push_back(n);
    id_node->inputs = {ograds[0]};
    ret.push_back(nnvm::NodeEntry{id_node, 0, 0});
  }
  return ret;
}

bool ElementWiseSumShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<TShape, shape_is_none, true>(
      attrs, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(ElementWiseSum)
.MXNET_DESCRIBE("Perform element sum of inputs")
.set_attr_parser(ParamParser<ElementWiseSumParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
    return ret;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FCompute>("FCompute<cpu>", ElementWiseSumCompute<cpu>)
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FInferShape>("FInferShape", ElementWiseSumShape)
.set_attr<nnvm::FGradient>("FGradient", ElementWiseSumGrad);



}  // namespace op
}  // namespace mxnet
