/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
template<typename GRAD_OP>
struct unary_backward {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(GRAD_OP::Map(a)*b);
  }
};

struct UnaryGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return std::vector<nnvm::NodeEntry>{
        MakeNode(op_name, n->attrs.name + "_backward",
                 {n->inputs[0], ograds[0]}, n->attrs.dict)
      };
  }
};

struct UnaryGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return std::vector<nnvm::NodeEntry>{
        MakeNode(op_name, n->attrs.name + "_backward",
                 {nnvm::NodeEntry{n, 0, 0}, ograds[0]}, n->attrs.dict)
      };
  }
};

struct UnaryGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return std::vector<nnvm::NodeEntry>{
        MakeNode(op_name, n->attrs.name + "_backward", {ograds[0]}, n->attrs.dict)
      };
  }
};

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}

#define MXNET_OPERATOR_REGISTER_UNARY(name)                     \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                 \
      return std::vector<std::pair<int, int> >{{0, 0}};         \
    })                                                          \
  .add_argument("src", "NDArray", "Source input")


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
