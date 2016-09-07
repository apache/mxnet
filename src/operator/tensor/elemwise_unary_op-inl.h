/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
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
                 {n->inputs[0], ograds[0]})
      };
  }
};

struct UnaryGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return std::vector<nnvm::NodeEntry>{
        MakeNode(op_name, n->attrs.name + "_backward",
                 {nnvm::NodeEntry{n, 0, 0}, ograds[0]})
      };
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_INL_H_
