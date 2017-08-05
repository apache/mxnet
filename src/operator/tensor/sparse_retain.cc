/*!
 * Copyright (c) 2017 by Contributors
 * \file sparse_retain.cc
 * \brief
*/

#include "./sparse_retain-inl.h"
namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sparse_retain)
.describe(R"code(pick rows specified by user input index array from a row sparse matrix
and save them in the output sparse matrix.

Example::

  data = [[1, 2], [3, 4], [5, 6]]
  indices = [0, 1, 3]
  shape = (4, 2)
  rsp_in = row_sparse(data, indices)
  to_retain = [0, 3]
  rsp_out = sparse_retain(rsp_in, to_retain)
  rsp_out.values = [[1, 2], [5, 6]]
  rsp_out.indices = [0, 3]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", SparseRetainOpShape)
.set_attr<nnvm::FInferType>("FInferType", SparseRetainOpType)
.set_attr<FInferStorageType>("FInferStorageType", SparseRetainForwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_sparse_retain", n, ograds,
                               {n->inputs[sr::kIdx]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array for sparse_retain operator.")
.add_argument("indices", "NDArray-or-Symbol", "The index array of rows ids that will be retained.");

NNVM_REGISTER_OP(_backward_sparse_retain)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", SparseRetainBackwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpBackwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
