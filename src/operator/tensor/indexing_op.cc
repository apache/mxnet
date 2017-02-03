/*!
 * Copyright (c) 2017 by Contributors
 * \file indexing_op.cc
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(EmbeddingParam);
DMLC_REGISTER_PARAMETER(TakeParam);

NNVM_REGISTER_OP(Embedding)
.MXNET_DESCRIBE("Map integer index to vector representations (embeddings)."
                " Those embeddings are learnable parameters. For a input of shape"
                " (d1, ..., dK), the output shape is (d1, ..., dK, output_dim)."
                " All the input values should be integers in the range [0, input_dim).")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", EmbeddingOpShape)
.set_attr<nnvm::FInferType>("FInferType", EmbeddingOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,  const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[0]);
    return MakeGradNode("_backward_Embedding", n, heads, n->attrs.dict);
  })
.add_argument("data", "Symbol", "Input data to the EmbeddingOp.")
.add_argument("weight", "Symbol", "Embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Embedding)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpBackward<cpu>);


NNVM_REGISTER_OP(take)
.MXNET_DESCRIBE("Take row vectors from an NDArray according to the indices"
                " For an input of index with shape (d1, ..., dK), the output"
                " shape is (d1, ..., dK, row_vector_length).All the input"
                " values should be integers in the range"
                " [0, column_vector_length).")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(TakeParamParser<TakeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", TakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", TakeOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TakeOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,  const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[1]);
    return MakeGradNode("_backward_take", n, heads, n->attrs.dict);
  })
.add_argument("a", "Symbol", "The source array.")
.add_argument("indices", "Symbol", "The indices of the values to extract.")
.add_arguments(TakeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_take)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TakeOpBackward<cpu>);


NNVM_REGISTER_OP(batch_take)
.MXNET_DESCRIBE(
  "Take scalar value from a batch of data vectos according to "
  "an index vector, i.e. out[i] = a[i, indices[i]]. out of bound "
  "indices are clipped to boundary.")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BatchTakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", BatchTakeOpType)
.set_attr<FCompute>("FCompute<cpu>", BatchTakeOpForward<cpu>)
.add_argument("a", "NDArray", "Input data array")
.add_argument("indices", "NDArray", "index array");

}  // namespace op
}  // namespace mxnet
