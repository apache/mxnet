/*!
 * Copyright (c) 2016 by Contributors
 * \file indexing_op.cc
 * \brief
 * \author Siyi Li
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(EmbeddingParam);

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
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_embedding"})
.add_argument("data", "Symbol", "Input data to the EmbeddingOp.")
.add_argument("weight", "Symbol", "Enbedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_embedding)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpBackward<cpu>);
}  // namespace op
}  // namespace mxnet
