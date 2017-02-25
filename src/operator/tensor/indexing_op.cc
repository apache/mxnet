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
DMLC_REGISTER_PARAMETER(OneHotParam);

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
.describe(R"code(Take elements from an array along an axis.

Slice along a particular axis with the provided indices. E.g., given an input array
with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, then the output
will have shape ``(i0, i1, d1, d2)``, with::

  output[i,j,:,:] = input[indices[i,j],:,:]

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

 take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
                            [ 3.,  4.]],

                           [[ 3.,  4.],
                            [ 5.,  6.]]]

.. note::
  Only slicing axis 0 is supported now.

)code" ADD_FILELINE)
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
.add_argument("a", "ndarray-or-symbol", "The source array.")
.add_argument("indices", "ndarray-or-symbol", "The indices of the values to extract.")
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
.describe(R"code(Take elements from a data batch.

Given an ``(d0, d1)`` input array, and ``(d0,)`` indices, the output will be a
``(d0,)`` computed by::

  output[i] = input[i, indices[i]]

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  batch_take(x, [0,1,0]) = [ 1.  4.  5.]

)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BatchTakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", BatchTakeOpType)
.set_attr<FCompute>("FCompute<cpu>", BatchTakeOpForward<cpu>)
.add_argument("a", "ndarray-or-symbol", "Input data array")
.add_argument("indices", "ndarray-or-symbol", "index array");

NNVM_REGISTER_OP(one_hot)
.describe(R"code(Returns a one-hot array.

The locations represented by ``indices`` take value ``on_value``, while all
other locations take value ``off_value``.

Assume ``indices`` has shape ``(i0, i1)``, then the output will have shape
``(i0, i1, depth)`` and::

  output[i,j,:] = off_value
  output[i,j,indices[i,j]] = on_value

Examples::

  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
                           [ 1.  0.  0.]
                           [ 0.  0.  1.]
                           [ 1.  0.  0.]]

  one_hot([1,0,2,0], 3, on_value=8, off_value=1,
          dtype='int32') = [[1 8 1]
                            [8 1 1]
                            [1 1 8]
                            [8 1 1]]

  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  0.  1.]
                                      [ 1.  0.  0.]]]
)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<OneHotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", OneHotOpShape)
.set_attr<nnvm::FInferType>("FInferType", OneHotOpType)
.set_attr<FCompute>("FCompute<cpu>", OneHotOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("indices", "ndarray-or-symbol", "array of locations where to set on_value")
.add_arguments(OneHotParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
