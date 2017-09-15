/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
DMLC_REGISTER_PARAMETER(ScatterNDParam);

NNVM_REGISTER_OP(Embedding)
.describe(R"code(Maps integer indices to vector representations (embeddings).

This operator maps words to real-valued vectors in a high-dimensional space,
called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
For example, it has been noted that in the learned embedding spaces, similar words tend
to be close to each other and dissimilar words far apart.

For an input array of shape (d1, ..., dK),
the shape of an output array is (d1, ..., dK, output_dim).
All the input values should be integers in the range [0, input_dim).

If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
(ip0, op0).

By default, if any index mentioned is too large, it is replaced by the index that addresses
the last vector in an embedding matrix.

Examples::

  input_dim = 4
  output_dim = 5

  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  y = [[  0.,   1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.,   9.],
       [ 10.,  11.,  12.,  13.,  14.],
       [ 15.,  16.,  17.,  18.,  19.]]

  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  x = [[ 1.,  3.],
       [ 0.,  2.]]

  // Mapped input x to its vector representation y.
  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                            [ 15.,  16.,  17.,  18.,  19.]],

                           [[  0.,   1.,   2.,   3.,   4.],
                            [ 10.,  11.,  12.,  13.,  14.]]]

)code" ADD_FILELINE)
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
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_Embedding", n, ograds,
                               {n->inputs[0]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array to the embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
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
.describe(R"code(Takes elements from an input array along the given axis.

This function slices the input array along a particular axis with the provided indices.

Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the output
will have shape ``(i0, i1, d1, d2)``, computed by::

  output[i,j,:,:] = input[indices[i,j],:,:]

.. note::
   - `axis`- Only slicing along axis 0 is supported for now.
   - `mode`- Only `clip` mode is supported for now.

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // takes elements with specified indices along axis 0
  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
                             [ 3.,  4.]],

                            [[ 3.,  4.],
                             [ 5.,  6.]]]

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
    return MakeNonlossGradNode("_backward_take", n, ograds,
                               {n->inputs[1]}, n->attrs.dict);
  })
.add_argument("a", "NDArray-or-Symbol", "The input array.")
.add_argument("indices", "NDArray-or-Symbol", "The indices of the values to be extracted.")
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
.describe(R"code(Takes elements from a data batch.

.. note::
  `batch_take` is deprecated. Use `pick` instead.

Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
an output array of shape ``(i0,)`` with::

  output[i] = input[i, indices[i]]

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // takes elements with specified indices
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
.add_argument("a", "NDArray-or-Symbol", "The input array")
.add_argument("indices", "NDArray-or-Symbol", "The index array");

NNVM_REGISTER_OP(one_hot)
.describe(R"code(Returns a one-hot array.

The locations represented by `indices` take value `on_value`, while all
other locations take value `off_value`.

`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
in an output array of shape ``(i0, i1, d)`` with::

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
.add_argument("indices", "NDArray-or-Symbol", "array of locations where to set on_value")
.add_arguments(OneHotParam::__FIELDS__());


NNVM_REGISTER_OP(gather_nd)
.describe(R"code(Gather elements or slices from `data` and store to a tensor whose
shape is defined by `indices`. `gather_nd` and `scatter_nd` are inverse functions
to each other.

Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,
where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.

The elements in output is defined as follows::

  output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],
                                                      ...,
                                                      indices[M-1, y_0, ..., y_{K-1}],
                                                      x_M, ..., x_{N-1}]

Examples::

  data = [[0, 1], [2, 3]]
  indices = [[1, 1, 0], [0, 1, 0]]
  gather_nd(data, indices) = [2, 3, 0]

)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", GatherNDShape)
.set_attr<nnvm::FInferType>("FInferType", GatherNDType)
.set_attr<FCompute>("FCompute<cpu>", GatherNDForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("scatter_nd");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{zero, 0, 0});
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices");


NNVM_REGISTER_OP(scatter_nd)
.describe(R"code(Scatters data into a new tensor according to indices.
`gather_nd` and `scatter_nd` are inverse functions to each other.

Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

The elements in output is defined as follows::

  output[indices[0, y_0, ..., y_{K-1}],
         ...,
         indices[M-1, y_0, ..., y_{K-1}],
         x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

all other entries in output are 0.

Examples::

  data = [2, 3, 0]
  indices = [[1, 1, 0], [0, 1, 0]]
  scatter_nd(data, indices) = [[0, 0], [2, 3]]

)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr_parser(ParamParser<ScatterNDParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", ScatterNDShape)
.set_attr<nnvm::FInferType>("FInferType", ScatterNDType)
.set_attr<FCompute>("FCompute<cpu>", ScatterNDForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("gather_nd");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{zero, 0, 0});
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices")
.add_arguments(ScatterNDParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
