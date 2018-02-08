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
  *  Copyright (c) 2017 by Contributors
  * \file bounding_box.cc
  * \brief Bounding box util functions and operators
  * \author Joshua Zhang
  */

#include "./bounding_box-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BoxNMSParam);
DMLC_REGISTER_PARAMETER(BoxOverlapParam);
DMLC_REGISTER_PARAMETER(BipartiteMatchingParam);

NNVM_REGISTER_OP(_contrib_box_nms)
.add_alias("_contrib_box_non_maximum_suppression")
.describe(R"code(Apply non-maximum suppression to input.

The output will be sorted in descending order according to `score`. Boxes with
overlaps larger than `overlap_thresh` and smaller scores will be removed and
filled with -1, the corresponding position will be recorded for backward propogation.

During back-propagation, the gradient will be copied to the original
position according to the input index. For positions that have been suppressed,
the in_grad will be assigned 0.
In summary, gradients are sticked to its boxes, will either be moved or discarded
according to its original index in input.

Input requirements:
1. Input tensor have at least 2 dimensions, (n, k), any higher dims will be regarded
as batch, e.g. (a, b, c, d, n, k) == (a*b*c*d, n, k)
2. n is the number of boxes in each batch
3. k is the width of each box item.

By default, a box is [id, score, xmin, ymin, xmax, ymax, ...],
additional elements are allowed.
- `id_index`: optional, use -1 to ignore, useful if `force_suppress=False`, which means
we will skip highly overlapped boxes if one is `apple` while the other is `car`.
- `coord_start`: required, default=2, the starting index of the 4 coordinates.
Two formats are supported:
  `corner`: [xmin, ymin, xmax, ymax]
  `center`: [x, y, width, height]
- `score_index`: required, default=1, box score/confidence.
When two boxes overlap IOU > `overlap_thresh`, the one with smaller score will be suppressed.
- `in_format` and `out_format`: default='corner', specify in/out box formats.

Examples::

  x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
       [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]
  box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0,
      force_suppress=True, in_format='corner', out_typ='corner') =
      [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
       [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
  out_grad = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
              [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
  # exe.backward
  in_grad = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<BoxNMSParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", BoxNMSNumVisibleOutputs)
.set_attr<nnvm::FInferShape>("FInferShape", BoxNMSShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BoxNMSForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_contrib_box_nms"})
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(BoxNMSParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_box_nms)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<BoxNMSParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BoxNMSBackward<cpu>)
.add_arguments(BoxNMSParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_box_iou)
.describe(R"doc(Bounding box overlap of two arrays.
  The overlap is defined as Intersection-over-Union, aka, IOU.
  - lhs: (a_1, a_2, ..., a_n, 4) array
  - rhs: (b_1, b_2, ..., b_n, 4) array
  - output: (a_1, a_2, ..., a_n, b_1, b_2, ..., b_n) array

  Note::

    Zero gradients are back-propagated in this op for now.

  Example::

    x = [[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]]
    y = [0.25, 0.25, 0.75, 0.75]
    box_iou(x, y, format='corner') = [[0.1428], [0.1428]]

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<BoxOverlapParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BoxOverlapShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", BoxOverlapForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_contrib_box_iou"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(BoxOverlapParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_box_iou)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<BoxOverlapParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BoxOverlapBackward<cpu>)
.add_arguments(BoxOverlapParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_bipartite_matching)
.describe(R"doc(Compute bipartite matching.
  The matching is performed on score matrix with shape [B, N, M]
  - B: batch_size
  - N: number of rows to match
  - M: number of columns as reference to be matched against.

  Returns:
  x : matched column indices. -1 indicating non-matched elements in rows.
  y : matched row indices.

  Note::

    Zero gradients are back-propagated in this op for now.

  Example::

    s = [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
    x, y = bipartite_matching(x, threshold=1e-12, is_ascend=False)
    x = [1, -1, 0]
    y = [2, 0]

)doc" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<BipartiteMatchingParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FInferShape>("FInferShape", MatchingShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<FCompute>("FCompute<cpu>", BipartiteMatchingForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseNone{"_backward_contrib_bipartite_matching"})
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(BipartiteMatchingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_bipartite_matching)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<BipartiteMatchingParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BipartiteMatchingBackward<cpu>)
.add_arguments(BipartiteMatchingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
