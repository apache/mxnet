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
* Copyright (c) 2018 by Contributors
* \file digitize_op.cc
* \brief
* \author Contributors
*/

#include "./digitize_op.h"
#include <mxnet/base.h>
#include <vector>
#include <algorithm>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DigitizeParam);

NNVM_REGISTER_OP(digitize)
    .describe(R"code(Return the indices of the bins to which each value in the input tensor
belongs.

Each index i returned is such that bins[i-1] <= x < bins[i]. For values of X beyond the
bounds of bins, 0 or len(bins) is returned as appropriate. If right is True, then the right bin
is closed, resulting in bins[i-1] <= x < bins[i].

.. Parameters:
  - right: whether the right edges of bins should be included in the interval.
  - otype: type of the output tensor (must be an integer type).

.. Input:
  - X: K-dimensional data tensor to be quantized. Can have any arbitrary shape.
  - bins: N-dimensional tensor containing the bin edges. The first N-1 dimensions must be the
same as those of the input data X. The last dimension corresponds to the vectors containing the
bin edges. Within this last dimension, bins must be strictly monotonically increasing.

.. Requirements:
 - (1 <= N <= K)

.. Output:
  - Tensor of the same shape as the input X containing the indices.

.. Example usage:
  >>> import mxnet as mx
  >>> x = [-2, 17.3, 5, 0.5]
  >>> bins = [0, 5, 10]
  >>> mx.nd.tensor.digitize([mx.nd.array(x), bins], right=True)
  [0, 3, 1, 1]

  >>> import mxnet as mx
  >>> x = [[-2, 17.3, 5, 0.5],
           [10, 30.1, 2, 1.3]]
  >>> bins = [[0, 5, 10],
              [-10, 0, 10]]
  >>> mx.nd.tensor.digitize([mx.nd.array(x), bins])
  [[0, 3, 2, 1],
   [3, 3, 2, 2]]

)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<DigitizeParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
        // List input names
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs &attrs) {
                                       return std::vector<std::string>{ "data", "bins" };
                                     })
    .set_attr<nnvm::FInferShape>("FInferShape", DigitizeOpShape)
    .set_attr<nnvm::FInferType>("FInferType", DigitizeOpType)
    .set_attr<FCompute>("FCompute", DigitizeOpForward<cpu>)
    .add_argument("data", "NDArray-or-Symbol", "Input data ndarray")
    .add_argument("bins", "NDArray-or-Symbol", "Bins ndarray")
    .add_arguments(DigitizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
