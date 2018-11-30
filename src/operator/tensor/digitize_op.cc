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


template<typename DType, typename OType>
struct ForwardKernel<cpu, DType, OType> {
  static MSHADOW_XINLINE void Map(int i,
                                  DType *in_data,
                                  OType *out_data,
                                  DType *bins,
                                  size_t batch_size,
                                  size_t bins_length,
                                  bool right) {

    const auto data = in_data[i];
    const auto batch = i / batch_size;

    auto
        elem = right ? std::lower_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data)
                     : std::upper_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data);

    out_data[i] = std::distance(bins, elem);
  }
};


DMLC_REGISTER_PARAMETER(DigitizeParam);

NNVM_REGISTER_OP(digitize)
    .describe(R"code(Return the indices of the bins to which each value in the input tensor
belongs.

Each index i returned is such that bins[i-1] <= x < bins[i]. For values of X beyond the
bounds of bins, 0 or len(bins) is returned as appropriate. If right is True, then the right bin
is closed, resulting in bins[i-1] <= x < bins[i].

.. Parameters:
  - right: whether the right edges of bins should be included in the interval.

.. Input:
  - X: data tensor to be quantized. Can have any arbitrary shape. If quantizing in batch mode,
the first dimension should correspond to the batch axis.
  - bins: tensor containing the bin edges. In the 2D case, the first dimension
 should correspond to the batch axis: each batch in X will be quantized using a different set of
bins. Within each batch, bins must be strictly monotonically increasing.

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
    .set_attr<nnvm::FInferShape>("FInferShape", InferShape)
    .set_attr<nnvm::FInferType>("FInferType", DigitizeOpType)
    .set_attr<FCompute>("FCompute", DigitizeOpForward<cpu>)
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_argument("bins", "NDArray-or-Symbol", "Bins ndarray")
    .add_arguments(DigitizeParam::__FIELDS__());
// TODO: Option to specify there's no backward pass?


}  // namespace op
}  // namespace mxnet
