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
 * \file dequantize.cc
 * \brief
 */
#include "./dequantize-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DequantizeParam);

NNVM_REGISTER_OP(_contrib_dequantize)
.describe(R"code(Dequantize the input tensor into a float tensor.
[min_range, max_range] are scalar floats that spcify the range for
the output data.

Each value of the tensor will undergo the following:

`out[i] = min_range + (in[i] * (max_range - min_range) / range(INPUT_TYPE))`

here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DequantizeParam>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", DequantizeShape)
.set_attr<nnvm::FInferType>("FInferType", DequantizeType)
.set_attr<FCompute>("FCompute<cpu>", DequantizeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_dequantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `uint8`")
.add_argument("min_range", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the input")
.add_argument("max_range", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the input")
.add_arguments(DequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
