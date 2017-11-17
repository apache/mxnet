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
 * \file quantize.cc
 * \brief
 */
#include "./quantize-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizeParam);

NNVM_REGISTER_OP(_contrib_quantize)
.describe(R"code(Quantize a input tensor from float to `out_type`,
with user-specified `min_range` and `max_range`.

[min_range, max_range] are scalar floats that spcify the range for
the input data. Each value of the tensor will undergo the following:

`out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range)`

here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<QuantizeParam>)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<nnvm::FInferShape>("FInferShape", QuantizeShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizeType)
.set_attr<FCompute>("FCompute<cpu>", QuantizeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_quantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("min_range", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the input")
.add_argument("max_range", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the input")
.add_arguments(QuantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
