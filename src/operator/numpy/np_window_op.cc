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
 *  Copyright (c) 2019 by Contributors
 * \file np_window_op.cc
 * \brief CPU Implementation of unary op hanning, hamming, blackman window.
 */

#include "np_window_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyWindowsParam);

inline bool NumpyWindowsShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_shapes,
                              mxnet::ShapeVector* out_shapes) {
  const NumpyWindowsParam& param = nnvm::get<NumpyWindowsParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK(param.M.has_value()) << "missing 1 required positional argument: 'M'";
  int64_t out_size = param.M.value() <= 0 ? 0 : param.M.value();
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, mxnet::TShape({static_cast<nnvm::dim_t>(out_size)}));
  return true;
}

NNVM_REGISTER_OP(_npi_hanning)
.describe("Return the Hanning window."
          "The Hanning window is a taper formed by using a weighted cosine.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyWindowsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWindowsShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<NumpyWindowsParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyWindowCompute<cpu, 0>)
.add_arguments(NumpyWindowsParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_hamming)
.describe("Return the Hamming window."
          "The Hamming window is a taper formed by using a weighted cosine.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyWindowsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWindowsShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<NumpyWindowsParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyWindowCompute<cpu, 1>)
.add_arguments(NumpyWindowsParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_blackman)
.describe("Return the Blackman window."
          "The Blackman window is a taper formed by using a weighted cosine.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyWindowsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWindowsShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<NumpyWindowsParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyWindowCompute<cpu, 2>)
.add_arguments(NumpyWindowsParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
