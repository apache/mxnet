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
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_dequantize-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DequantizeParam);

bool DequantizeStorageType(const nnvm::NodeAttrs& attrs,
                           const int dev_mask,
                           DispatchMode* dispatch_mode,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  (*out_attrs)[0] = kDefaultStorage;
  (*out_attrs)[1] = kDefaultStorage;
  (*out_attrs)[2] = kDefaultStorage;
  return true;
}

NNVM_REGISTER_OP(_contrib_dequantize)
.describe(R"code(Dequantize the input tensor into a float tensor.
min_range and max_range are scalar floats that specify the range for
the output data.

When input data type is `uint8`, the output is calculated using the following equation:

`out[i] = in[i] * (max_range - min_range) / 255.0`,

When input data type is `int8`, the output is calculate using the following equation
by keep zero centered for the quantized value:

`out[i] = in[i] * MaxAbs(min_range, max_range) / 127.0`,

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DequantizeParam>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", DequantizeShape)
.set_attr<nnvm::FInferType>("FInferType", DequantizeType)
.set_attr<FInferStorageType>("FInferStorageType", DequantizeStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNDequantizeCompute)
#endif
.set_attr<FCompute>("FCompute<cpu>", DequantizeCompute<cpu>)
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `uint8`")
.add_argument("min_range", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the input in float32")
.add_argument("max_range", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the input in float32")
.add_arguments(DequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
