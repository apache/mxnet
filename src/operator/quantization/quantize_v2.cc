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

#include "./quantize_v2-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_quantize_v2-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizeV2Param);

static bool QuantizeV2StorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                  DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
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

NNVM_REGISTER_OP(_contrib_quantize_v2)
.describe(R"code(Quantize a input tensor from float to `out_type`,
with user-specified `min_calib_range` and `max_calib_range` or the input range collected at runtime.

Output `min_range` and `max_range` are scalar floats that specify the range for the input data.

When out_type is `uint8`, the output is calculated using the following equation:

`out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range) + 0.5`,

where `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`.

When out_type is `int8`, the output is calculate using the following equation
by keep zero centered for the quantized value:

`out[i] = sign(in[i]) * min(abs(in[i] * scale + 0.5f, quantized_range)`,

where
`quantized_range = MinAbs(max(int8), min(int8))` and
`scale = quantized_range / MaxAbs(min_range, max_range).`

When out_type is `auto`, the output type is automatically determined by min_calib_range if presented.
If min_calib_range < 0.0f, the output type will be int8, otherwise will be uint8.
If min_calib_range isn't presented, the output type will be int8.

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<QuantizeV2Param>)
.set_num_inputs(1)
.set_num_outputs(3)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FInferShape>("FInferShape", QuantizeV2Shape)
.set_attr<nnvm::FInferType>("FInferType", QuantizeV2Type)
.set_attr<FInferStorageType>("FInferStorageType", QuantizeV2StorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizeV2Compute)
#endif
.set_attr<FCompute>("FCompute<cpu>", QuantizeV2Compute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  const QuantizeV2Param &param = nnvm::get<QuantizeV2Param>(attrs.parsed);
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    return std::vector<ResourceRequest>();
  } else {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  }
})
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_arguments(QuantizeV2Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
