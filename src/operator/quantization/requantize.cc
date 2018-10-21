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
 * \file requantize.cc
 * \brief
 */
#include "./requantize-inl.h"
#include "./quantize-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_requantize-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(RequantizeParam);

bool RequantizeStorageType(const nnvm::NodeAttrs& attrs,
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

NNVM_REGISTER_OP(_contrib_requantize)
.describe(R"code(Given data that is quantized in int32 and the corresponding thresholds,
requantize the data into int8 using min and max thresholds either calculated at runtime
or from calibration. It's highly recommended to pre-calucate the min and max thresholds
through calibration since it is able to save the runtime of the operator and improve the
inference accuracy.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<RequantizeParam>)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<nnvm::FInferShape>("FInferShape", QuantizeShape)
.set_attr<nnvm::FInferType>("FInferType", RequantizeType)
.set_attr<FInferStorageType>("FInferStorageType", RequantizeStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNRequantizeForward)
#else
.set_attr<FCompute>("FCompute<cpu>", RequantizeForward<cpu>)
#endif
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
    const RequantizeParam& param =
      nnvm::get<RequantizeParam>(attrs.parsed);
    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      return std::vector<ResourceRequest>();
    } else {
      return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
    }
  })
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `int32`")
.add_argument("min_range", "NDArray-or-Symbol", "The original minimum scalar value "
  "in the form of float32 used for quantizing data into int32.")
.add_argument("max_range", "NDArray-or-Symbol", "The original maximum scalar value "
  "in the form of float32 used for quantizing data into int32.")
.add_arguments(RequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
