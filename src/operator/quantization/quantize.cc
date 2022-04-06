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
 * \file quantize.cc
 * \brief
 */
#include "./quantize-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "./dnnl/dnnl_quantize-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizeParam);

bool QuantizeStorageType(const nnvm::NodeAttrs& attrs,
                         const int dev_mask,
                         DispatchMode* dispatch_mode,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_ONEDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  (*out_attrs)[0] = kDefaultStorage;
  (*out_attrs)[1] = kDefaultStorage;
  (*out_attrs)[2] = kDefaultStorage;
  return true;
}

#if MXNET_USE_ONEDNN == 1
static void QuantizeComputeExCPU(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);

  if (SupportDNNLQuantize(param.out_type)) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLQuantizeCompute, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(QuantizeCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(QuantizeCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

NNVM_REGISTER_OP(_contrib_quantize)
    .add_alias("_npx_contrib_quantize")
    .describe(R"code(Quantize a input tensor from float to `out_type`,
with user-specified `min_range` and `max_range`.

min_range and max_range are scalar floats that specify the range for
the input data.

When out_type is `uint8`, the output is calculated using the following equation:

`out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range) + 0.5`,

where `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`.

When out_type is `int8`, the output is calculate using the following equation
by keep zero centered for the quantized value:

`out[i] = sign(in[i]) * min(abs(in[i] * scale + 0.5f, quantized_range)`,

where
`quantized_range = MinAbs(max(int8), min(int8))` and
`scale = quantized_range / MaxAbs(min_range, max_range).`

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<QuantizeParam>)
    .set_num_inputs(3)
    .set_num_outputs(3)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"data", "min_range", "max_range"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizeShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizeType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizeStorageType)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", QuantizeComputeExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", QuantizeCompute<cpu>)
    .add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
    .add_argument("min_range",
                  "NDArray-or-Symbol",
                  "The minimum scalar value "
                  "possibly produced for the input")
    .add_argument("max_range",
                  "NDArray-or-Symbol",
                  "The maximum scalar value "
                  "possibly produced for the input")
    .add_arguments(QuantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
