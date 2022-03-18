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
 * \file quantize_asym.cc
 * \brief implementation of asymmetric quantize operation
 */

#include <string>

#include "operator/quantization/quantize_asym-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/quantization/dnnl/dnnl_quantize_asym-inl.h"
#endif

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizeAsymParam);

inline bool QuantizeAsymShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);

  mxnet::TShape dshape = in_attrs->at(0);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape(1, 1));

  if (out_attrs->at(0).ndim() > 0) {
    dshape[0] = out_attrs->at(0)[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, dshape);
  }

  return !shape_is_none(out_attrs->at(0));
}

inline bool QuantizeAsymType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);

  CHECK_EQ(in_attrs->at(0), mshadow::kFloat32);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);

  return !type_is_none(out_attrs->at(0));
}

bool QuantizeAsymStorageType(const nnvm::NodeAttrs& attrs,
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
  out_attrs->at(0) = kDefaultStorage;
  out_attrs->at(1) = kDefaultStorage;
  out_attrs->at(2) = kDefaultStorage;
  return true;
}

OpStatePtr CreateQuantizeAsymState(const nnvm::NodeAttrs& attrs,
                                   const Context& ctx,
                                   const std::vector<TShape>& in_shapes,
                                   const std::vector<int>& in_types) {
  OpStatePtr state;
  if (ctx.dev_type == kGPU) {
    state = OpStatePtr::Create<QuantizeAsymOp<gpu>>(attrs);
  } else {
#if MXNET_USE_ONEDNN == 1
    if (in_shapes[0].ndim() == 3 && in_types[0] == mshadow::kFloat32) {
      state = OpStatePtr::Create<DNNLQuantizeAsymOp>(attrs);
      return state;
    }
#else
    state = OpStatePtr::Create<QuantizeAsymOp<cpu>>(attrs);
#endif
  }
  return state;
}

NNVM_REGISTER_OP(_contrib_quantize_asym)
    .describe(R"code(Quantize a input tensor from float to uint8_t.
Output `scale` and `shift` are scalar floats that specify the quantization
parameters for the input data. The output is calculated using the following equation:

`out[i] = in[i] * scale + shift + 0.5`,

where `scale = uint8_range / (max_range - min_range)` and
`shift = numeric_limits<T>::max - max_range * scale`.

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<QuantizeAsymParam>)
    .set_num_inputs(1)
    .set_num_outputs(3)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output", "scale", "shift"};
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizeAsymShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizeAsymType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizeAsymStorageType)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FCreateOpState>("FCreateOpState", CreateQuantizeAsymState)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", DNNLQuantizeAsymForward)
#endif
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", QuantizeAsymForward<cpu>)
    .set_attr<FNeedCalibrateInput>("FNeedCalibrateInput",
                                   [](const NodeAttrs& attrs) { return std::vector<int>{0}; })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  const QuantizeAsymParam& param =
                                      nnvm::get<QuantizeAsymParam>(attrs.parsed);
                                  if (param.max_calib_range.has_value() &&
                                      param.max_calib_range.has_value()) {
                                    return std::vector<ResourceRequest>();
                                  } else {
                                    return std::vector<ResourceRequest>(
                                        1, ResourceRequest::kTempSpace);
                                  }
                                })
    .add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
    .add_arguments(QuantizeAsymParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
