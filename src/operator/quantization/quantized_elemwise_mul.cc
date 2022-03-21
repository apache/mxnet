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
 * \file quantized_elemwise_mul.cc
 * \brief CPU Implementation of basic elementwise binary mul operators
 */
#include <mxnet/op_attr_types.h>
#include "../tensor/elemwise_binary_op-inl.h"
#include "./quantized_elemwise_mul-inl.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizeElemwiseMulParam);

inline bool QuantizedElemwiseMulOpShape(const nnvm::NodeAttrs& attrs,
                                        mxnet::ShapeVector* in_attrs,
                                        mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const QuantizeElemwiseMulParam& params = nnvm::get<QuantizeElemwiseMulParam>(attrs.parsed);
  const mxnet::TShape& lshape            = (*in_attrs)[quantized_elemwise_mul::kLhs];
  const mxnet::TShape& rshape            = (*in_attrs)[quantized_elemwise_mul::kRhs];
  if (!ndim_is_known(lshape) || !ndim_is_known(rshape))
    return false;
  CHECK_EQ(lshape.ndim(), rshape.ndim())
      << "Currently, quantized elemwise multiply doesn't support broadcast.";
  for (int i = 0; i < lshape.ndim(); ++i) {
    CHECK_EQ(lshape[i], rshape[i]);
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_elemwise_mul::kLhsMin, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_elemwise_mul::kLhsMax, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_elemwise_mul::kRhsMin, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_elemwise_mul::kRhsMax, mxnet::TShape(1, 1));

  SHAPE_ASSIGN_CHECK(*out_attrs, quantized_elemwise_mul::kOut, lshape);
  if (!params.enable_float_output) {
    SHAPE_ASSIGN_CHECK(*out_attrs, quantized_elemwise_mul::kOutMin, mxnet::TShape(1, 1));
    SHAPE_ASSIGN_CHECK(*out_attrs, quantized_elemwise_mul::kOutMax, mxnet::TShape(1, 1));
  }
  return true;
}

inline bool QuantizedElemwiseMulOpType(const nnvm::NodeAttrs& attrs,
                                       std::vector<int>* in_type,
                                       std::vector<int>* out_type) {
  const QuantizeElemwiseMulParam& params = nnvm::get<QuantizeElemwiseMulParam>(attrs.parsed);
  for (int i = 0; i < 2; ++i) {
    if (in_type->at(i) == mshadow::kInt8) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
    } else {
      LOG(ERROR) << "currently, quantized elemwise mul only support int8 inputs.";
    }
  }
  TYPE_ASSIGN_CHECK(*in_type, quantized_elemwise_mul::kLhsMin, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, quantized_elemwise_mul::kLhsMax, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, quantized_elemwise_mul::kRhsMin, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, quantized_elemwise_mul::kRhsMax, mshadow::kFloat32);

  int dtype = mshadow::kInt32;
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    dtype = mshadow::kInt8;
  }
  if (!params.enable_float_output) {
    TYPE_ASSIGN_CHECK(*out_type, quantized_elemwise_mul::kOut, dtype);
    TYPE_ASSIGN_CHECK(*out_type, quantized_elemwise_mul::kOutMin, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_type, quantized_elemwise_mul::kOutMax, mshadow::kFloat32);
  } else {
    TYPE_ASSIGN_CHECK(*out_type, quantized_elemwise_mul::kOut, mshadow::kFloat32);
  }
  return true;
}

inline bool QuantizedElemwiseMulOpStorageType(const nnvm::NodeAttrs& attrs,
                                              int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs) {
  using namespace common;
  *dispatch_mode = DispatchMode::kFCompute;

  for (auto& v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto& v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
}

void QuantizedElemwiseMulOpForward(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  const QuantizeElemwiseMulParam& params = nnvm::get<QuantizeElemwiseMulParam>(attrs.parsed);
  using namespace mxnet_op;

  float lhs_min = inputs[quantized_elemwise_mul::kLhsMin].dptr<float>()[0];
  float lhs_max = inputs[quantized_elemwise_mul::kLhsMax].dptr<float>()[0];
  float rhs_min = inputs[quantized_elemwise_mul::kRhsMin].dptr<float>()[0];
  float rhs_max = inputs[quantized_elemwise_mul::kRhsMax].dptr<float>()[0];

  float cached_output_min_ = 0.f;
  float cached_output_max_ = 0.f;
  float out_data_scale     = 1.f;
  float out_scale          = 1.f;
  if (!params.enable_float_output) {
    double output_data_range;
    // dataA && dataB are int8
    if (outputs[quantized_elemwise_mul::kOut].type_flag_ == mshadow::kInt8) {
      output_data_range = kInt8Range;
    } else {
      output_data_range = kInt32Range;
    }
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      cached_output_min_ = params.min_calib_range.value();
      cached_output_max_ = params.max_calib_range.value();
      out_data_scale     = output_data_range / MaxAbs(cached_output_min_, cached_output_max_);
      auto lhs_scale     = kInt8Range / MaxAbs(lhs_min, lhs_max);
      auto rhs_scale     = kInt8Range / MaxAbs(rhs_min, rhs_max);
      out_scale          = out_data_scale / lhs_scale / rhs_scale;
    } else {
      Stream<cpu>* s = ctx.get_stream<cpu>();
      if (inputs[quantized_elemwise_mul::kLhs].type_flag_ == mshadow::kInt8 &&
          inputs[quantized_elemwise_mul::kRhs].type_flag_ == mshadow::kInt8) {
        mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
            s, 1, &cached_output_min_, &cached_output_max_, &lhs_min, &lhs_max, &rhs_min, &rhs_max);
      } else {
        LOG(ERROR) << "lhs and rhs only support iny8 dtype.";
      }
    }
  } else {
    auto lhs_scale = kInt8Range / MaxAbs(lhs_min, lhs_max);
    auto rhs_scale = kInt8Range / MaxAbs(rhs_min, rhs_max);
    out_scale      = 1.0 / lhs_scale / rhs_scale;
  }

  size_t out_size = outputs[quantized_elemwise_mul::kOut].Size();
  auto* input_l   = inputs[quantized_elemwise_mul::kLhs].dptr<int8_t>();
  auto* input_r   = inputs[quantized_elemwise_mul::kRhs].dptr<int8_t>();
  // TODO(Xinyu): a temp solution to enable Elemwise INT8 computation,
  // will be refactored after the DNNL primitive is done.
  if (!params.enable_float_output) {
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      typedef int8_t out_type;
      auto* out_data = outputs[quantized_elemwise_mul::kOut].dptr<out_type>();
#if !defined(_MSC_VER)
#pragma omp simd
#endif
      for (size_t i = 0; i < out_size; ++i) {
        const int8_t a = input_l[i];
        const int8_t b = input_r[i];
        out_data[i]    = static_cast<out_type>(a * b * out_scale);
      }
    } else {
      using out_type = int32_t;
      auto* out_data = outputs[quantized_elemwise_mul::kOut].dptr<out_type>();
#if !defined(_MSC_VER)
#pragma omp simd
#endif
      for (size_t i = 0; i < out_size; ++i) {
        const int8_t a = input_l[i];
        const int8_t b = input_r[i];
        out_data[i]    = static_cast<out_type>(a * b * out_scale);
      }
    }
  } else {
    using out_type = float;
    auto* out_data = outputs[quantized_elemwise_mul::kOut].dptr<out_type>();
#if !defined(_MSC_VER)
#pragma omp simd
#endif
    for (size_t i = 0; i < out_size; ++i) {
      const int8_t a = input_l[i];
      const int8_t b = input_r[i];
      out_data[i]    = static_cast<out_type>(a * b * out_scale);
    }
  }

  if (!params.enable_float_output) {
    outputs[quantized_elemwise_mul::kOutMin].dptr<float>()[0] = cached_output_min_;
    outputs[quantized_elemwise_mul::kOutMax].dptr<float>()[0] = cached_output_max_;
  }
}

NNVM_REGISTER_OP(_contrib_quantized_elemwise_mul)
    .add_alias("_npx_quantized_elemwise_mul")
    .describe(R"code(Multiplies arguments int8 element-wise.
)code" ADD_FILELINE)
    .set_num_inputs(6)
    .set_num_outputs([](const NodeAttrs& attrs) {
      const QuantizeElemwiseMulParam& params = nnvm::get<QuantizeElemwiseMulParam>(attrs.parsed);
      return (!params.enable_float_output) ? 3 : 1;
    })
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"lhs", "rhs", "lhs_min", "lhs_max", "rhs_min", "rhs_max"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedElemwiseMulOpShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedElemwiseMulOpType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedElemwiseMulOpStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>", QuantizedElemwiseMulOpForward)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("lhs", "NDArray-or-Symbol", "first input")
    .add_argument("rhs", "NDArray-or-Symbol", "second input")
    .add_argument("lhs_min", "NDArray-or-Symbol", "Minimum value of first input.")
    .add_argument("lhs_max", "NDArray-or-Symbol", "Maximum value of first input.")
    .add_argument("rhs_min", "NDArray-or-Symbol", "Minimum value of second input.")
    .add_argument("rhs_max", "NDArray-or-Symbol", "Maximum value of second input.")
    .set_attr_parser(ParamParser<QuantizeElemwiseMulParam>)
    .add_arguments(QuantizeElemwiseMulParam::__FIELDS__());

NNVM_REGISTER_OP(elemwise_mul).set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_contrib_quantized_elemwise_mul");
  node->attrs.name     = "quantized_" + attrs.name;
  node->attrs.dict     = attrs.dict;
  if (node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
