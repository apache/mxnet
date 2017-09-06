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
 * \file quantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  int   out_type;
  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("int8", mshadow::kInt8)
    .add_enum("uint8", mshadow::kUint8)
    .set_default(mshadow::kUint8)
    .describe("Output data type.");
  }
};

// quantize float to uint8_t
struct quantize_unsigned {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range,
                                  float *omax_range, const SrcDType *in,
                                  const float *imin_range, const float *imax_range,
                                  const double min_limit, const double max_limit) {
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    const float scale = (max_limit - min_limit) / (*imax_range - *imin_range);
    out[i] = static_cast<DstDType>((in[i] - *imin_range) * scale + 0.5);
    *omin_range = *imin_range;
    *omax_range = *imax_range;
  }
};


// keep zero-center
struct quantize_zero_centered {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range,
                                  float *omax_range, const SrcDType *in,
                                  const float *imin_range, const float *imax_range,
                                  const float quantized_range) {
    float real_range = MaxAbs(*imin_range, *imax_range);
    float scale = quantized_range / real_range;
    SrcDType x = in[i];
    out[i] = static_cast<DstDType>(
        Sign(x) * Min(Abs(x) * scale + 0.5f, quantized_range));
    *omin_range = -real_range;
    *omax_range =  real_range;
  }
};

template<typename xpu>
void QuantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  if (param.out_type == mshadow::kUint8) {
    Kernel<quantize_unsigned, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<uint8_t>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[0].dptr<float>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      MinValue<uint8_t>(), MaxValue<uint8_t>());
  } else if (param.out_type == mshadow::kInt8) {  // zero-centered quantization
    Kernel<quantize_zero_centered, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<int8_t>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[0].dptr<float>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
  } else {
    LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
  }
}

inline bool QuantizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);

  for (size_t i = 1; i < 3; ++i) {
    SHAPE_ASSIGN_CHECK(*in_attrs, i, TShape({1}));
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape{1});
  return !shape_is_none(out_attrs->at(0));
}

inline bool QuantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  if (param.out_type == mshadow::kUint8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  } else if (param.out_type == mshadow::kInt8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  } else {
    LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_INL_H_
