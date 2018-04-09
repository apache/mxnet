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
 * \file dequantize-inl.h
 * \brief Implementation of dequantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DEQUANTIZE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct DequantizeParam : public dmlc::Parameter<DequantizeParam> {
  int out_type;
  DMLC_DECLARE_PARAMETER(DequantizeParam) {
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("float32", mshadow::kFloat32)
    .set_default(mshadow::kFloat32)
    .describe("Output data type.");
  }
};

// dequantize unsigned int8 to float32
struct dequantize_unsigned {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, const SrcDType *in,
                                  const float *imin_range, const float *imax_range,
                                  const float imin_limit, const float imax_limit) {
    const float scale = (*imax_range - *imin_range) / (imax_limit - imin_limit);
    out[i] = static_cast<DstDType>(in[i] * scale + *imin_range);
  }
};

// keep zero-center
struct dequantize_zero_centered {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, const SrcDType *in,
                                  const float *imin_range, const float *imax_range,
                                  const float quantized_range) {
    const float real_range = MaxAbs(*imax_range, *imin_range);
    out[i] = in[i] * (real_range / quantized_range);
  }
};

template<typename xpu>
void DequantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using mshadow::red::limits::MinValue;
  using mshadow::red::limits::MaxValue;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (inputs[0].type_flag_ == mshadow::kUint8) {
    Kernel<dequantize_unsigned, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<float>(),
      inputs[0].dptr<uint8_t>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      MinValue<uint8_t>(), MaxValue<uint8_t>());
  } else if (inputs[0].type_flag_ == mshadow::kInt8) {
    Kernel<dequantize_zero_centered, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<float>(),
      inputs[0].dptr<int8_t>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
  } else {
    LOG(FATAL) << "dequantize op only supports input type int8 or uint8";
  }
}

inline bool DequantizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);

  for (size_t i = 1; i < 3; ++i) {
    SHAPE_ASSIGN_CHECK(*in_attrs, i, TShape({1}));
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  return !shape_is_none(out_attrs->at(0));
}

inline bool DequantizeType(const nnvm::NodeAttrs& attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK(in_attrs->at(0) == mshadow::kUint8 || in_attrs->at(0) == mshadow::kInt8)
    << "the input data type of dequantize op must be provided, either uint8 or int8";
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_DEQUANTIZE_INL_H_
