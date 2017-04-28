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
#ifndef MXNET_OPERATOR_CONTRIB_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_CONTRIB_DEQUANTIZE_INL_H_

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

struct dequantize {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, const SrcDType *in,
                                  float *imin_range, float *imax_range,
                                  double imin_limit, double imax_limit,
                                  float half_range) {
    float scale = (*imax_range - *imin_range) / (imax_limit - imin_limit);
    out[i] = static_cast<DstDType>((in[i] + half_range) * scale + *imin_range);
  }
};

// keep zero-center
struct dequantize_v2 {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, const SrcDType *in,
                                  const float *imin_range, const float *imax_range) {
    // out[i] = QuantizeToFloat(in[i], *imin_range, *imax_range);
    float real_range = MaxAbs(*imax_range, *imin_range);
    float quantized_range = MinAbs(MinValue<SrcDType>(), MaxValue<SrcDType>());
    float scale = real_range / quantized_range;
    out[i] = in[i] * scale;
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
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);
  // for now, only supports dequantize from int8 to float
  typedef int8_t SrcDType;
  typedef float  DstDType;

  Kernel<dequantize_v2, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DstDType>(),
    inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>());
}

inline bool DequantizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);

  CHECK(!shape_is_none(in_attrs->at(0)));
  for (size_t i = 1; i < 3; ++i) {
    CHECK(shape_is_scalar(in_attrs->at(i))) << in_attrs->at(i);
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  return true;
}

inline bool DequantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK(in_attrs->at(0) == mshadow::kInt8 ||
        in_attrs->at(0) == mshadow::kInt32)
    << "`dequantize` only supports int8 or int32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "the second input of `dequantize` should be a tensor with type of float";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
    << "the third input of `dequantize` should be a tensor with type of float";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_DEQUANTIZE_INL_H_
