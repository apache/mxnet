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
 * \file quantized_flatten-inl.h
 * \brief implementation of quantized flatten operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FLATTEN_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FLATTEN_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

// keep zero-center
struct quantized_flatten {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range,
                                  float *omax_range, const SrcDType *in,
                                  const float *imin_range, const float *imax_range) {
    out[i] = in[i];
    omin_range[0] = imin_range[0];
    omax_range[0] = imax_range[0];
  }
};

template<typename xpu>
void QuantizedFlattenCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(req.size(), 3U);
  if (req[0] == kWriteInplace && req[1] == kWriteInplace && req[2] == kWriteInplace) return;
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  typedef int8_t DstDType;
  typedef int8_t  SrcDType;
  Kernel<quantized_flatten, xpu>::Launch(s, outputs[0].Size(),
    outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
    inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>());
}

inline bool QuantizedFlattenShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_attrs,
                                  std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);

  const TShape &dshape = (*in_attrs)[0];
  if (shape_is_none(dshape)) return false;

  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }

  SHAPE_ASSIGN_CHECK(*in_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(dshape[0], target_dim));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape{1});
  return true;
}

inline bool QuantizedFlattenType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FLATTEN_INL_H_
