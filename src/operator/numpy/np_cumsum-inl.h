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
 * \file np_cumsum-inl.h
 * \brief Function definition of numpy-compatible cumsum operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_CUMSUM_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_CUMSUM_INL_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct CumsumParam : public dmlc::Parameter<CumsumParam> {
  dmlc::optional<int> axis;
  dmlc::optional<int> dtype;
  DMLC_DECLARE_PARAMETER(CumsumParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("Axis along which the cumulative sum is computed."
        " The default (None) is to compute the cumsum over the flattened array.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .set_default(dmlc::optional<int>())
      .describe("Type of the returned array and of the accumulator in which the elements"
                " are summed. If dtype is not specified, it defaults to the dtype of a,"
                " unless a has an integer dtype with a precision less than that of the"
                " default platform integer. In that case, the default platform integer is used.");
  }
};

struct cumsum_forward {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i,
                                  OType *out,
                                  const IType *in,
                                  const int middle,
                                  const int trailing) {
    int left = i / trailing, right = i % trailing;
    int offset = left * middle * trailing + right;
    const IType *lane_in = in + offset;
    OType *lane_out = out + offset;
    lane_out[0] = OType(lane_in[0]);
    for (int j = 1; j < middle; ++j) {
      lane_out[j * trailing] = lane_out[(j - 1) * trailing] + OType(lane_in[j * trailing]);
    }
  }
};

template<typename xpu>
void CumsumForwardImpl(const OpContext& ctx,
                       const TBlob& in,
                       const TBlob& out,
                       const dmlc::optional<int>& axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  int middle = axis.has_value() ? out.shape_[axis.value()] : out.Size();
  if (middle == 0 || out.Size() == 0) return;
  int trailing = 1;
  if (axis.has_value()) {
    for (int i = axis.value() + 1; i < out.shape_.ndim(); ++i) {
      trailing *= out.shape_[i];
    }
  }

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(in.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(out.type_flag_, OType, {
      Kernel<cumsum_forward, xpu>::Launch(
        s, out.Size() / middle, out.dptr<OType>(),
        in.dptr<IType>(), middle, trailing);
    });
  });
}

template<typename xpu>
void CumsumForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  CumsumForwardImpl<xpu>(ctx, inputs[0], outputs[0], param.axis);
}

struct cumsum_backward {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i,
                                  IType *igrad,
                                  const OType *ograd,
                                  const int middle,
                                  const int trailing) {
    int left = i / trailing, right = i % trailing;
    int offset = left * middle * trailing + right;
    const OType *lane_ograd = ograd + offset;
    IType *lane_igrad = igrad + offset;
    lane_igrad[(middle - 1) * trailing] = IType(lane_ograd[(middle - 1) * trailing]);
    for (int j = middle - 2; j >= 0; --j) {
      lane_igrad[j * trailing] = lane_igrad[(j + 1) * trailing] + IType(lane_ograd[j * trailing]);
    }
  }
};

template<typename xpu>
void CumsumBackwardImpl(const OpContext& ctx,
                        const TBlob& ograd,
                        const TBlob& igrad,
                        const dmlc::optional<int>& axis) {
  using namespace mshadow;
  using namespace mxnet_op;
  int middle = axis.has_value() ? igrad.shape_[axis.value()] : igrad.Size();
  if (middle == 0 || igrad.Size() == 0) return;
  int trailing = 1;
  if (axis.has_value()) {
    for (int i = axis.value() + 1; i < igrad.shape_.ndim(); ++i) {
      trailing *= igrad.shape_[i];
    }
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(igrad.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, OType, {
      Kernel<cumsum_backward, xpu>::Launch(
        s, igrad.Size() / middle, igrad.dptr<IType>(),
        ograd.dptr<OType>(), middle, trailing);
    });
  });
}

template<typename xpu>
void CumsumBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  CumsumBackwardImpl<xpu>(ctx, inputs[0], outputs[0], param.axis);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_CUMSUM_INL_H_
