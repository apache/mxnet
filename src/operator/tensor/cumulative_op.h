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
 *  Copyright (c) 2015 by Contributors
 * \file cumulative_op.h
 * \brief Function definition of cumulative operators
 */
#ifndef MXNET_OPERATOR_TENSOR_CUMULATIVE_OP_H_
#define MXNET_OPERATOR_TENSOR_CUMULATIVE_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct CumsumParam : public dmlc::Parameter<CumsumParam> {
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(CumsumParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("int or None. The axis along which the cumulative sum"
                "is to be calculated."
                "If is `None`, calculate the sum over the flattened input");
  }
};

inline bool CumSumOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
   SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
   return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
  // SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  // const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  // if (param.axis.has_value()) {
  //   return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
  // } else {
  //   TShape out_shape(1, in_attrs->at(0).Size());
  //   SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  //   return shape_is_known(out_attrs->at(0));
  // }
}

inline bool CumSumOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  // if (param.dtype.has_value()) {
  //   TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  // } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  // }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

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

inline bool CumSumOpForwardStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const CumsumParam& param = nnvm::get<CumsumParam>(attrs.parsed);
  const int in_stype = in_attrs->at(0);
  int& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && in_stype == kDefaultStorage) {
    // dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  // if (!dispatched && in_stype == kCSRStorage && param.c == 0.0) {
  //   // csr -> csr
  //   dispatched = storage_type_assign(&out_stype, kCSRStorage,
  //                                    dispatch_mode, DispatchMode::kFComputeEx);
  // }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

// template<int req>
// struct cumsum_forward {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data) {
//     KERNEL_ASSIGN(out_data[i], req, in_data[i] * (a * in_data[i] + b) + c);
//   }
// };

template<typename xpu>
void CumsumForwardOpImpl(const OpContext& ctx,
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
void CumSumOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  // mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // const TBlob& in_data = inputs[0];
  // const TBlob& out_data = outputs[0];
  const CumsumParam& param = nnvm::get<CumsumParam>(attrs.parsed);
  CumsumForwardImpl<xpu>(ctx, inputs[0], outputs[0], param.axis);
  // MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
  //   MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
  //     // Kernel<cumsum_forward<req_type>, xpu>::Launch(
  //     //     s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
  //     //     param.a, param.b, param.c);
  //   });
  // });
  // if (!param.axis) {
    // for (size_t j = 0; j < in_data.ndim(); ++j) {
    //   for (size_t j = 0; j < in_data[0]; ++j) {
    //     for (size_t k = 0; k < in_data[1]; ++k) {
    //   out_data[j][k] += in_data[j][k];
    // }
  // } else {

  // }
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
void CumsumBackwardOpImpl(const OpContext& ctx,
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
void CumsumOpBackward(const nnvm::NodeAttrs& attrs,
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

  CumsumBackwardOpImpl<xpu>(ctx, inputs[0], outputs[0], param.axis);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CUMULATIVE_OP_H_
