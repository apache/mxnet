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
 * \file broadcast_reduce_op.h
 * \brief Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_

#include <algorithm>
#include <vector>
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyReduceAxesParam : public dmlc::Parameter<NumpyReduceAxesParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .set_default(dmlc::optional<int>())
      .describe("The type of the returned array and of the accumulator in which the elements are "
                "summed. The dtype of a is used by default unless a has an integer dtype of less "
                "precision than the default platform integer. In that case, if a is signed then "
                "the platform integer is used while if a is unsigned then an unsigned integer of "
                "the same precision as the platform integer is used.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(initial).set_default(dmlc::optional<double>())
      .describe("Starting value for the sum.");
  }
};

inline TShape NumpyReduceAxesShapeImpl(const TShape& ishape,
                                       const dmlc::optional<mxnet::Tuple<int>>& axis,
                                       bool keepdims) {
  // TODO(junwu): improve the logic
  // If input is a scalar, output should be a scalar too
  if (ishape.ndim() == 0) {
    if (axis.has_value()) {
      const mxnet::Tuple<int>& axes = axis.value();
      if (axes.ndim() > 0) {
        CHECK_EQ(axes.ndim(), 1);
        CHECK(axes[0] == 0 || axes[0] == -1);
      }
    }
    return TShape(0, -1);
  }

  // axis=None, do global reduction
  if (!axis.has_value()) {
    if (keepdims) {
      return TShape(ishape.ndim(), 1);
    } else {
      return TShape(0, -1);
    }
  }

  // axis = (), will return identity(input)
  if (axis.value().ndim() == 0) {
    return ishape;
  }

  // axis has value
  mxnet::Tuple<int> axes(axis.value());
  for (index_t i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      axes[i] += ishape.ndim();
    }
  }
  std::sort(axes.begin(), axes.end());

  for (index_t i = 1; i < axes.ndim(); i++) {
    CHECK_LT(axes[i-1], axes[i])
        << "Reduction axes have duplicates "
        << axes;
  }
  CHECK_LT(axes[axes.ndim()-1], ishape.ndim())
      << "Reduction axis " << axes[axes.ndim()-1]
      << " Exceeds input dimensions " << ishape;
  CHECK_GE(axes[0], 0)
      << "Reduction axis " << axis.value()
      << " Exceeds input dimensions " << ishape;

  TShape oshape;
  if (keepdims) {
    oshape = TShape(ishape);
  } else {
    oshape = TShape(ishape.ndim() - axes.ndim(), -1);
  }

  if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
    }
  } else {
    for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[k++] = ishape[i];
    }
  }
  return oshape;
}

inline bool NumpyReduceAxesShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

template<bool safe_acc_hint = false>
inline bool NeedSafeAcc(int itype, int otype) {
  bool rule = (itype != otype) || (itype != mshadow::kFloat32 && itype != mshadow::kFloat64);
  return safe_acc_hint && rule;
}

template<typename xpu, typename reducer, bool safe_acc_hint = false, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void NumpyReduceAxesCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  if (NeedSafeAcc<safe_acc_hint>(inputs[0].type_flag_, outputs[0].type_flag_)) {
    ReduceAxesComputeImpl<xpu, reducer, true, normalize, OP>(ctx, inputs, req, outputs, small);
  } else {
    ReduceAxesComputeImpl<xpu, reducer, false, normalize, OP>(ctx, inputs, req, outputs, small);
  }
}

template<typename xpu, bool normalize = false>
inline void NumpyReduceAxesBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const std::vector<TBlob>& inputs,
                                           const std::vector<OpReqType>& req,
                                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }

  BroadcastComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, small);
  if (normalize) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, IType, {
      Tensor<xpu, 1, IType> igrad = outputs[0].FlatTo1D<xpu, IType>(s);
      igrad /= scalar<IType>(outputs[0].Size()/inputs[0].Size());
    });
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
