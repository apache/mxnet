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

#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyReduceAxesParam : public dmlc::Parameter<NumpyReduceAxesParam> {
  dmlc::optional<nnvm::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<nnvm::Tuple<int>>())
      .describe(R"code()code");
    DMLC_DECLARE_FIELD(dtype).set_default(dmlc::optional<int>())
      .describe("");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
  }
};

inline TShape NumpyReduceAxesShapeImpl(const TShape& ishape,
                                       const dmlc::optional<nnvm::Tuple<int>>& axis,
                                       bool keepdims) {
  // TODO(junwu): improve the logic
  // If input is a scalar, output should be a scalar too
  if (ishape.ndim() == 0) {
    if (axis.has_value()) {
      const nnvm::Tuple<int>& axes = axis.value();
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
  nnvm::Tuple<int> axes(axis.value());
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

template<typename xpu, typename reducer, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void NumpyReduceAxesCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  ReduceAxesComputeImpl<xpu, reducer, normalize, OP>(ctx, inputs, req, outputs, small);
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
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> igrad = outputs[0].FlatTo1D<xpu, DType>(s);
      igrad /= scalar<DType>(outputs[0].Size()/inputs[0].Size());
    });
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
