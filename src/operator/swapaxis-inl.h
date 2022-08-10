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
 * \file swapaxis-inl.h
 * \brief
 * \author Ming Zhang
 */
#ifndef MXNET_OPERATOR_SWAPAXIS_INL_H_
#define MXNET_OPERATOR_SWAPAXIS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

struct SwapAxisParam : public dmlc::Parameter<SwapAxisParam> {
  // use int for enumeration
  int dim1, dim2;
  DMLC_DECLARE_PARAMETER(SwapAxisParam) {
    DMLC_DECLARE_FIELD(dim1).set_default(0).describe("the first axis to be swapped.");
    DMLC_DECLARE_FIELD(dim2).set_default(0).describe("the second axis to be swapped.");
  }
};

inline void Reshape2Five(mshadow::Shape<5>* inter_shape,
                         const mxnet::TShape& shape,
                         int dim1,
                         int dim2) {
  using namespace mshadow;
  using namespace mshadow::expr;
  int ndim_in = shape.ndim();
  int si;

  if (dim1 > dim2) {
    std::swap(dim1, dim2);
  }

  for (si = 0; si < 5; si++) {
    (*inter_shape)[si] = 1;
  }
  // dim_0
  for (si = 0; si < dim1; si++) {
    (*inter_shape)[0] *= shape[si];
  }
  // dim_1
  (*inter_shape)[1] = shape[dim1];
  // dim_2
  for (si = dim1 + 1; si < dim2; si++) {
    (*inter_shape)[2] *= shape[si];
  }
  // dim_3
  (*inter_shape)[3] = shape[dim2];
  // dim_4
  for (si = dim2 + 1; si < ndim_in; si++) {
    (*inter_shape)[4] *= shape[si];
  }
}

template <typename xpu, typename DType>
void SwapAxis(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const std::vector<TBlob>& in_data,
              const std::vector<TBlob>& out_data,
              const std::vector<OpReqType>& req) {
  using namespace mshadow;
  using namespace mshadow::expr;

  TBlob data_in              = in_data[0];
  TBlob data_out             = out_data[0];
  OpReqType out_req          = req[0];
  Stream<xpu>* s             = ctx.get_stream<xpu>();
  const SwapAxisParam& param = nnvm::get<SwapAxisParam>(attrs.parsed);

  mxnet::TShape shape_in  = data_in.shape_;
  mxnet::TShape shape_out = data_out.shape_;
  int axis1               = param.dim1;
  if (axis1 < 0) {
    axis1 += shape_in.ndim();
  }
  CHECK(axis1 >= 0 && axis1 < shape_in.ndim())
      << "axis1: axis " << param.dim1 << " is out of bounds for array of ndim " << shape_in.ndim();

  int axis2 = param.dim2;
  if (axis2 < 0) {
    axis2 += shape_in.ndim();
  }
  CHECK(axis2 >= 0 && axis2 < shape_in.ndim())
      << "axis2: axis " << param.dim2 << " is out of bounds for array of ndim " << shape_in.ndim();

  if (shape_in.Size() == 0U)
    return;

  if (axis1 == axis2) {
    if (out_req == kAddTo) {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, kAddTo>, xpu>::Launch(
          s, data_out.Size(), data_out.dptr<DType>(), data_in.dptr<DType>());
    } else {
      mxnet_op::copy(s, data_out, data_in);
    }
    return;
  }

  Shape<5> inter_shape;

  Reshape2Five(&inter_shape, shape_in, axis1, axis2);

  Tensor<xpu, 5, DType> inter_data_in = data_in.get_with_shape<xpu, 5, DType>(inter_shape, s);

  Shape<5> inter_shape2 = inter_shape;
  std::swap(inter_shape2[1], inter_shape2[3]);

  Tensor<xpu, 5, DType> inter_data_out = data_out.get_with_shape<xpu, 5, DType>(inter_shape2, s);

  if (out_req == kAddTo) {
    inter_data_out += swapaxis<3, 1>(inter_data_in);
  } else {
    inter_data_out = swapaxis<3, 1>(inter_data_in);
  }
}

template <typename xpu>
void SwapAxisCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& in_data,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH_EXT_WITH_BOOL(
      in_data[0].type_flag_, DType, { SwapAxis<xpu, DType>(attrs, ctx, in_data, out_data, req); });
}

template <typename xpu>
void SwapAxisGrad(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& in_data,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH(
      in_data[0].type_flag_, DType, { SwapAxis<xpu, DType>(attrs, ctx, in_data, out_data, req); });
}

inline bool SwapAxisShape(const nnvm::NodeAttrs& attrs,
                          std::vector<mxnet::TShape>* in_shape,
                          std::vector<mxnet::TShape>* out_shape) {
  CHECK_EQ(in_shape->size(), 1U);
  const SwapAxisParam& param = nnvm::get<SwapAxisParam>(attrs.parsed);

  mxnet::TShape& shape0 = (*in_shape)[0];
  if (!ndim_is_known(shape0))
    return false;
  int axis1 = param.dim1;
  if (axis1 < 0) {
    axis1 += shape0.ndim();
  }
  CHECK(axis1 >= 0 && axis1 < shape0.ndim())
      << "axis1: axis " << param.dim1 << " is out of bounds for array of ndim " << shape0.ndim();

  int axis2 = param.dim2;
  if (axis2 < 0) {
    axis2 += shape0.ndim();
  }
  CHECK(axis2 >= 0 && axis2 < shape0.ndim())
      << "axis2: axis " << param.dim2 << " is out of bounds for array of ndim " << shape0.ndim();

  out_shape->clear();
  out_shape->push_back(shape0);
  mxnet::TShape& shape1 = (*out_shape)[0];

  std::swap(shape1[axis1], shape1[axis2]);

  return shape_is_known(*out_shape);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SWAPAXIS_INL_H_
