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
 * Copyright (c) 2020 by Contributors
 * \file np_matrix_rank.cc
 * \brief CPU implementation of the matrix_rank Operator
 */
#include "./np_matrix_rank-inl.h"

namespace mxnet {
namespace op {

inline bool MatrixRankNoneTolShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector *in_attrs,
                                   mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const int a_ndim = a_shape.ndim();

  if (shape_is_known(a_shape)) {
    CHECK_GT(a_shape.Size(), 0U)
      << "Not support zero-size input array which has no identity";
    if (a_ndim < 2) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
    } else {
      mxnet::TShape rank_shape(a_ndim - 2, 0);
      for (int i = 0; i < a_ndim - 2; ++i) { rank_shape[i] = a_shape[i]; }
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, rank_shape);
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool MatrixRankNoneTolType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);

  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg.";
  CHECK(a_type == mshadow::kFloat32 || a_type == mshadow::kFloat64)
    << "array type should be float32 or float64.";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(MatrixRankNoneTolParam);

NNVM_REGISTER_OP(_npi_matrix_rank_none_tol)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<MatrixRankNoneTolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"M"};
})
.set_attr<mxnet::FInferShape>("FInferShape", MatrixRankNoneTolShape)
.set_attr<nnvm::FInferType>("FInferType", MatrixRankNoneTolType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", MatrixRankNoneTolForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("M", "NDArray-or-Symbol", "Tensor of matrix")
.add_arguments(MatrixRankNoneTolParam::__FIELDS__());

inline bool MatrixRankShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_attrs,
                            mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const mxnet::TShape& tol_shape = (*in_attrs)[1];
  const int a_ndim = a_shape.ndim();
  const int tol_ndim = tol_shape.ndim();

  if (shape_is_known(a_shape) && shape_is_known(tol_shape)) {
    CHECK_GT(a_shape.Size(), 0U)
      << "Not support zero-size input array which has no identity";
    if (a_ndim < 2) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
    } else {
      mxnet::TShape broadcast_shape;
      GetOrCheckBroadcastShape(attrs, a_shape, tol_shape, &broadcast_shape);
      if (broadcast_shape.ndim() == 1) {
        if (tol_ndim == 0) {
          SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
        } else {
          SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 1));
        }
      } else {
        mxnet::TShape rank_shape(broadcast_shape.ndim() - 1, 0);
        for (int i = 0; i < broadcast_shape.ndim() - 1; ++i) {
          rank_shape[i] = broadcast_shape[i];
        }
        SHAPE_ASSIGN_CHECK(*out_attrs, 0, rank_shape);
      }
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool MatrixRankType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  int tol_type = in_attrs->at(1);

  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg.";
  CHECK(a_type == mshadow::kFloat32 || a_type == mshadow::kFloat64)
    << "array type should be float32 or float64.";
  CHECK(tol_type == mshadow::kFloat32 || tol_type == mshadow::kFloat64)
    << "tol type should be float32 or float64.";
  CHECK_EQ(a_type, tol_type)
    << "array type and tol type should be the same.";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(MatrixRankParam);

NNVM_REGISTER_OP(_npi_matrix_rank)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<MatrixRankParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"M", "tol"};
})
.set_attr<mxnet::FInferShape>("FInferShape", MatrixRankShape)
.set_attr<nnvm::FInferType>("FInferType", MatrixRankType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", MatrixRankForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("M", "NDArray-or-Symbol", "Tensor of matrix")
.add_argument("tol", "NDArray-or-Symbol", "Tensor of matrix")
.add_arguments(MatrixRankParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
