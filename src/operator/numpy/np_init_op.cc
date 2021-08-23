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
 *  Copyright (c) 2019 by Contributors
 * \file np_init_op.cc
 * \brief CPU Implementation of numpy init op
 */

#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"
#include "./np_init_op.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(NumpyEyeParam);
DMLC_REGISTER_PARAMETER(IndicesOpParam);
DMLC_REGISTER_PARAMETER(LogspaceParam);
DMLC_REGISTER_PARAMETER(FullLikeOpParam);
DMLC_REGISTER_PARAMETER(AtleastNDParam);

inline bool NumpyIndicesShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_shapes,
                              mxnet::ShapeVector* out_shapes) {
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK_GE(param.dimensions.ndim(), 0)
    << "_npi_indices dimensions the number of dim must not be less than 0";
  mxnet::TShape param_dim = param.dimensions;
  if (!shape_is_known(param_dim)) return false;
  CHECK_LT(param_dim.Size(), INT32_MAX) << "ValueError: np.indices does not support large"
     << " input tensors (containing >= 2^31 elements).";
  const int indim = param.dimensions.ndim();
  mxnet::TShape ret(indim + 1, -1);
  ret[0] = indim;
  for (int i = 1; i < indim + 1; ++i) {
    ret[i] = param_dim[i-1];
  }
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, ret);
  return shape_is_known(out_shapes->at(0));
}

inline bool NumpyIndicesType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype == -1 ? mshadow::kInt64 : param.dtype);
  return true;
}

inline bool LogspaceShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  const LogspaceParam& param = nnvm::get<LogspaceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_GE(param.num, 0)
    << "Number of sequence should be non-negative, received " << param.num;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({static_cast<nnvm::dim_t>(param.num)}));
  return true;
}

NNVM_REGISTER_OP(_npi_zeros)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_ones)
.describe("Return a new array of given shape, type, and context, filled with ones.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_identity)
.describe("Return a new identity array of given shape, type, and context.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.add_arguments(InitOpParam::__FIELDS__());

template<int NDim>
inline bool AtleastNDShape(const nnvm::NodeAttrs& attrs,
                           std::vector<mxnet::TShape> *in_attrs,
                           std::vector<mxnet::TShape> *out_attrs) {
  auto &param = nnvm::get<AtleastNDParam>(attrs.parsed);

  CHECK_EQ(in_attrs->size(), param.num_args);
  CHECK_EQ(out_attrs->size(), param.num_args);

  for (int i = 0; i < param.num_args; ++i) {
    auto &shape = in_attrs->at(i);
    if (shape.ndim() < NDim) {
      mxnet::TShape new_shape(NDim, 1);
      if (NDim == 2) {
        if (shape.ndim() == 1) {
          new_shape[1] = shape[0];
        }
      } else if (NDim == 3) {
        if (shape.ndim() == 1) {
          new_shape[1] = shape[0];
        } else if (shape.ndim() == 2) {
          new_shape[0] = shape[0];
          new_shape[1] = shape[1];
        }
      }
      SHAPE_ASSIGN_CHECK(*out_attrs, i, new_shape);
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, i, shape);
    }
  }

  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

#define NNVM_REGISTER_ATLEAST_ND(N)                                       \
NNVM_REGISTER_OP(_npi_atleast_##N##d)                                      \
.set_attr_parser(ParamParser<AtleastNDParam>)                             \
.set_num_inputs(                                                          \
[](const NodeAttrs& attrs) {                                              \
  auto &param = nnvm::get<AtleastNDParam>(attrs.parsed);                  \
  return param.num_args;                                                  \
})                                                                        \
.set_num_outputs(                                                         \
[](const NodeAttrs& attrs) {                                              \
  auto &param = nnvm::get<AtleastNDParam>(attrs.parsed);                  \
  return param.num_args;                                                  \
})                                                                        \
.set_attr<std::string>("key_var_num_args", "num_args")                    \
.set_attr<nnvm::FListInputNames>("FListInputNames",                       \
[](const nnvm::NodeAttrs& attrs) {                                        \
  int num_args = nnvm::get<AtleastNDParam>(attrs.parsed).num_args;        \
  std::vector<std::string> ret;                                           \
  ret.reserve(num_args);                                                  \
  for (int i = 0; i < num_args; i++) {                                    \
    ret.push_back(std::string("ary") + std::to_string(i));                \
  }                                                                       \
  return ret;                                                             \
})                                                                        \
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, -1>)           \
.set_attr<mxnet::FInferShape>("FInferShape", AtleastNDShape<N>)           \
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                \
.set_attr<FCompute>("FCompute<cpu>", AtleastNDCompute<cpu>)               \
.add_argument("arys", "NDArray-or-Symbol[]", "List of input arrays")      \
.add_arguments(AtleastNDParam::__FIELDS__())                              \

NNVM_REGISTER_ATLEAST_ND(1);

NNVM_REGISTER_ATLEAST_ND(2);

NNVM_REGISTER_ATLEAST_ND(3);

NNVM_REGISTER_OP(_npi_full_like)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FullLikeOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", FullLikeOpType<FullLikeOpParam>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<FCompute>("FCompute<cpu>", FullLikeOpCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol",
              "The shape and data-type of a define these same attributes of the returned array.")
.add_arguments(FullLikeOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_full)
  .describe("fill target with a scalar value")
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr_parser(ParamParser<InitOpWithScalarParam>)
  .set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpWithScalarParam>)
  .set_attr<nnvm::FInferType>("FInferType", InitNumpyType<InitOpWithScalarParam>)
  .set_attr<FCompute>("FCompute<cpu>", InitFillWithScalarCompute<cpu>)
.add_arguments(InitOpWithScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_arange)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(RangeParamParser)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyRangeShape)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<RangeParam>)
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu, RangeParam>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_eye)
.describe("Return a 2-D array with ones on the diagonal and zeros elsewhere.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyEyeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyEyeShape)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<NumpyEyeParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyEyeFill<cpu>)
.add_arguments(NumpyEyeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_indices)
.describe("Return an array representing the indices of a grid.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<IndicesOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyIndicesShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyIndicesType)
.set_attr<FCompute>("FCompute<cpu>", IndicesCompute<cpu>)
.add_arguments(IndicesOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_linspace)
.describe("Return evenly spaced numbers over a specified interval. Similar to Numpy")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LinspaceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LinspaceShape)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<LinspaceParam>)
.set_attr<FCompute>("FCompute<cpu>", LinspaceCompute<cpu>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_logspace)
.describe("Return numbers spaced evenly on a log scale.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LogspaceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LogspaceShape)
.set_attr<nnvm::FInferType>("FInferType", InitNumpyType<LogspaceParam>)
.set_attr<FCompute>("FCompute<cpu>", LogspaceCompute<cpu>)
.add_arguments(LogspaceParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
