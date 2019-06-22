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
 * \file np_matrix_op.cc
 * \brief CPU Implementation of numpy matrix operations
 */

#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

bool NumpyVstackType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.num_args);
  CHECK_EQ(out_type->size(), 1);
  int dtype = -1;
  for (int i = 0; i < param.num_args; i++) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    }
  }
  if (dtype == -1) {
    dtype = out_type->at(0);
  }
  for (int i = 0; i < param.num_args; i++) {
    TYPE_ASSIGN_CHECK(*in_type, i, dtype);
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return dtype != -1;
}

bool NumpyVstackShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_attrs,
                            mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_args);
  std::vector<mxnet::TShape> in_attrs_tmp(param.num_args);
  TShape dshape;
  for (int i = 0; i < param.num_args; i++) {
    if ((*in_attrs)[i].ndim() == 0) {
      in_attrs_tmp[i] = TShape(2, 1);
    } else if ((*in_attrs)[i].ndim() == 1) {
      in_attrs_tmp[i] = TShape(2, 1);
      in_attrs_tmp[i][1] = (*in_attrs)[i][0];
    } else {
      in_attrs_tmp[i] = (*in_attrs)[i];
    }
    TShape tmp(in_attrs_tmp[i].ndim(), -1);
    shape_assign(&dshape, tmp);
  }
  TShape tmp((*out_attrs)[0].ndim(), -1);
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  if (dshape.ndim() == -1) {
    return false;
  }
  int cnt = 0, sum = 0, pos = -1;
  for (int i = 0; i < param.num_args; i++) {
    TShape tmp = in_attrs_tmp[i];
    if (!dim_size_is_known(tmp, 0)) {
      cnt++;
      pos = i;
    } else {
      sum += tmp[0];
    }
    tmp[0] = -1;
    shape_assign(&dshape, tmp);
  }
  tmp = out_attrs->at(0);
  if (!dim_size_is_known(tmp, 0)) {
    cnt++;
    pos = -1;
  } else {
    sum += tmp[0];
  }
  tmp[0] = -1;
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)\
  dshape[0] = 0;
  if (!shape_is_known(dshape)) {
    return false;
  }

  dshape[0] = sum;
  if (cnt == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  } else if (cnt == 1) {
    if (pos >= 0) {
      in_attrs_tmp[pos][0] = out_attrs->at(0)[0] - sum;
    } else {
      out_attrs->at(0)[0] = sum;
    }
  } else {
    return false;
  }

  for (int i = 0; i < param.num_args; i++) {
    if (in_attrs->at(i).ndim() == 1) {
      in_attrs->at(i)[0] = in_attrs_tmp[i][1];
    } else if (in_attrs->at(i).ndim() >= 2) {
      in_attrs->at(i) = in_attrs_tmp[i];
    }
  }

  return true;
}

DMLC_REGISTER_PARAMETER(NumpyVstackParam);

NNVM_REGISTER_OP(_npi_vstack)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyVstackParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    int num_args = dmlc::get<NumpyVstackParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (int i = 0; i < num_args; i++) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<mxnet::FInferShape>("FInferShape", NumpyVstackShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyVstackType)
.set_attr<FCompute>("FCompute<cpu>", NumpyVstackForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_vstack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to vstack")
.add_arguments(NumpyVstackParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_vstack)
.set_attr_parser(ParamParser<NumpyVstackParam>)
.set_num_inputs(1)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyVstackBackward<cpu>);

}  // namespace op
}  // namespace mxnet
