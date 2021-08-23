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
 * Copyright (c) 2019 by Contributors
 * \file np_percentile_op.cc
 * \brief CPU Implementation of Numpy-compatible percentile
*/

#include <string>
#include "np_percentile_op-inl.h"

namespace mxnet {
namespace op {

template<typename QType, typename cpu>
bool CheckInvalidInput(mshadow::Stream<cpu> *s, const QType *data,
                       const size_t& data_size, char* is_valid_ptr) {
  for (size_t i = 0; i < data_size; i++) {
    if (data[i] < 0.0 || data[i] > 100) {
      return false;
    }
  }
  return true;
}

inline bool NumpyPercentileShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_GE(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  mxnet::TShape shape = NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims);

  mxnet::TShape qshape = param.q_scalar.has_value()? mxnet::TShape(0, 1) : in_attrs->at(1);
  CHECK_LE(qshape.ndim(), 1U);

  if (qshape.ndim() == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  } else {
    mxnet::TShape oshape(shape.ndim() + 1 , -1);
    oshape[0] = qshape[0];
    for (int i = 1 ; i < oshape.ndim(); i ++)
      oshape[i] = shape[i - 1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyPercentileType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_GE(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  if (in_attrs->at(0) == mshadow::kFloat64) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  }
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(NumpyPercentileParam);

NNVM_REGISTER_OP(_npi_percentile)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NumpyPercentileParam& param =
    nnvm::get<NumpyPercentileParam>(attrs.parsed);
  return param.q_scalar.has_value()? 1 : 2;
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyPercentileParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyPercentileShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyPercentileType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyPercentileParam& param =
      nnvm::get<NumpyPercentileParam>(attrs.parsed);
    return param.q_scalar.has_value() ?
           std::vector<std::string>{"a"} :
           std::vector<std::string>{"a", "q"};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyPercentileForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol", "Input data")
.add_argument("q", "NDArray-or-Symbol", "Input percentile")
.add_arguments(NumpyPercentileParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
