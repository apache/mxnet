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
 * \file all_finite-inl.h
 * \brief operator for checking if a group of array is all finite
 * \author Clement Fuji Tsang
 */

#ifndef MXNET_OPERATOR_CONTRIB_ALL_FINITE_INL_H_
#define MXNET_OPERATOR_CONTRIB_ALL_FINITE_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct AllFiniteParam: public dmlc::Parameter<AllFiniteParam> {
  bool init_output;
  DMLC_DECLARE_PARAMETER(AllFiniteParam) {
    DMLC_DECLARE_FIELD(init_output)
    .set_default(true)
    .describe("Initialize output to 1.");
  }
};

struct MultiAllFiniteParam : public dmlc::Parameter<MultiAllFiniteParam> {
  int num_arrays;
  bool init_output;
  DMLC_DECLARE_PARAMETER(MultiAllFiniteParam) {
    DMLC_DECLARE_FIELD(num_arrays)
    .set_default(1)
    .describe("Number of arrays.");
    DMLC_DECLARE_FIELD(init_output)
    .set_default(true)
    .describe("Initialize output to 1.");
  }
};

template<typename DType>
struct MultiAllFiniteKernelParam {
  static const int N = 200;
  int count;
  size_t max_size;
  size_t sizes[N];
  DType *arrays[N];
};

template<typename xpu, typename DType>
MultiAllFiniteKernelParam<DType> FillMultiAllFiniteParam(const MultiAllFiniteParam& op_param,
                                                         const OpContext &ctx,
                                                         const std::vector<TBlob> &inputs) {
  MultiAllFiniteKernelParam<DType> param;
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  param.count = op_param.num_arrays;
  param.max_size = 0;
  for (int i = 0; i < param.count; ++i) {
    param.sizes[i] = inputs[i].shape_.Size();
    if (param.max_size < param.sizes[i]) {
      param.max_size = param.sizes[i];
    }
    param.arrays[i] = inputs[i].FlatTo2D<xpu, DType>(s).dptr_;
  }
  return param;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ALL_FINITE_INL_H_
