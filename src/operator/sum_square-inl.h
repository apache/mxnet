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
 * Copyright (c) 2018 by Contributors
 * \file sum_square-inl.h
 * \brief
 * \author Hang Zhang
 */

#ifndef MXNET_OPERATOR_SUM_SQUARE_INL_H_
#define MXNET_OPERATOR_SUM_SQUARE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
inline void SumSquareForward(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> data;
  if (inputs[0].ndim() == 2) {
    Shape<4> dshape = Shape4(inputs[0].shape_[0],
                             inputs[0].shape_[1], 1, 1);
    data = inputs[0].get_with_shape<xpu, 4, real_t>(dshape, s);
  } else {
    data = inputs[0].get<xpu, 4, real_t>(s);
  }
  Tensor<xpu, 1> sumx = outputs[0].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> sumx2 = outputs[1].get<xpu, 1, real_t>(s);
  Assign(sumx, req[0], sumall_except_dim<1>(data));
  Assign(sumx2, req[1], sumall_except_dim<1>(F<mshadow_op::power>(data, 2.f)));
}


template<typename xpu>
inline void SumSquareBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 1U);
  std::vector<TBlob> out_grad(inputs.begin(), inputs.begin() + 2);
  std::vector<TBlob> in_data(inputs.begin() + 2,
                             inputs.begin() + 3);
  std::vector<TBlob> in_grad(outputs.begin(), outputs.end());
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> data;
  Tensor<xpu, 4> grad_in;
  Tensor<xpu, 1> gradx;
  Tensor<xpu, 1> gradx2;
  if (in_data[0].ndim() == 2) {
    Shape<4> dshape = Shape4(out_grad[0].shape_[0],
                             out_grad[0].shape_[1], 1, 1);
    data = in_data[0].get_with_shape<xpu, 4, real_t>(dshape, s);
    grad_in = in_grad[0].get_with_shape<xpu, 4, real_t>(dshape, s);
  } else {
    data = in_data[0].get<xpu, 4, real_t>(s);
    grad_in = in_grad[0].get<xpu, 4, real_t>(s);
  }
  gradx = out_grad[0].get<xpu, 1, real_t>(s);
  gradx2 = out_grad[1].get<xpu, 1, real_t>(s);
  Assign(grad_in, req[0], broadcast<1>(gradx, data.shape_) + 
         data * broadcast<1>(gradx2 * 2.f, data.shape_));
}

static bool SumSquareInferShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_shape,
                                 std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  out_shape->clear();
  out_shape->push_back(TShape(Shape1(dshape[1])));
  out_shape->push_back(TShape(Shape1(dshape[1])));
  return true;
}

static bool SumSquareInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_type,
                                std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param = 0;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  std::vector<std::string> args{"data"};
  for (index_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  return true;
}

static inline bool SumSquareStorageType(const nnvm::NodeAttrs &attrs,
                                        const int dev_mask,
                                        DispatchMode *dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}

static inline bool backward_SumSquareStorageType(const nnvm::NodeAttrs &attrs,
                                                 const int dev_mask,
                                                 DispatchMode *dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 5);
  CHECK_EQ(out_attrs->size(), 1);
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUM_SQUARE_INL_H_
