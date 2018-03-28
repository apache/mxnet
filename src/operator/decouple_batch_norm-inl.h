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
 * \file sync_batch_norm-inl.h
 * \brief
 * \author Hang Zhang
 * Adapted from BatchNormV1
 */
#ifndef MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
//#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

using namespace mshadow;
template<typename xpu>
inline void BNForward(Tensor<xpu, 4> data,
                      Tensor<xpu, 1> gamma,
                      Tensor<xpu, 1> beta,
                      Tensor<xpu, 1> mean,
                      Tensor<xpu, 1> std,
                      Tensor<xpu, 4> &out,
                      const std::vector<OpReqType> &req) {
  using namespace mshadow::expr;
  Assign(out, req[0], broadcast<1>(gamma / std, data.shape_) * data +
           broadcast<1>(beta - (gamma * mean) / std, data.shape_));
}

template<typename xpu>
inline void BNBackward(Tensor<xpu, 4> grad,
                       Tensor<xpu, 4> data,
                       Tensor<xpu, 1> mean,
                       Tensor<xpu, 1> std,
                       Tensor<xpu, 1> gamma,
                       Tensor<xpu, 4> &grad_in,
                       Tensor<xpu, 1> &gradGamma,
                       Tensor<xpu, 1> &gradBeta,
                       Tensor<xpu, 1> &gradMean,
                       Tensor<xpu, 1> &gradStd,
                       const std::vector<OpReqType> &req) {
  using namespace mshadow::expr;
  // the grad may not be zero originally? check this
  Assign(gradMean, req[3], -1.f *
         sumall_except_dim<1>(grad* broadcast<1>(gamma / std, data.shape_)));
  Assign(gradStd, req[4],
         sumall_except_dim<1>((grad * broadcast<1>(gamma, data.shape_)) *
                              (data - broadcast<1>(mean, data.shape_)) *
                              -1.f *
                              F<mshadow_op::power>(broadcast<1>(std, data.shape_),
                                                   -2.f)));
  Assign(gradGamma, req[1],
         sumall_except_dim<1>(
             grad * (data - broadcast<1>(mean, data.shape_)) /
             broadcast<1>(std, data.shape_)));
  Assign(gradBeta, req[2], sumall_except_dim<1>(grad));
  Assign(grad_in, req[0],
         (grad * broadcast<1>(gamma / std, data.shape_)))
}

template<typename xpu>
inline void DecoupleBNForward(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (!ctx.is_train) {
    CHECK_EQ(req[0], kWriteTo);
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> data;
  Tensor<xpu, 4> out;
  if (inputs[0].ndim() == 2) {
    Shape<4> dshape = Shape4(inputs[0].shape_[0],
                             inputs[0].shape_[1], 1, 1);
    data = inputs[0].get_with_shape<xpu, 4, real_t>(dshape, s);
    out = outputs[0].get_with_shape<xpu, 4, real_t>(dshape, s);
  } else {
    data = inputs[0].get<xpu, 4, real_t>(s);
    out = outputs[0].get<xpu, 4, real_t>(s);
  }
  Tensor<xpu, 1> gamma = inputs[1].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> beta = inputs[2].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> mean = inputs[3].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> std = inputs[4].get<xpu, 1, real_t>(s);

  BNForward<xpu>(data, gamma, beta, mean,std, out, req);
}

template<typename xpu>
inline void DecoupleBNBackward(const nnvm::NodeAttrs& attrs, 
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 7U);
  CHECK_EQ(outputs.size(), 5U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<TBlob> out_grad(inputs.begin(), inputs.begin() + 1);
  std::vector<TBlob> in_data(inputs.begin() + 1,
                             inputs.begin() + 6);
  //std::vector<TBlob> out_data(inputs.begin() + 6, inputs.end());
  std::vector<TBlob> in_grad(outputs.begin(), outputs.begin() + 5);
  Tensor<xpu, 4> data, grad, grad_in;
  if (in_data[0].ndim() == 2) {
    Shape<4> dshape = Shape4(out_grad[0].shape_[0],
                             out_grad[0].shape_[1], 1, 1);
    data = in_data[0].get_with_shape<xpu, 4, real_t>(dshape, s);
    grad = out_grad[0].get_with_shape<xpu, 4, real_t>(dshape, s);
    grad_in = in_grad[0].get_with_shape<xpu, 4, real_t>(dshape, s);
  } else {
    data = in_data[0].get<xpu, 4, real_t>(s);
    grad = out_grad[0].get<xpu, 4, real_t>(s);
    grad_in = in_grad[0].get<xpu, 4, real_t>(s);
  }

  Tensor<xpu, 1> gamma = in_data[1].get<xpu, 1, real_t>(s);
  //Tensor<xpu, 1> beta = in_data[2].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> mean = in_data[3].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> std = in_data[4].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> gradGamma = in_grad[1].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> gradBeta = in_grad[2].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> gradMean = in_grad[3].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> gradStd = in_grad[4].get<xpu, 1, real_t>(s);
  BNBackward<xpu>(grad, data, mean, std, gamma, 
                  grad_in, gradGamma, gradBeta, gradMean, gradStd, req);
}


static bool DecoupleBNInferShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_shape,
                                 std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, mean, std]";
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  in_shape->at(1) = TShape(Shape1(dshape[1]));
  in_shape->at(2) = TShape(Shape1(dshape[1]));
  in_shape->at(3) = TShape(Shape1(dshape[1]));
  in_shape->at(4) = TShape(Shape1(dshape[1]));
  out_shape->clear();
  out_shape->push_back(dshape);

  return true;
}

static bool DecoupleBNInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_type,
                                std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 5U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param = 0;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  std::vector<std::string> args{"data", "gamma", "beta", "mean", "var"};
  for (index_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

static inline bool DecoupleBNStorageType(const nnvm::NodeAttrs &attrs,
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

static inline bool backward_DecoupleBNStorageType(const nnvm::NodeAttrs &attrs,
                                                 const int dev_mask,
                                                 DispatchMode *dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 7);
  CHECK_EQ(out_attrs->size(), 5);
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

#endif  // MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_
