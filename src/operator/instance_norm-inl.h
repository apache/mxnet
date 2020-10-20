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
 * Copyright (c) 2015 by Contributors
 * \file instance_norm-inl.h
 * \brief Reproducing paper Instance Normalization: The Missing Ingredient for
 * Fast Stylization, D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_INSTANCE_NORM_INL_H_
#define MXNET_OPERATOR_INSTANCE_NORM_INL_H_
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

namespace instance_norm {
enum InstanceNormInputs { kData, kGamma, kBeta };
enum InstanceNormOutputs { kOut, kMean, kVar };
enum InstanceNormBackResource { kTempSpace };
}  // namespace instance_norm

struct InstanceNormParam : public dmlc::Parameter<InstanceNormParam> {
  float eps;
  DMLC_DECLARE_PARAMETER(InstanceNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f).describe(
        "An `epsilon` parameter to prevent division by 0.");
  }
};  // struct InstanceNormParam

template<typename xpu>
void InstanceNormForward(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(in_data.size(), 3U);
  CHECK_EQ(out_data.size(), 3U);

  CHECK_GE(in_data[instance_norm::kData].ndim(), 3)
      << "InstanceNorm only supports input tensors of rank >= 3.";

  const InstanceNormParam& param = nnvm::get<InstanceNormParam>(attrs.parsed);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape dshape = in_data[instance_norm::kData].shape_;
  CHECK(mxnet::shape_is_known(dshape)) << "Found unknown shape in InstanceNormForward, "
                                       << "received: " << dshape;
  if (dshape.Size() == 0) {
    return;  // noop for empty array
  }

  int n = dshape[0];
  int c = dshape[1];
  int rest_dim =
      static_cast<int>(in_data[instance_norm::kData].Size() / n / c);
  Shape<2> s2 = Shape2(n * c, rest_dim);
  const real_t scale = static_cast<real_t>(1) / static_cast<real_t>(rest_dim);
  // Get Inputs
  Tensor<xpu, 2> data =
      in_data[instance_norm::kData].get_with_shape<xpu, 2, real_t>(s2, s);
  Tensor<xpu, 1> gamma =
      in_data[instance_norm::kGamma].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> beta = in_data[instance_norm::kBeta].get<xpu, 1, real_t>(s);
  // Get Outputs
  Tensor<xpu, 2> out =
      out_data[instance_norm::kOut].get_with_shape<xpu, 2, real_t>(s2, s);
  Tensor<xpu, 1> var = out_data[instance_norm::kVar].FlatTo1D<xpu, real_t>(s);
  Tensor<xpu, 1> mean =
      out_data[instance_norm::kMean].FlatTo1D<xpu, real_t>(s);
  // Calculate mean + var
  mean = scale * sumall_except_dim<0>(data);
  var = scale * sumall_except_dim<0>(F<mshadow_op::square>(
                    data - broadcast<0>(mean, data.shape_)));
  Assign(
      out, req[instance_norm::kOut],
      broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), out.shape_) *
              (data - broadcast<0>(mean, data.shape_)) /
              F<mshadow_op::square_root>(
                  broadcast<0>(var + param.eps, data.shape_)) +
          broadcast<0>(reshape(repmat(beta, n), Shape1(n * c)), out.shape_));
}

template<typename xpu>
void InstanceNormBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 3U);

  CHECK_GE(inputs[3].ndim(), 3)
      << "InstanceNorm only supports input tensors of rank > 2.";

  const InstanceNormParam& param = nnvm::get<InstanceNormParam>(attrs.parsed);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape dshape = inputs[3].shape_;
  CHECK(mxnet::shape_is_known(dshape)) << "Found unknown shape in InstanceNormBackward, "
                                       << "received: " << dshape;
  if (dshape.Size() == 0) {
    return;  // noop for unknown shape or empty array
  }
  int n = inputs[3].size(0);
  int c = inputs[3].size(1);
  int rest_dim =
      static_cast<int>(inputs[3].Size() / n / c);
  Shape<2> s2 = Shape2(n * c, rest_dim);
  Shape<3> s3 = Shape3(n, c, rest_dim);
  const real_t scale = static_cast<real_t>(1) / static_cast<real_t>(rest_dim);
  // Get Inputs
  Tensor<xpu, 2> data =
      inputs[3].get_with_shape<xpu, 2, real_t>(s2, s);
  Tensor<xpu, 2> gdata =
      outputs[instance_norm::kData].get_with_shape<xpu, 2, real_t>(s2, s);
  Tensor<xpu, 1> gamma =
      inputs[4].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> ggamma =
      outputs[instance_norm::kGamma].get<xpu, 1, real_t>(s);
  Tensor<xpu, 1> gbeta = outputs[instance_norm::kBeta].get<xpu, 1, real_t>(s);
  // Get Outputs
  Tensor<xpu, 2> gout = inputs[0].get_with_shape<xpu, 2, real_t>(s2, s);
  Tensor<xpu, 1> var = inputs[2].FlatTo1D<xpu, real_t>(s);
  Tensor<xpu, 1> mean = inputs[1].FlatTo1D<xpu, real_t>(s);
  // Get temp space
  Tensor<xpu, 2> workspace =
      ctx.requested[instance_norm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(3, mean.shape_[0]), s);
  Tensor<xpu, 1> gmean = workspace[0];
  Tensor<xpu, 1> gvar = workspace[1];
  Tensor<xpu, 1> tmp = workspace[2];

  // calculate temps
  gvar = sumall_except_dim<0>(
      (gout *
       broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), data.shape_)) *
      (data - broadcast<0>(mean, data.shape_)) * -0.5f *
      F<mshadow_op::power>(broadcast<0>(var + param.eps, data.shape_),
                           -1.5f));
  gmean = sumall_except_dim<0>(
      gout *
      broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)), data.shape_));
  gmean *= -1.0f / F<mshadow_op::square_root>(var + param.eps);
  tmp = scale * sumall_except_dim<0>(
                    -2.0f * (data - broadcast<0>(mean, data.shape_)));
  tmp *= gvar;
  gmean += tmp;

  // Calculate grads
  Assign(gbeta, req[instance_norm::kBeta],
         sumall_except_dim<0>(swapaxis<1, 0>(reshape(gout, s3))));
  Assign(ggamma, req[instance_norm::kGamma],
         sumall_except_dim<0>(swapaxis<1, 0>(
             reshape(gout * (data - broadcast<0>(mean, data.shape_)) /
                         F<mshadow_op::square_root>(
                             broadcast<0>(var + param.eps, data.shape_)),
                     s3))));
  Assign(gdata, req[instance_norm::kData],
         (gout * broadcast<0>(reshape(repmat(gamma, n), Shape1(n * c)),
                              data.shape_)) *
                 broadcast<0>(
                     1.0f / F<mshadow_op::square_root>(var + param.eps),
                     data.shape_) +
             broadcast<0>(gvar, data.shape_) * scale * 2.0f *
                 (data - broadcast<0>(mean, data.shape_)) +
             broadcast<0>(gmean, data.shape_) * scale);
}

inline bool InstanceNormShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector *in_shape,
                              mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data]";
  const mxnet::TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;

  in_shape->at(1) = mxnet::TShape(Shape1(dshape[1]));
  in_shape->at(2) = mxnet::TShape(Shape1(dshape[1]));
  out_shape->clear();
  out_shape->push_back(dshape);
  out_shape->push_back(Shape2(dshape[0], dshape[1]));
  out_shape->push_back(Shape2(dshape[0], dshape[1]));
  return true;
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_INSTANCE_NORM_INL_H_
