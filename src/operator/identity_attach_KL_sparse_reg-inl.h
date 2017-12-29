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
 * \file sparse_reg-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_IDENTITY_ATTACH_KL_SPARSE_REG_INL_H_
#define MXNET_OPERATOR_IDENTITY_ATTACH_KL_SPARSE_REG_INL_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace sparsereg {
enum IdentityAttachKLSparseRegOpInputs {kData};
enum IdentityAttachKLSparseRegOpOutputs {kOut};
enum IdentityAttachKLSparseRegOpAuxiliary {kMovingAvg};
enum IdentityAttachKLSparseRegBackResource {kTempSpace};
}  // namespace sparsereg

struct IdentityAttachKLSparseRegParam : public dmlc::Parameter<IdentityAttachKLSparseRegParam> {
  float penalty;
  float sparseness_target;
  float momentum;
  DMLC_DECLARE_PARAMETER(IdentityAttachKLSparseRegParam) {
    DMLC_DECLARE_FIELD(sparseness_target).set_default(0.1)
    .set_range(0, 1)
    .describe("The sparseness target");
    DMLC_DECLARE_FIELD(penalty).set_default(0.001)
    .describe("The tradeoff parameter for the sparseness penalty");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9)
    .set_range(0, 1)
    .describe("The momentum for running average");
  }
};  // struct IdentityAttachKLSparseRegParam

// This op regularizes the output of a sigmoid activation function.
// In forward, it simply copies the input.
// In backward, it attaches sparseness penalty to the gradient.
// The regularization is based on the KL divergence of mean activation and target.
// More details: P11 of https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
// Please make sure that it is only paired with sigmoid activation, otherwise NaN may occur.
template<typename xpu>
class IdentityAttachKLSparseRegOp : public Operator {
 public:
  explicit IdentityAttachKLSparseRegOp(IdentityAttachKLSparseRegParam param) {
    this->param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[sparsereg::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[sparsereg::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[sparsereg::kData], F<mshadow_op::identity>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad_in = in_grad[sparsereg::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> data_in = in_data[sparsereg::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad_out = out_grad[sparsereg::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 1> moving_avg = aux_args[sparsereg::kMovingAvg].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> avg = ctx.requested[sparsereg::kTempSpace].get_space<xpu>(
        mshadow::Shape1(moving_avg.shape_[0]), s);
    avg = sumall_except_dim<1>(data_in);
    avg /= data_in.shape_[0];
    moving_avg = param_.momentum * moving_avg + (1 - param_.momentum) * avg;
    Assign(grad_in, req[sparsereg::kData], grad_out + param_.penalty *
      (-param_.sparseness_target / broadcast<1>(moving_avg, data_in.shape_) +
      ((1 - param_.sparseness_target) / (1 - broadcast<1>(moving_avg, data_in.shape_)))));
  }

 private:
  IdentityAttachKLSparseRegParam param_;
};  // class IdentityAttachKLSparseRegOp

template<typename xpu>
Operator *CreateOp(IdentityAttachKLSparseRegParam param);

#if DMLC_USE_CXX11
class IdentityAttachKLSparseRegProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = in_shape->at(sparsereg::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new IdentityAttachKLSparseRegProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "IdentityAttachKLSparseReg";
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[sparsereg::kOut], in_data[sparsereg::kData]};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const override {
    return {{in_data[sparsereg::kData], out_data[sparsereg::kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const override {
    return { {out_grad[sparsereg::kOut], in_grad[sparsereg::kData]} };
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_avg"};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  IdentityAttachKLSparseRegParam param_;
};  // class IdentityAttachKLSparseRegProperty

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IDENTITY_ATTACH_KL_SPARSE_REG_INL_H_
