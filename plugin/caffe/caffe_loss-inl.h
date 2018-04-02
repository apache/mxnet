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
 * Copyright (c) 2016 by Contributors
 * \file caffe_loss-inl.h
 * \brief Caffe Operator
 * \author Haoran Wang
*/
#ifndef PLUGIN_CAFFE_CAFFE_LOSS_INL_H_
#define PLUGIN_CAFFE_CAFFE_LOSS_INL_H_

#include <caffe/proto/caffe.pb.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

#include "../../src/operator/operator_common.h"
#include "caffe_common.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"

namespace mxnet {
namespace op {

struct CaffeLossParam : public dmlc::Parameter<CaffeLossParam> {
  ::caffe::LayerParameter prototxt;
  int num_data, num_out;
  float grad_scale;

  DMLC_DECLARE_PARAMETER(CaffeLossParam) {
    DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(num_data).set_range(0, 100).set_default(2)
    .describe("Operator input number");
    DMLC_DECLARE_FIELD(num_out).set_range(0, 100).set_default(1)
    .describe("Operator output number");
    DMLC_DECLARE_FIELD(grad_scale)
    .set_default(1.0f)
    .describe("Scale the gradient by a float factor (a.k.a weight of this loss).");
  }
};

/**
 * \brief this is the implementation of caffe operator in caffe.
 * \tparam xpu the device that the op will be executed on.
 */
template<typename xpu, typename Dtype>
class CaffeLoss : public Operator {
 public:
  explicit CaffeLoss(CaffeLossParam p):param_(p),
                                       setup_(false) {
    std::string type = param_.prototxt.type();
    caffeOp_ = caffe::LayerRegistry<Dtype>::CreateLayer(param_.prototxt);
    grad_scale_ = (Dtype)param_.grad_scale;

    caffe::InitCaffeBlobs<Dtype>(&bot_, param_.num_data);
    caffe::InitCaffeBlobs<Dtype>(&top_, param_.num_out);
    flags_.resize(param_.num_data);
  }

  ~CaffeLoss() {
    caffe::DelCaffeBlobs(&bot_, param_.num_data);
    caffe::DelCaffeBlobs(&top_, param_.num_out);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    // Set mode before forward
    caffe::CaffeMode::SetMode<xpu>();
    using ::caffe::Blob;
    using std::vector;
    using namespace mshadow;
    using namespace mshadow::expr;
    for (uint32_t i = 0; i < req.size(); ++i)
      CHECK_EQ(req[i], kWriteTo);

    CHECK_EQ(in_data.size(), param_.num_data);
    CHECK_EQ(out_data.size(), param_.num_out);

#if defined(__CUDACC__)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Data,
                                      bot_.begin(),
                                      in_data.begin(),
                                      param_.num_data);
    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Data,
                                      top_.begin(),
                                      out_data.begin(),
                                      param_.num_out);
    CaffeOpSetup();
    if (ctx.is_train)
      MXCAFFELAYER(caffeOp_, Dtype)->SetPhase(::caffe::TRAIN);
    else
      MXCAFFELAYER(caffeOp_, Dtype)->SetPhase(::caffe::TEST);
    caffeOp_->Forward(bot_, top_);

#if defined(__CUDACC__)
    // Sync cpu data to gpu data
    for (uint32_t i = 0; i < top_.size(); ++i)
      top_[i]->gpu_data();

    CHECK_EQ(cudaStreamSynchronize(NULL), cudaSuccess);
#endif  // __CUDACC__
  }

  // Set up caffe op with real data
  void CaffeOpSetup() {
    if (!setup_) {
      setup_ = true;
      caffeOp_->SetUp(bot_, top_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    // Set mode before backward
    caffe::CaffeMode::SetMode<xpu>();
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.num_out);
    for (int i = 0; i < param_.num_data; ++i)
      CHECK(req[i] != kAddTo) << "caffe doesn't accm diff on bottom data";
    CHECK(in_data.size() == param_.num_data);

#if defined(__CUDACC__)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Grad,
                                      bot_.begin(),
                                      in_grad.begin(),
                                      param_.num_data);
    // Pass grad scale to caffe blob
    MXCAFFEBLOB(top_[0], Dtype)->set_cpu_diff(&grad_scale_);

    // Set BP flag
    for (int i = 0; i < param_.num_data; ++i)
      flags_[i] = req[i] != kNullOp;

    caffeOp_->Backward(top_, flags_, bot_);

#if defined(__CUDACC__)
    // Sync cpu diff to gpu diff
    for (uint32_t i = 0; i < bot_.size(); ++i)
      bot_[i]->gpu_diff();

    CHECK_EQ(cudaStreamSynchronize(NULL), cudaSuccess);
#endif  // __CUDACC__
  }

 private:
  CaffeLossParam param_;
  ::caffe::Layer<Dtype> *caffeOp_;
  Dtype grad_scale_;
  std::vector< ::caffe::Blob<Dtype> *> bot_, top_;
  std::vector<bool> flags_;
  bool setup_;
};  // class CaffeLoss

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeLossParam param, int);

#if DMLC_USE_CXX11
class CaffeLossProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    CHECK_EQ(param_.num_out, 1);
    CHECK_EQ(param_.num_data, 2);

    // Fetch grad_scale from prototxt
    if ((param_.prototxt.loss_weight_size() > 0))
      param_.grad_scale = param_.prototxt.loss_weight(0);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  /*brief Set up caffeop to infer output shape*/
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    using ::caffe::Blob;
    using std::vector;
    if (caffeOp_ == NULL)
      caffeOp_ = caffe::LayerRegistry<float>::CreateLayer(param_.prototxt);

    CHECK_GE(in_shape->size(), param_.num_data);
    // Initialize empty bottom & top blobs for caffeOp setup
    vector<Blob<float> *> bot_blobs, top_blobs;

    for (int i = 0; i < param_.num_data; ++i) {
      TShape tshape = (*in_shape)[i];
      if (tshape.ndim() == 0) return false;
      auto blob_ptr = new Blob<float>();
      blob_ptr->Reshape(caffe::TShape2Vector(tshape));
      bot_blobs.push_back(blob_ptr);
    }

    for (int i = 0; i < param_.num_out; ++i)
      top_blobs.push_back(new Blob<float>());

    caffeOp_->SetUp(bot_blobs, top_blobs);
    CHECK_EQ(in_shape->size(), caffeOp_->blobs().size() + param_.num_data);
    // Initialize out shapes
    out_shape->clear();
    for (auto blob : top_blobs) {
      TShape tshape = caffe::Vector2TShape(blob->shape());
      out_shape->push_back(tshape);
    }

    for (auto blob_ptr : bot_blobs)
      delete blob_ptr;
    for (auto blob_ptr : top_blobs)
      delete blob_ptr;

    return true;
  }

  OperatorProperty* Copy() const override {
    auto copy_prop = new CaffeLossProp();
    copy_prop->param_ = this->param_;
    return copy_prop;
  }

  std::string TypeString() const override {
    return "CaffeLoss";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep;
    dep.insert(dep.end(), in_data.begin(), in_data.end());
    dep.insert(dep.end(), out_data.begin(), out_data.end());
    return dep;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;


 private:
  mutable CaffeLossParam param_;
  mutable ::caffe::Layer<float> *caffeOp_;
};  // class CaffeLossSymbol
#endif

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_LOSS_INL_H_
