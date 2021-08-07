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
 * \file caffe_op-inl.h
 * \brief Caffe Operator
 * \author Haoran Wang
*/
#ifndef PLUGIN_CAFFE_CAFFE_OP_INL_H_
#define PLUGIN_CAFFE_CAFFE_OP_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <caffe/proto/caffe.pb.h>

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

struct CaffeOpParam : public dmlc::Parameter<CaffeOpParam> {
  ::caffe::LayerParameter prototxt;
  int num_data, num_weight, num_out;

  DMLC_DECLARE_PARAMETER(CaffeOpParam) { DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(num_data).set_default(1)
    .describe("Operator input number");
    DMLC_DECLARE_FIELD(num_weight).set_default(0)
    .describe("Weight number");
    DMLC_DECLARE_FIELD(num_out).set_default(1)
    .describe("Operator output number");
  }
};


/**
 * \brief this is the implementation of caffe operator in caffe.
 * \tparam xpu the device that the op will be executed on.
 */
template<typename xpu, typename Dtype>
class CaffeOp : public Operator {
 public:
  explicit CaffeOp(CaffeOpParam p):param_(p),
                                   init_w_(false),
                                   init_wd_(false),
                                   setup_(false) {
    std::string type = param_.prototxt.type();
    caffeOp_ = caffe::LayerRegistry<Dtype>::CreateLayer(param_.prototxt);

    caffe::InitCaffeBlobs<Dtype>(&bot_, param_.num_data);
    caffe::InitCaffeBlobs<Dtype>(&top_, param_.num_out);
    caffe::InitCaffeBlobs<Dtype>(&wei_, param_.num_weight);
    flags_.resize(param_.num_data);
  }

  ~CaffeOp() {
    caffe::DelCaffeBlobs(&bot_, param_.num_data);
    caffe::DelCaffeBlobs(&top_, param_.num_out);
    caffe::DelCaffeBlobs(&wei_, param_.num_weight);
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
    int expected_num_data = param_.num_weight + param_.num_data;
    CHECK_EQ(in_data.size(), expected_num_data);
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
    // Init caffe's weight pointer
    if (!init_w_) {
      init_w_ = true;
      caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Data,
                                         wei_.begin(),
                                         in_data.begin() + param_.num_data,
                                         param_.num_weight);
      caffe::SetOpBlobs(caffeOp_, wei_);
    }
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

    int expected_num_data = param_.num_weight + param_.num_data;
    CHECK(in_data.size() == expected_num_data && in_grad.size() == expected_num_data);
    CHECK_EQ(req.size(), expected_num_data);

    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Grad,
                                       bot_.begin(),
                                       in_grad.begin(),
                                       param_.num_data);
    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Grad,
                                       top_.begin(),
                                       out_grad.begin(),
                                       param_.num_out);

    // Init caffe's gradient pointer
    if (!init_wd_) {
      init_wd_ = true;
      caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Grad,
                                         wei_.begin(),
                                         in_grad.begin() + param_.num_data,
                                         param_.num_weight);
    }

    // Handle OpReqType of weights
    for (int i = param_.num_data; i < expected_num_data; ++i)
      HandleOpReq(s, req[i], in_grad[i]);

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

  void HandleOpReq(mshadow::Stream<xpu>*s, OpReqType req, const TBlob& in_g) {
    if ((req == kWriteInplace) || (req == kWriteTo)) {
      mshadow::Tensor<xpu, 2, Dtype> grad = in_g.FlatTo2D<xpu, Dtype>(s);
      grad = 0;
    }
  }

 private:
  CaffeOpParam param_;
  ::caffe::Layer<Dtype> *caffeOp_;
  std::vector< ::caffe::Blob<Dtype> *> bot_, top_, wei_;
  std::vector<bool> flags_;
  bool init_w_, init_wd_, setup_;
};  // class CaffeOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeOpParam param, int);

#if DMLC_USE_CXX11
class CaffeOpProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> res;
    for (int i = 0; i < param_.num_data; ++i)
      res.push_back(std::string("data_") + std::to_string(i));

    for (int i = 0; i < param_.num_weight; ++i) {
      if (i == 0)
        res.push_back(std::to_string(i) + "_weight");
      else
        res.push_back(std::to_string(i) + "_bias");
    }
    return res;
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.num_out > 1) {
      std::vector<std::string> ret;
      for (int i = 0; i < param_.num_out; ++i)
        ret.push_back("output" + std::to_string(i));
      return ret;
    } else {
      return {"output"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  /*
   * \brief Set up caffeOp_ to infer weights & output shape
   * \brief Initialize param_'s in & out dims
   */
  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    if (caffeOp_ == NULL)
      caffeOp_ = caffe::LayerRegistry<float>::CreateLayer(param_.prototxt);
    using namespace mshadow;
    using ::caffe::Blob;
    using std::vector;
    CHECK_GE(in_shape->size(), param_.num_data);
    // Initialize emtryp bottom & top blobs for caffeop
    vector<Blob<float> *> bot_blobs, top_blobs;

    for (int i = 0; i < param_.num_data; ++i) {
      mxnet::TShape tshape = (*in_shape)[i];
      if (tshape.ndim() == 0) return false;
      auto blob_ptr = new Blob<float>();
      blob_ptr->Reshape(caffe::TShape2Vector(tshape));
      bot_blobs.push_back(blob_ptr);
    }

    for (int i = 0; i < param_.num_out; ++i)
      top_blobs.push_back(new Blob<float>());

    caffeOp_->SetUp(bot_blobs, top_blobs);
    CHECK_EQ(in_shape->size(), caffeOp_->blobs().size() + param_.num_data);
    // Set weight shape
    CHECK_EQ(param_.num_weight, caffeOp_->blobs().size());
    for (int i = 0; i < param_.num_weight ; ++i) {
      mxnet::TShape tshape = caffe::Vector2mxnet::TShape(caffeOp_->blobs()[i]->shape());
      SHAPE_ASSIGN_CHECK(*in_shape, i + param_.num_data, tshape);
    }
    // Initialize out shapes
    out_shape->clear();
    for (auto blob : top_blobs) {
      mxnet::TShape tshape = caffe::Vector2mxnet::TShape(blob->shape());
      out_shape->push_back(tshape);
    }

    for (auto blob_ptr : bot_blobs)
      delete blob_ptr;
    for (auto blob_ptr : top_blobs)
      delete blob_ptr;
    return true;
  }

  OperatorProperty* Copy() const override {
    auto copy_prop = new CaffeOpProp();
    copy_prop->param_ = this->param_;
    return copy_prop;
  }

  std::string TypeString() const override {
    return "CaffeOp";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  mutable CaffeOpParam param_;
  mutable ::caffe::Layer<float> *caffeOp_;
};  // class CaffeOpSymbol
#endif

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OP_INL_H_
