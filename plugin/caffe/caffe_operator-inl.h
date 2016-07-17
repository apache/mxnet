/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator-inl.h
 * \brief Caffe Operator
 * \author Haoran Wang 
*/
#ifndef PLUGIN_CAFFE_CAFFE_OPERATOR_INL_H_
#define PLUGIN_CAFFE_CAFFE_OPERATOR_INL_H_

#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/blob.hpp>

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <exception>

#include "../../src/operator/operator_common.h"

#include "caffe_common.h"
#include "caffe_operator_util.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"

namespace mxnet {
namespace op {

struct CaffeOperatorParam : public dmlc::Parameter<CaffeOperatorParam> {
  caffe::LayerParameter prototxt;
  int in_num, w_num, out_num;

  DMLC_DECLARE_PARAMETER(CaffeOperatorParam) { DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(in_num).set_range(0, 100).set_default(1)
    .describe("Operator input number");
    DMLC_DECLARE_FIELD(out_num).set_range(0, 100).set_default(1)
    .describe("Operator output number");
  }
};


/**
 * \brief this is the implementation of caffe operator in caffe.
 * \tparam xpu the device that the op will be executed on.
 */
template<typename xpu, typename Dtype>
class CaffeOperator : public Operator {
 public:
  explicit CaffeOperator(CaffeOperatorParam p):param_(p),
                                               setup_(false),
                                               init_w_(false),
                                               init_wd_(false) {
    std::string type = param_.prototxt.type();
    CaffeOpInitEntry<Dtype>* e = CaffeOpInitRegistry<Dtype>::Get()->Find(type);
    caffeOp_ = e->gen_f_(param_.prototxt);

    InitCaffeBlobs<Dtype>(bot_, param_.in_num);
    InitCaffeBlobs<Dtype>(top_, param_.out_num);
    InitCaffeBlobs<Dtype>(wei_, param_.w_num);
    flags_.resize(param_.in_num);
  }

  ~CaffeOperator() {
    DelCaffeBlobs(bot_, param_.in_num);
    DelCaffeBlobs(top_, param_.out_num);
    DelCaffeBlobs(wei_, param_.w_num);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    // Set mode before forward
    ::mxnet::CaffeMode::SetMode<xpu>();
    using ::caffe::Blob;
    using std::vector;
    using namespace mshadow;
    using namespace mshadow::expr;
    for (index_t i = 0; i < req.size(); ++i)
      CHECK_EQ(req[i], kWriteTo);
    index_t expected_in_num = param_.w_num + param_.in_num;
    CHECK_EQ(in_data.size(), expected_in_num);
    CHECK_EQ(out_data.size(), param_.out_num);

    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Data, bot_.begin(), in_data.begin(), param_.in_num);
    TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Data, top_.begin(), out_data.begin(), param_.out_num);

    CaffeOpSetup();

    // Init caffe's weight pointer
    if (!init_w_) {
      init_w_ = true;
      TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Data,
                      wei_.begin(),
                      in_data.begin() + param_.in_num,
                      param_.w_num);
      caffeOp_->SetBlobs(wei_);
    }    

    caffeOp_->Forward(bot_, top_);
  }

  // Set up caffe op with real data
  void CaffeOpSetup() {
    if (!setup_ ) {
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
    ::mxnet::CaffeMode::SetMode<xpu>();
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.out_num);
    for (index_t i = 0; i < param_.in_num; ++i)
      CHECK(req[i] != kAddTo) << "caffe doesn't accm diff on bottom data";

    index_t expected_in_num = param_.w_num + param_.in_num;
    CHECK(in_data.size() == expected_in_num && in_grad.size() == expected_in_num);
    CHECK_EQ(req.size(), expected_in_num);

    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Grad, bot_.begin(), in_grad.begin(), param_.in_num);
    TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Grad, top_.begin(), out_grad.begin(), param_.out_num);

    // Init caffe's gradient pointer
    if (!init_wd_) {
      init_wd_ = true;
      TBlob2CaffeBlob<xpu, Dtype>(caffememtype::Grad,
                            wei_.begin(),
                            in_grad.begin() + param_.in_num,
                            param_.w_num);
    }

    // Handle OpReqType of weights
    for (index_t i = param_.in_num; i < expected_in_num; ++i)
      HandleOpReq(s, req[i], &in_grad[i]);

    // Set BP flag
    for (index_t i = 0; i < param_.in_num; ++i)
      flags_[i] = req[i] != kNullOp;

    caffeOp_->Backward(top_, flags_, bot_);
  }

  void HandleOpReq(mshadow::Stream<xpu>*s, OpReqType req, const TBlob* in_g) {
    if ((req == kWriteInplace) || (req == kWriteTo))
      switch (in_g->shape_.ndim()) {
        case 1: CleanTBlob<1>(s, req, in_g); break;
        case 2: CleanTBlob<2>(s, req, in_g); break;
        case 3: CleanTBlob<3>(s, req, in_g); break;
        case 4: CleanTBlob<4>(s, req, in_g); break;
        default:
          LOG(FATAL) << "unknown expected weight dim" << in_g->shape_.ndim();
          break;
      }
  }

  template<index_t dim>
  void CleanTBlob(mshadow::Stream<xpu>*s, OpReqType req, const TBlob* in_grad) {
    mshadow::Tensor<xpu, dim> t = in_grad->get<xpu, dim, real_t>(s);
    t = 0;
  }

 private:
  CaffeOperatorParam param_;
  caffe::Layer<Dtype> *caffeOp_;
  std::vector<caffe::Blob<Dtype> *> bot_, top_, wei_;
  std::vector<bool> flags_;
  bool init_w_, init_wd_, setup_;
};  // class CaffeOperator

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeOperatorParam param, int);

#if DMLC_USE_CXX11
class CaffeOperatorProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> res;
    for (index_t i = 0; i < param_.in_num; ++i)
      res.push_back(std::string("data_") + static_cast<char>('0' + i));

    for (index_t i = 0; i < GetBlobNum(); ++i) {
      if (i == 0)
        res.push_back(std::to_string(i) + "_weight");
      else
        res.push_back(std::to_string(i) + "_bias");
    }
    return res;
  }

  int GetBlobNum() const {
    std::string type = param_.prototxt.type();
    entry_ = CaffeOpInitRegistry<float>::Get()->Find(param_.prototxt.type());
    /* get weight value in registery */
    int blob_num = entry_->b_num_;
    /* otherwise, calculate blob num in runtime */
    if (!type.compare("InnerProduct"))
      blob_num = (param_.prototxt.inner_product_param().bias_term())?2:1;
    else if (!type.compare("Convolution")||
             !type.compare("CuDNNConvolution")||
             !type.compare("Deconvolution"))
      blob_num = (param_.prototxt.convolution_param().bias_term())?2:1;
    else if (!type.compare("Scale"))
      blob_num = (param_.prototxt.scale_param().bias_term())?2:1;
    else if (!type.compare("Embed"))
      blob_num = (param_.prototxt.embed_param().bias_term())?2:1;

    CHECK(blob_num>=0);
    return blob_num;
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
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    if (caffeOp_ == NULL) {
      entry_ = CaffeOpInitRegistry<float>::Get()->Find(param_.prototxt.type());
      caffeOp_ = entry_->gen_f_(this->param_.prototxt);
    }
    using namespace mshadow;
    using caffe::Blob;
    using std::vector;
    CHECK_GE(in_shape->size(), param_.in_num);
    // Initialize emtryp bottom & top blobs for caffeop
    vector<Blob<float> *> bot_blobs, top_blobs;

    for (index_t i = 0; i < param_.in_num; ++i) {
      TShape tshape = (*in_shape)[i];
      if (tshape.ndim() == 0) return false;
      auto blob_ptr = new Blob<float>();
      blob_ptr->Reshape(TShape2Vector(tshape));
      bot_blobs.push_back(blob_ptr);
    }

    for (index_t i = 0; i < param_.out_num; ++i)
      top_blobs.push_back(new Blob<float>());

    caffeOp_->SetUp(bot_blobs, top_blobs);
    CHECK_EQ(in_shape->size(), caffeOp_->blobs().size() + param_.in_num);
    // Set weight shape
    param_.w_num = caffeOp_->blobs().size();
    for (index_t i = 0; i < param_.w_num ; ++i) {
      TShape tshape = Vector2TShape(caffeOp_->blobs()[i]->shape());
      SHAPE_ASSIGN_CHECK(*in_shape, i + param_.in_num, tshape);
    }
    // Initialize out shapes
    out_shape->clear();
    for (auto blob : top_blobs) {
      TShape tshape = Vector2TShape(blob->shape());
      out_shape->push_back(tshape);
    }

    for (auto blob_ptr : bot_blobs)
      delete blob_ptr;
    for (auto blob_ptr : top_blobs)
      delete blob_ptr;
    return true;
  }

  OperatorProperty* Copy() const override {
    auto copy_prop = new CaffeOperatorProp();
    copy_prop->param_ = this->param_;
    return copy_prop;
  }

  std::string TypeString() const override {
    return "CaffeOperator";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  mutable CaffeOperatorParam param_;
  mutable CaffeOpInitEntry<float>* entry_;
  mutable caffe::Layer<float>* caffeOp_;
};  // class CaffeOperatorSymbol
#endif

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_INL_H_
