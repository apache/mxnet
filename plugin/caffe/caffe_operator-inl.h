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

#include "caffe_base.h"
#include "caffe_operator_util.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"

namespace mxnet {
namespace op {

// Enumeration for inputs, outputs
namespace caffeEnum {
enum FetchType {DataOnly, GradOnly, DataWithGrad};
}  // namespace caffeEnum

struct CaffeOperatorParam : public dmlc::Parameter<CaffeOperatorParam> {
  caffe::LayerParameter prototxt;
  int in_num, w_num, out_num;
  caffe::Layer<float> *caffe_op;

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
template<typename xpu>
class CaffeOperator : public Operator {
 public:
  explicit CaffeOperator(CaffeOperatorParam p):param_(p),
                                               caffeOp_(p.caffe_op),
                                               init_w_(false),
                                               init_wd_(false) {
    InitCaffeBlobs(bot_, param_.in_num);
    InitCaffeBlobs(top_, param_.out_num);
    InitCaffeBlobs(wei_, param_.w_num);
  }

  ~CaffeOperator() {
    DelCaffeBlobs(bot_, param_.in_num);
    DelCaffeBlobs(top_, param_.out_num);
    DelCaffeBlobs(wei_, param_.w_num);
  }

  void CaffeForward(std::vector<caffe::Blob<float>*> bottom, std::vector<caffe::Blob<float>*> top);
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
    for (size_t i = 0; i < req.size(); ++i)
      CHECK_EQ(req[i], kWriteTo); size_t expected_in_num = param_.w_num + param_.in_num; CHECK_EQ(in_data.size(), expected_in_num);
    CHECK_EQ(out_data.size(), param_.out_num);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    TBlob2CaffeBlob<xpu>(caffememtype::Data, bot_.begin(), in_data.begin(), param_.in_num);
    TBlob2CaffeBlob<xpu>(caffememtype::Data, top_.begin(), out_data.begin(), param_.out_num);

    // Init caffe's weight pointer
    if (!init_w_) {
      init_w_ = true;
      TBlob2CaffeBlob<xpu>(caffememtype::Data,
                      wei_.begin(),
                      in_data.begin() + param_.in_num,
                      param_.w_num);
      caffeOp_->SetBlobs(wei_);
    }    

    // Set caffe's input & output blobs and forward
    this->CaffeForward(bot_, top_);

  }

  void CaffeBackward(std::vector<caffe::Blob<float>*> top, \
      std::vector<bool> bp_flags, std::vector<caffe::Blob<float>*> bottom);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    // Set mode before backward
    ::mxnet::CaffeMode::SetMode<xpu>();
    using std::vector;
    using ::caffe::Blob;
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.out_num);
    for (size_t i = 0; i < param_.in_num; ++i)
      CHECK(req[i] != kAddTo) << "caffe not support write as kAddTo";

    size_t expected_in_num = param_.w_num + param_.in_num;
    CHECK(in_data.size() == expected_in_num && in_grad.size() == expected_in_num);
    CHECK_EQ(req.size(), expected_in_num);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    TBlob2CaffeBlob<xpu>(caffememtype::Grad, bot_.begin(), in_grad.begin(), param_.in_num);
    TBlob2CaffeBlob<xpu>(caffememtype::Grad, top_.begin(), out_grad.begin(), param_.out_num);

    // Init caffe's gradient pointer
    if (!init_wd_) {
      init_wd_ = true;
      TBlob2CaffeBlob<xpu>(caffememtype::Grad,
                            wei_.begin(),
                            in_grad.begin() + param_.in_num,
                            param_.w_num);
    }

    // Set grad to zero
    for (size_t i = param_.in_num; i < expected_in_num; ++i)
        this->HandleOpReqType(s, req[i], &in_grad[i]);

    std::vector<bool> flags;
    // deal with OpReqType
    for (size_t i = 0; i < param_.in_num; ++i)
      flags.push_back(req[i] != kNullOp);

    // Set caffe's data and gradient blobs of input/output and do backward
    CaffeBackward(top_, flags, bot_);
  }

  void HandleOpReqType(mshadow::Stream<xpu>*s, OpReqType req, const TBlob* in_g) {
    switch (in_g->shape_.ndim()) {
      case 1: this->HandleOpReq<1>(s, req, in_g); break;
      case 2: this->HandleOpReq<2>(s, req, in_g); break;
      case 3: this->HandleOpReq<3>(s, req, in_g); break;
      case 4: this->HandleOpReq<4>(s, req, in_g); break;
      default:
        LOG(FATAL) << "unknown expected weight dim" << in_g->shape_.ndim();
        break;
    }
  }

  template<size_t dim>
  void HandleOpReq(mshadow::Stream<xpu>*s, OpReqType req, const TBlob* in_grad) {
    mshadow::Tensor<xpu, dim> w_g = in_grad->get<xpu, dim, real_t>(s);
    if ((req == kWriteInplace) || (req == kWriteTo))
      w_g = 0;
  }

 private:
  CaffeOperatorParam param_;
  caffe::Layer<float> *caffeOp_;
  std::vector<caffe::Blob<float> *> bot_, top_, wei_;
  bool init_w_, init_wd_;
};  // class CaffeOperator

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeOperatorParam param);

#if DMLC_USE_CXX11
class CaffeOperatorProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> res;
    for (size_t i = 0; i < param_.in_num; ++i)
      res.push_back(std::string("data_") + static_cast<char>('0' + i));
    /*
     * \brief the assumption is: first blob is weight, second is bias.
     * \brief However, some types of caffe-layers might not follow this
     * \brief Customization is then required.
     */
    for (int i = 0; i < GetBlobNum(); ++i) {
      if (i == 0)
        res.push_back(std::to_string(i) + "_weight");
      else
        res.push_back(std::to_string(i) + "_bias");
    }
    return res;
  }

  int GetBlobNum() const {
    std::string type = param_.prototxt.type();
    entry_ = CaffeOpInitRegistry::Get()->Find(param_.prototxt.type());
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
    entry_ = CaffeOpInitRegistry::Get()->Find(param_.prototxt.type());
    param_.caffe_op = entry_->gen_f_(this->param_.prototxt);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  /*
   * \brief Set up caffe_op to infer weights & output shape
   * \brief Initialize param_'s in & out dims
   */
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    using caffe::Blob;
    using std::vector;
    CHECK_GE(in_shape->size(), param_.in_num);
    // Initialize bottom & top blobs for caffe_op setup
    vector<Blob<float> *> bot_blobs, top_blobs;
    // Set OperatorParam input dims & caffe op input blobs
    for (size_t i = 0; i < param_.in_num; ++i) {
      TShape tshape = (*in_shape)[i];
      if (tshape.ndim() == 0) return false;
      auto blob_ptr = new Blob<float>();
      blob_ptr->Reshape(TShape2Vector(tshape));
      bot_blobs.push_back(blob_ptr);
    }
    // Set caffe op output blobs
    for (size_t i = 0; i < param_.out_num; ++i)
      top_blobs.push_back(new Blob<float>());

    param_.caffe_op->SetUp(bot_blobs, top_blobs);
    CHECK_EQ(in_shape->size(), param_.caffe_op->blobs().size() + param_.in_num);
    // Set weight shape
    param_.w_num = param_.caffe_op->blobs().size();
    for (size_t i = 0; i < param_.w_num ; ++i) {
      TShape tshape = Vector2TShape(param_.caffe_op->blobs()[i]->shape());
      SHAPE_ASSIGN_CHECK(*in_shape, i + param_.in_num, tshape);
    }
    // Initialize out dims & out shapes
    out_shape->clear();
    for (auto blob : top_blobs) {
      TShape tshape = Vector2TShape(blob->shape());
      out_shape->push_back(tshape);
    }
    // Free caffe in & out blobs
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

  Operator* CreateOperator(Context ctx) const override;

 private:
  mutable CaffeOperatorParam param_;
  mutable CaffeOpInitEntry* entry_;
};  // class CaffeOperatorSymbol
#endif

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_INL_H_
