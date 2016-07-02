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
  std::string op_type_string;
  std::vector<int> in_dims, w_dims, out_dims;
  caffe::Layer<float> *caffe_op;
  int in_num, out_num;

  DMLC_DECLARE_PARAMETER(CaffeOperatorParam) {
    DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(op_type_string)
    .describe("Operator type name");
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
                                               initWeight_(false),
                                               initWeightDelta_(false) {
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
      CHECK_EQ(req[i], kWriteTo);
    size_t expected_in_num = param_.w_dims.size() + param_.in_dims.size();
    CHECK_EQ(in_data.size(), expected_in_num);
    CHECK_EQ(out_data.size(), param_.out_dims.size());
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    // TODO(Haoran): when need cublas handle in stream?
    if (!param_.op_type_string.compare("fullyConnected"))
      CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    vector<Blob<float> *> bot_blobs, top_blobs;
    vector<TBlob> empty_tblobs;
    this->BuildOrModifyBlobs(s, caffeEnum::DataOnly, param_.in_dims,
                             false, bot_blobs, 0, in_data, empty_tblobs);
    this->BuildOrModifyBlobs(s, caffeEnum::DataOnly, param_.out_dims,
                             false, top_blobs, 0, out_data, empty_tblobs);

    if (!initWeight_) {
      // Init caffe's weight pointer
      initWeight_ = true;
      weightDataList_ = new std::vector<void*>();
      for (int i = param_.in_dims.size(); i < expected_in_num; ++i)
        weightDataList_->push_back(in_data[i].dptr_);
      vector<Blob<float>*> w_blobs;
      this->BuildOrModifyBlobs(s, caffeEnum::DataOnly, param_.w_dims,
                               false, w_blobs, param_.in_dims.size(), in_data, empty_tblobs);
      caffeOp_->SetBlobs(w_blobs);
    } else {
      // TODO(Haoran): Delete this chekcer
      // pointer of weights should align with the weights passed in
      for (int i = param_.in_dims.size(); i < expected_in_num; ++i)
        CHECK_EQ(weightDataList_->at(i-param_.in_dims.size()), in_data[i].dptr_);
    }

    // Set caffe's input & output blobs and forward
    this->CaffeForward(bot_blobs, top_blobs);

    // Free caffe in & out blobs
    for (size_t i = 0; i < bot_blobs.size(); ++i)
      delete bot_blobs[i];
    for (size_t i = 0; i < top_blobs.size(); ++i)
      delete top_blobs[i];
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
    CHECK_EQ(out_grad.size(), param_.out_dims.size());
    for (size_t i = 0; i < param_.in_dims.size(); ++i)
      CHECK(req[i] != kAddTo) << "caffe not support write as kAddTo";

    size_t expected_in_num = param_.w_dims.size() + param_.in_dims.size();
    CHECK(in_data.size() == expected_in_num && in_grad.size() == expected_in_num);
    CHECK_EQ(req.size(), expected_in_num);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    if (!param_.op_type_string.compare("fullyConnected"))
      CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    vector<Blob<float>*> top_blobs, bot_blobs;
    vector<TBlob> empty_tblobs;
    this->BuildOrModifyBlobs(s, caffeEnum::DataWithGrad, param_.in_dims,
                             false, bot_blobs, 0, in_data, in_grad);
    this->BuildOrModifyBlobs(s, caffeEnum::DataWithGrad, param_.out_dims,
                             false, top_blobs, 0, out_data, out_grad);

    if (!initWeightDelta_) {
      // Init caffe's gradient pointer
      initWeightDelta_ = true;
      weightDeltaList_ = new vector<void*>();
      for (int i = param_.in_dims.size(); i < expected_in_num; ++i)
        weightDeltaList_->push_back(in_grad[i].dptr_);

      vector<Blob<float>*> w_blobs = caffeOp_->GetBlobs();
      this->BuildOrModifyBlobs(s, caffeEnum::GradOnly, param_.w_dims,
                               true, w_blobs, param_.in_dims.size(), in_grad, empty_tblobs);
    } else {
      // TODO(Haoran): Delete this chekcer
      // pointer of gradient should align with the gradient passed in
      for (int i = param_.in_dims.size(); i < expected_in_num; ++i) {
        CHECK_EQ(weightDeltaList_->at(i-param_.in_dims.size()), in_grad[i].dptr_);
        CHECK_EQ(weightDataList_->at(i-param_.in_dims.size()), in_data[i].dptr_);
      }
    }

    // Set grad to zero
    for (size_t i = param_.in_dims.size(); i < expected_in_num; ++i) {
        int dim = param_.w_dims[i - param_.in_dims.size()];
        this->HandleOpReqType(s, req[i], dim, &in_grad[i]);
    }

    std::vector<bool> flags;
    // deal with OpReqType
    for (size_t i = 0; i < param_.in_dims.size(); ++i)
      flags.push_back(req[i] != kNullOp);

    // Set caffe's data and gradient blobs of input/output and do backward
    CaffeBackward(top_blobs, flags, bot_blobs);

    // Free caffe in & out blobs
    for (size_t i = 0; i < bot_blobs.size(); ++i)
      delete bot_blobs[i];
    for (size_t i = 0; i < top_blobs.size(); ++i)
      delete top_blobs[i];
  }

  template<size_t dim>
  void ConvertTBlob2Blob(mshadow::Stream<xpu>* s,
                    int fetch_type,
                    ::caffe::Blob<float>* blob_ptr,
                    const TBlob *tblob_0,
                    const TBlob *tblob_1 = NULL) {
    using mshadow::Tensor;
    switch (fetch_type) {
      case caffeEnum::DataOnly: {
        Tensor<xpu, dim> data = tblob_0->get<xpu, dim, real_t>(s);
        TensorToBlob<xpu, dim>(blob_ptr, caffememtype::Data, &data);
        break;
      }
      case caffeEnum::GradOnly: {
        Tensor<xpu, dim> grad = tblob_0->get<xpu, dim, real_t>(s);
        TensorToBlob<xpu, dim>(blob_ptr, caffememtype::Grad, &grad);
        break;
      }
      case caffeEnum::DataWithGrad: {
        CHECK(tblob_1 != NULL);
        Tensor<xpu, dim> data = tblob_0->get<xpu, dim, real_t>(s);
        Tensor<xpu, dim> grad = tblob_1->get<xpu, dim, real_t>(s);
        TensorToBlob<xpu, dim>(blob_ptr, caffememtype::Data, \
          &data, caffememtype::Grad, &grad);
        break;
      }
      default: {
        LOG(FATAL) << "unexpected fetch type " << fetch_type;
      }
    }
  }

  void BuildOrModifyBlobs(mshadow::Stream<xpu> *s, int fetch_type, const std::vector<int>& dims,
                          bool blobs_inited, std::vector<caffe::Blob<float> *>& blobs,
                          int tblob_start_dim, const std::vector<TBlob>& tblobs_0,
                          const std::vector<TBlob>& tblobs_1) {
    for (size_t i = 0; i < dims.size(); ++i) {
      int dim = dims[i];
      const TBlob* tblob_0 = &tblobs_0[tblob_start_dim + i];
      const TBlob* tblob_1 = NULL;
      if (fetch_type == caffeEnum::DataWithGrad)
        tblob_1 = &tblobs_1[tblob_start_dim + i];
      ::caffe::Blob<float>* blob_ptr;
      if (blobs_inited)
        blob_ptr = blobs[i];
      else
        blob_ptr = new ::caffe::Blob<float>();
      switch (dim) {
        case 1: {
          ConvertTBlob2Blob<1>(s, fetch_type, blob_ptr, tblob_0, tblob_1);
          break;
        }
        case 2: {
          ConvertTBlob2Blob<2>(s, fetch_type, blob_ptr, tblob_0, tblob_1);
          break;
        }
        case 3: {
          ConvertTBlob2Blob<3>(s, fetch_type, blob_ptr, tblob_0, tblob_1);
          break;
        }
        case 4: {
          ConvertTBlob2Blob<4>(s, fetch_type, blob_ptr, tblob_0, tblob_1);
          break;
        }
        default: {
          LOG(FATAL) << "unexpected dim " << dim;
          break;
        }
      }
      if (!blobs_inited)
        blobs.push_back(blob_ptr);
    }
  }

  void HandleOpReqType(mshadow::Stream<xpu>*s, OpReqType req, int shape_dim, const TBlob* in_g) {
    switch (shape_dim) {
      case 1: this->HandleOpReq<1>(s, req, in_g); break;
      case 2: this->HandleOpReq<2>(s, req, in_g); break;
      case 3: this->HandleOpReq<3>(s, req, in_g); break;
      case 4: this->HandleOpReq<4>(s, req, in_g); break;
      default: LOG(FATAL) << "unknown expected weight dim" << shape_dim; break;
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
  ::caffe::Layer<float> *caffeOp_;
  ::std::vector<void*> *weightDataList_, *weightDeltaList_;
  bool initWeight_, initWeightDelta_;
};  // class CaffeOperator

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeOperatorParam param);

#if DMLC_USE_CXX11
class CaffeOperatorProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> res;
    for (size_t i = 0; i < param_.in_dims.size(); ++i)
      res.push_back(std::string("data_") + static_cast<char>('0' + i));

    int blob_cnt = 0;
    if (!param_.op_type_string.compare("InnerProduct")) {
      if (param_.prototxt.inner_product_param().bias_term())
        blob_cnt = 2;
      else
        blob_cnt = 1;
    } else if (!param_.op_type_string.compare("Conv")) {
      if (param_.prototxt.convolution_param().bias_term())
        blob_cnt = 2;
      else
        blob_cnt = 1;
    }
    /*
     * \brief the assumption is: first blob is weight, second is bias.
     * \brief However, some types of caffe-layers might not follow this
     * \brief Customization is then required.
     */
    for (int i = 0; i < blob_cnt; ++i) {
      if (i == 0)
        res.push_back(std::to_string(i) + "_weight");
      else
        res.push_back(std::to_string(i) + "_bias");
    }
    return res;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    CaffeOpInitEntry* e = CaffeOpInitRegistry::Get()->Find(param_.op_type_string);
    param_.caffe_op = e->gen_f_(this->param_.prototxt);
    param_.in_dims.resize(param_.in_num);
    param_.out_dims.resize(param_.out_num);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<int> TShape2Vector(const TShape &tshape) const {
    std::vector<int> s;
    for (unsigned int i =0 ; i < tshape.ndim(); ++i)
      s.push_back(tshape[i]);
    return s;
  }

  TShape Vector2TShape(const std::vector<int> &vec_int) const {
    TShape shp;
    std::vector<index_t> vec_indx;
    for (size_t i = 0; i < vec_int.size(); ++i)
      vec_indx.push_back(vec_int[i]);
    shp = vec_indx;
    return shp;
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
    CHECK_GE(in_shape->size(), param_.in_dims.size());
    // Initialize bottom & top blobs for caffe_op setup
    size_t in_dims_cnt = param_.in_dims.size();
    param_.in_dims.clear();
    vector<Blob<float> *> bot_blobs, top_blobs;
    // Set OperatorParam input dims & caffe op input blobs
    for (size_t i = 0; i < in_dims_cnt; ++i) {
      TShape tshape = (*in_shape)[i];
      if (tshape.ndim() == 0) return false;
      param_.in_dims.push_back(tshape.ndim());
      auto blob_ptr = new Blob<float>();
      blob_ptr->Reshape(this->TShape2Vector(tshape));
      bot_blobs.push_back(blob_ptr);
    }
    // Set caffe op output blobs
    for (size_t i = 0; i < param_.out_dims.size(); ++i) {
      top_blobs.push_back(new Blob<float>());
    }

    param_.caffe_op->SetUp(bot_blobs, top_blobs);
    CHECK_EQ(in_shape->size(), param_.caffe_op->blobs().size() + param_.in_dims.size());
    // Set weight shape
    param_.w_dims.clear();
    for (size_t i = 0; i < param_.caffe_op->blobs().size(); ++i) {
      auto tshape = this->Vector2TShape(param_.caffe_op->blobs()[i]->shape());
      param_.w_dims.push_back(tshape.ndim());
      SHAPE_ASSIGN_CHECK(*in_shape, i + param_.in_dims.size(), tshape);
    }
    // Initialize out dims & out shapes
    param_.out_dims.clear();
    out_shape->clear();
    for (auto blob : top_blobs) {
      auto tshape = this->Vector2TShape(blob->shape());
      param_.out_dims.push_back(tshape.ndim());
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
};  // class FullyConnectedSymbol
#endif

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_INL_H_
