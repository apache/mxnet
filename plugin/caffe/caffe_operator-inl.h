/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator-inl.h
 * \brief Caffe Operator and symbol 
 * \author Haoran Wang 
*/
#ifndef MXNET_OPERATOR_CAFFE_OPERATOR_INL_H_
#define MXNET_OPERATOR_CAFFE_OPERATOR_INL_H_

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

#include "caffe_mode.h"
#include "layerparameter_stream.h"
#include "dmlc_parameter.h"
#include "tensor_to_blob.h"
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace caffeEnum {
enum CaffeOpInputs {kData};
enum CaffeOpOutputs {kOut};
enum CaffeOpType {fullyconnected, tanh, relu, conv};
}  // caffeop

struct CaffeOperatorParam : public dmlc::Parameter<CaffeOperatorParam> {
  caffe::LayerParameter para;
  std::string op_type_name;
  caffe::Layer<float> *caffe_op;

  int input_dim, out_dim;
  int op_type_value;
  DMLC_DECLARE_PARAMETER(CaffeOperatorParam) {
    DMLC_DECLARE_FIELD(para);
    DMLC_DECLARE_FIELD(op_type_name);
  }
};


typedef caffe::Layer<float>* (*pFunc) (caffe::LayerParameter);

class CaffeTypeNameMap{
public:
  static void DoInit();
  static pFunc toFn(std::string name);
  static int toVal(std::string name);
private:  
  static bool init;
  static std::map<std::string, pFunc> to_gen_func;
  static std::map<std::string, int> to_type_value;
};


/**
 * \brief This is the implementation of caffe operator in Caffe.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class CaffeOperator : public Operator {
 public:
  explicit CaffeOperator(CaffeOperatorParam p):param_(p), caffeOp_(p.caffe_op), inputDim_(p.input_dim), outDim_(p.out_dim), initWeight_(false), initWeightDelta_(false) {
  }
  
  void CaffeForward(std::vector<caffe::Blob<float>*> bottom, std::vector<caffe::Blob<float>*> top);
  
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {

    CaffeMode::SetMode<xpu>();
    if ((inputDim_ == 2)&&(outDim_ == 2))
      RealForward<2, 2>(ctx, in_data, req, out_data, aux_args);
    else if ((inputDim_ == 4)&&(outDim_ == 2))
      RealForward<4, 2>(ctx, in_data, req, out_data, aux_args);
    else if ((inputDim_ == 4)&&(outDim_ == 4))
      RealForward<4, 4>(ctx, in_data, req, out_data, aux_args);
    else
      LOG(FATAL) << "unexpected input dim " << inputDim_ <<" output dim" << outDim_;
  }

  template<size_t input_dim, size_t out_dim>
  void RealForward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[caffeEnum::kOut], kWriteTo);

    // input num cannot be accessed directly from caffe layer without doing layer set-up
    size_t expected_in_num = caffeOp_->blobs().size() + 1;
    CHECK_EQ(in_data.size(), expected_in_num);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    if ( param_.op_type_value == caffeEnum::fullyconnected)
      CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    Tensor<xpu, input_dim> data = in_data[caffeEnum::kData].get<xpu, input_dim, real_t>(s);
    Tensor<xpu, out_dim> out = out_data[caffeEnum::kOut].get<xpu, out_dim, real_t>(s);

    ::caffe::Blob<float> *bottomBlobPtr = new ::caffe::Blob<float>(), *topBlobPtr = new ::caffe::Blob<float>();
    ::caffe::TensorToBlob<xpu, input_dim>(bottomBlobPtr, ::caffe::caffememtype::Data, &data);
    ::caffe::TensorToBlob<xpu, out_dim>(topBlobPtr, ::caffe::caffememtype::Data, &out);

    //weight has to be pointer. Caffe will free old weights when new ones come. You want to keep content before caffe frees 
    std::vector<caffe::Blob<float>*> weightBlobPtrs; 

    for(int i = 1; i < expected_in_num; ++ i){
      int shape_dim = caffeOp_->blobs()[i-1]->shape().size();
      weightBlobPtrs.push_back(new ::caffe::Blob<float>());
      ConvertToCaffeBlobInForward(s, shape_dim, &in_data[i], weightBlobPtrs[i-1]);
    }

    if (!initWeight_){
      initWeight_ = true;

      weightDataRecord_ForDebug = new std::vector<void*>();
      for(int i = 1; i < expected_in_num; ++ i)
        weightDataRecord_ForDebug->push_back(in_data[i].dptr_);

      caffeOp_->SetLearnableWeights(weightBlobPtrs);
    } else {
      // checking info. make sure pointers of weights caffe layer are the same as pointers of weights passed in
      for(int i = 1; i < expected_in_num; ++ i)
        CHECK_EQ(weightDataRecord_ForDebug->at(i-1), in_data[i].dptr_);
    }

    std::vector<caffe::Blob<float>*> botVec, topVec;
    botVec.push_back(bottomBlobPtr);
    topVec.push_back(topBlobPtr);
    this->CaffeForward(botVec, topVec);
  }

  void CaffeBackward(std::vector<caffe::Blob<float>*> top, std::vector<bool> bp_flags, std::vector<caffe::Blob<float>*> bottom);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    CaffeMode::SetMode<xpu>();
    if ((inputDim_ == 2)&&(outDim_ == 2))
      RealBackward<2, 2>(ctx, out_grad, in_data, out_data, req, in_grad, aux_args);
    else if ((inputDim_ == 4)&&(outDim_ == 2))
      RealBackward<4, 2>(ctx, out_grad, in_data, out_data, req, in_grad, aux_args);
    else if ((inputDim_ == 4)&&(outDim_ == 4))
      RealBackward<4, 4>(ctx, out_grad, in_data, out_data, req, in_grad, aux_args);
    else
      LOG(FATAL) << "unexpected input dim " << inputDim_ <<" output dim" << outDim_;
  }

  template<size_t input_dim, size_t out_dim>
  void RealBackward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args){
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(req[caffeEnum::kData] != kAddTo) << "caffe not support  write add to";

    size_t expected_in_num = caffeOp_->blobs().size() + 1;
    CHECK(in_data.size() == expected_in_num && in_grad.size() == expected_in_num);
    CHECK_EQ(req.size(), expected_in_num);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    if(param_.op_type_value == caffeEnum::fullyconnected)
      CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif //__CUDACC__

    Tensor<xpu, input_dim> data = in_data[caffeEnum::kData].get<xpu, input_dim, real_t>(s),
                          gdata = in_grad[caffeEnum::kData].get<xpu, input_dim, real_t>(s);
    Tensor<xpu, out_dim> out = out_data[caffeEnum::kOut].get<xpu, out_dim, real_t>(s),
                        gout = out_grad[caffeEnum::kOut].get<xpu, out_dim, real_t>(s);

    caffe::Blob<float> *bottomBlobPtr = new ::caffe::Blob<float>(), *topBlobPtr = new ::caffe::Blob<float>();
    caffe::TensorToBlob<xpu, input_dim>(bottomBlobPtr, caffe::caffememtype::Data, &data, caffe::caffememtype::Grad, &gdata);
    caffe::TensorToBlob<xpu, out_dim>(topBlobPtr, caffe::caffememtype::Data, &out, caffe::caffememtype::Grad, &gout);


    if (!initWeightDelta_){
      initWeightDelta_ = true;
      weightDeltaRecord_ForDebug = new std::vector<void*>();
      for(int i = 1; i < expected_in_num; ++ i)
        weightDeltaRecord_ForDebug->push_back(in_grad[i].dptr_);

      std::vector<caffe::Blob<float>*> weightBlobPtrs = caffeOp_->GetLearnableWeights(); 
      for(int i = 1; i < expected_in_num; ++ i){
        int shape_dim = caffeOp_->blobs()[i-1]->shape().size();
        this->ConvertToCaffeBlobInBackward(s, req[i], shape_dim, &in_grad[i], weightBlobPtrs[i-1]);
      }
    } else {
      // checking info. make sure pointers of weights caffe layer are the same as pointers of weights passed in
      for(int i = 1; i < expected_in_num; ++ i){
        CHECK_EQ(weightDeltaRecord_ForDebug->at(i-1), in_grad[i].dptr_);
        CHECK_EQ(weightDataRecord_ForDebug->at(i-1), in_data[i].dptr_);
      }
    }

    for(int i = 1; i < expected_in_num; ++ i){
        int shape_dim = caffeOp_->blobs()[i-1]->shape().size();
        this->HandleOpReqType(s, req[i], shape_dim, &in_grad[i]);
    }
    
    std::vector<caffe::Blob<float>*> topVec, botVec;
    std::vector<bool> flagVec;

    //deal with OpReqType
    flagVec.push_back(req[caffeEnum::kData] != kNullOp);
    topVec.push_back(topBlobPtr);
    botVec.push_back(bottomBlobPtr);
    CaffeBackward(topVec, flagVec, botVec);
  }

  void ConvertToCaffeBlobInForward(mshadow::Stream<xpu>*s, int shape_dim,
                            const TBlob* in, caffe::Blob<float>* w){
    switch (shape_dim){
      case 1: this->ConvWeiFor<1>(s, in, w); break;
      case 2: this->ConvWeiFor<2>(s, in, w); break;
      case 3: this->ConvWeiFor<3>(s, in, w); break;
      case 4: this->ConvWeiFor<4>(s, in, w); break;
      default: LOG(FATAL) << "unknown expected weight dim" << shape_dim; break;
    }
  }
  
  template<size_t dim>
  void ConvWeiFor(mshadow::Stream<xpu>*s,  
                            const TBlob* in_data, caffe::Blob<float>* weight_blob){
    mshadow::Tensor<xpu, dim> w = in_data->get<xpu, dim, real_t>(s);
    caffe::TensorToBlob<xpu, dim>(weight_blob, caffe::caffememtype::Data, &w);
  }


  void HandleOpReqType(mshadow::Stream<xpu>*s, OpReqType req, int shape_dim, const TBlob* in_g){
    switch (shape_dim){
      case 1: this->HandleOpReq<1>(s, req, in_g); break;
      case 2: this->HandleOpReq<2>(s, req, in_g); break;
      case 3: this->HandleOpReq<3>(s, req, in_g); break;
      case 4: this->HandleOpReq<4>(s, req, in_g); break;
      default: LOG(FATAL) << "unknown expected weight dim" << shape_dim; break;
    }
  }

  void ConvertToCaffeBlobInBackward(mshadow::Stream<xpu>*s, OpReqType req, int shape_dim, 
                                    const TBlob* in_g,  caffe::Blob<float>* w){
    switch (shape_dim){
      case 1: this->ConvWeiBac<1>(s, req, in_g, w); break;
      case 2: this->ConvWeiBac<2>(s, req, in_g, w); break;
      case 3: this->ConvWeiBac<3>(s, req, in_g, w); break;
      case 4: this->ConvWeiBac<4>(s, req, in_g, w); break;
      default: LOG(FATAL) << "unknown expected weight dim" << shape_dim; break;
    }
  }

  template<size_t dim>
  void HandleOpReq(mshadow::Stream<xpu>*s, OpReqType req, const TBlob* in_grad){
    mshadow::Tensor<xpu, dim> w_g = in_grad->get<xpu, dim, real_t>(s);
    if ((req == kWriteInplace)||(req == kWriteTo))
      w_g = 0;
  }

  template<size_t dim>
  void ConvWeiBac(mshadow::Stream<xpu>*s, OpReqType req,
                  const TBlob* in_grad, caffe::Blob<float>* weight_blob){
    mshadow::Tensor<xpu, dim> w_g = in_grad->get<xpu, dim, real_t>(s);
    caffe::TensorToBlob<xpu, dim>(weight_blob, caffe::caffememtype::Grad, &w_g);
  }

 private:
  CaffeOperatorParam param_;
  ::caffe::Layer<float> *caffeOp_;
  ::std::vector<void*> *weightDataRecord_ForDebug, *weightDeltaRecord_ForDebug;
  int inputDim_, outDim_;
  bool initWeight_, initWeightDelta_;
};// class CaffeOperator

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CaffeOperatorParam param);

#if DMLC_USE_CXX11
class CaffeOperatorProp : public OperatorProperty {
 public:
  //TODO(Haoran): when to delete caffe op. 
  std::vector<std::string> ListArguments() const override {
    ::caffe::Layer<float> *tmp_op = CaffeTypeNameMap::toFn(param_.op_type_name)(this->param_.para);
    std::vector<std::string> res = {"data"};

    int blobs_size = tmp_op->GetWeightsNumber();
    for (int i = 0; i < blobs_size; ++ i){
    //TODO(Haoran): needs to assign by name
      if (i == 0)
        res.push_back("caffe_" + std::to_string(i) + "_weight");
      else
        res.push_back("caffe_" + std::to_string(i) + "_bias");
    }
    return res; 
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    param_.op_type_value = CaffeTypeNameMap::toVal(param_.op_type_name);
    param_.caffe_op = CaffeTypeNameMap::toFn(param_.op_type_name)(this->param_.para);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<int> TShapeToVector(const TShape &dshape) const{
    std::vector<int> s;
    for (unsigned int i =0 ; i < dshape.ndim(); ++ i)
      s.push_back(dshape[i]);
    return s;
  }

  TShape ToTShape(const std::vector<int> &vec_int) const{
    TShape shp;
    std::vector<index_t> vec_indx;

    for (auto v : vec_int)
      vec_indx.push_back(v);

    shp = vec_indx;
    return shp;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_GE(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[caffeEnum::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    ::caffe::Blob<float> bottomBlob, topBlob;
    auto bottom_shape = this->TShapeToVector(dshape);
    bottomBlob.Reshape(bottom_shape);
    param_.caffe_op->SetUp({&bottomBlob}, {&topBlob});

    param_.input_dim = dshape.ndim();

    CHECK_EQ(in_shape->size(), param_.caffe_op->blobs().size() + 1);

    out_shape->clear();
    //Set Weight & Bias Shape
    //keep the same order to weight blobs
    TShape shp;
    for (unsigned int i = 0; i < param_.caffe_op->blobs().size(); ++ i){
      shp = this->ToTShape(param_.caffe_op->blobs()[i]->shape());
      SHAPE_ASSIGN_CHECK(*in_shape, i + 1, shp);
    }

    out_shape->push_back(this->ToTShape(topBlob.shape()));
    param_.out_dim = (*out_shape)[0].ndim();
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
#endif  // MXNET_OPERATOR_CAFFE_OPERATOR_INL_H_
