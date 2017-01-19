/*!
 * Copyright (c) 2017 by Contributors
 * \file bilinear_sampling-inl.h
 * \brief
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_BILINEAR_SAMPLING_INL_H_
#define MXNET_OPERATOR_BILINEAR_SAMPLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace bs {
enum BilinearSamplingOpInputs {kData, kGrid};
enum BilinearSamplingOpOutputs {kOut, kTmp};
}

struct BilinearSamplingParam : public dmlc::Parameter<BilinearSamplingParam> {
  DMLC_DECLARE_PARAMETER(BilinearSamplingParam) {
  }
};

template<typename xpu, typename DType>
class BilinearSamplingOp : public Operator {
 public:
  explicit BilinearSamplingOp(BilinearSamplingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[bs::kOut].get<xpu, 4, DType>(s);

    BilinearSamplingForward(out, data, grid);
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
    CHECK_EQ(in_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> ggrid = in_grad[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[bs::kOut].get<xpu, 4, DType>(s);
    gdata = 0.0f;
    ggrid = 0.0f;
    BilinearSamplingBackward(gdata, ggrid, grad, data, grid);
  }

 private:
  BilinearSamplingParam param_;
};  // class BilinearSamplingOp

template<typename xpu>
Operator* CreateOp(BilinearSamplingParam param, int dtype);

#if DMLC_USE_CXX11
class BilinearSamplingProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "grid"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp"};
  }

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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, grid]";
    const TShape &dshape = (*in_shape)[bs::kData];
    const TShape &lshape = (*in_shape)[bs::kGrid];
    if (dshape.ndim() == 0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "input data should be 4D in batch-num_filter-y-x";
    if (lshape.ndim() ==  0) return false;
    CHECK_EQ(lshape.ndim(), 4) \
      << "sampling grid should be 4D in batch-2-y-x";
    CHECK_EQ(dshape[0], lshape[0]);
    CHECK_EQ(lshape[1], 2) << "incorrect grid shape[1], should be 2";
    // target height
    CHECK_GT(lshape[2], 0) \
            << "incorrect grid_shape: " << lshape[2];
    // target width
    CHECK_GT(lshape[3], 0) \
        << "incorrect grid_shape: " << lshape[3];
    out_shape->clear();
    // output_shape : (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3])
    out_shape->push_back(dshape);
    (*out_shape)[bs::kOut][2] = lshape[2];
    (*out_shape)[bs::kOut][3] = lshape[3];
    out_shape->push_back(Shape4(lshape[0], lshape[2], lshape[3], 2));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      int dtype = -1;
      for (size_t i = 0; i < in_type->size(); ++i) {
        if (dtype == -1) {
          dtype = in_type->at(i);
        } else {
          CHECK(in_type->at(i) == dtype ||
                in_type->at(i) == -1) <<
                "Non-uniform data type in BilinearSampling";
        }
      }
      if (dtype == -1) {
        LOG(FATAL) << "Not enough information to infer type in BilinearSampling.";
        return false;
      }
      size_t nin = this->ListArguments().size();
      in_type->clear();
      for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);
      size_t naux = this->ListAuxiliaryStates().size();
      aux_type->clear();
      for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);
      size_t nout = this->ListOutputs().size();
      out_type->clear();
      for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
      return true;
    }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearSamplingProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BilinearSampling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[bs::kOut],
            in_data[bs::kData],
            out_data[bs::kTmp],
            in_data[bs::kGrid]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BilinearSamplingParam param_;
};  // class BilinearSamplingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BILINEAR_SAMPLING_INL_H_
