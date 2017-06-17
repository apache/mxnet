/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling-inl.h
 * \brief psroi pooling operator and symbol
 * \author Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi, Jifeng Dai
*/
#ifndef MXNET_OPERATOR_CONTRIB_PSROI_POOLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_PSROI_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace psroipool {
enum PSROIPoolingOpInputs {kData, kBox};
enum PSROIPoolingOpOutputs {kOut};
}  // psroipool

struct PSROIPoolingParam : public dmlc::Parameter<PSROIPoolingParam> {
  // TShape pooled_size;
  float spatial_scale;
  int output_dim;
  int pooled_size;
  int group_size;
  DMLC_DECLARE_PARAMETER(PSROIPoolingParam) {
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(output_dim).describe("fix output dim");
  DMLC_DECLARE_FIELD(pooled_size).describe("fix pooled size");
    DMLC_DECLARE_FIELD(group_size).set_default(0).describe("fix group size");
  }
};

template<typename xpu, typename DType>
class PSROIPoolingOp : public Operator {
 public:
  explicit PSROIPoolingOp(PSROIPoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_data[psroipool::kOut].shape_[0], in_data[psroipool::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[psroipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[psroipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[psroipool::kOut].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    out = -FLT_MAX;
    PSROIPoolForward(out, data, bbox, param_.spatial_scale, param_.output_dim, param_.group_size);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_grad[psroipool::kOut].shape_[0], in_data[psroipool::kBox].shape_[0]);
    CHECK_NE(req[psroipool::kData], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[psroipool::kBox], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[psroipool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[psroipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[psroipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[psroipool::kBox].get<xpu, 2, DType>(s);

    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);

    if (kAddTo == req[psroipool::kData] || kWriteTo == req[psroipool::kData]) {
      if (kWriteTo == req[psroipool::kData]) {
        grad_in = 0.0f;
      }
      PSROIPoolBackwardAcc(grad_in, grad_out, bbox, param_.spatial_scale,
                           param_.output_dim, param_.group_size);
    }
    if (kWriteTo == req[psroipool::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  PSROIPoolingParam param_;
};  // class PSROIPoolingOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(PSROIPoolingParam param, int dtype);

#if DMLC_USE_CXX11
class PSROIPoolingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  if (param_.group_size == 0) {
    param_.group_size = param_.pooled_size;
  }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(psroipool::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(psroipool::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
         Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    PSROIPoolingProp* psroi_pooling_sym = new PSROIPoolingProp();
    psroi_pooling_sym->param_ = this->param_;
    return psroi_pooling_sym;
  }

  std::string TypeString() const override {
    return "_contrib_PSROIPooling";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[psroipool::kOut], in_data[psroipool::kBox]};
  }


  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;


 private:
  PSROIPoolingParam param_;
};  // class PSROIPoolingProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_PSROI_POOLING_INL_H_
