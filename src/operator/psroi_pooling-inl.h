/*!
 * Copyright (c) 2016 by Contributors
 * \file psroi_pooling-inl.h
 * \brief psroi pooling operator and symbol
 * \author Yi Li, Tairui Chen
*/
#ifndef MXNET_OPERATOR_PSROI_POOLING_INL_H_
#define MXNET_OPERATOR_PSROI_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace PSROIPool {
enum PSROIPoolingOpInputs {kData, kBox};
enum PSROIPoolingOpOutputs {kOut, kMaxIdx};
}  // PSROIPool

struct PSROIPoolingParam : public dmlc::Parameter<PSROIPoolingParam> {
  // TShape pooled_size;
  float spatial_scale;
  int output_dim;
  int group_size;
  DMLC_DECLARE_PARAMETER(PSROIPoolingParam) {
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(output_dim).describe("fix output dim");
    DMLC_DECLARE_FIELD(group_size).describe("fix group size");
  }
};

template<typename xpu>
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
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_data[PSROIPool::kOut].shape_[0], in_data[PSROIPool::kBox].shape_[0]);
    CHECK_EQ(out_data[PSROIPool::kMaxIdx].shape_[0], in_data[PSROIPool::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4> data = in_data[PSROIPool::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> bbox = in_data[PSROIPool::kBox].get<xpu, 2, real_t>(s);
    Tensor<xpu, 4> out = out_data[PSROIPool::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mapping_channel = out_data[PSROIPool::kMaxIdx].get<xpu, 4, real_t>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(mapping_channel.CheckContiguous(), true);
    out = -FLT_MAX;
    mapping_channel = -1.0f;
    PSROIPoolForward(out, data, bbox, mapping_channel, param_.spatial_scale, param_.output_dim, param_.group_size);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_grad[PSROIPool::kOut].shape_[0], in_data[PSROIPool::kBox].shape_[0]);
    CHECK_EQ(out_data[PSROIPool::kMaxIdx].shape_[0], in_data[PSROIPool::kBox].shape_[0]);
    CHECK_EQ(req[PSROIPool::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4> grad_out = out_grad[PSROIPool::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> bbox = in_data[PSROIPool::kBox].get<xpu, 2, real_t>(s);
    Tensor<xpu, 4> mapping_channel = out_data[PSROIPool::kMaxIdx].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_in = in_grad[PSROIPool::kData].get<xpu, 4, real_t>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(mapping_channel.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    grad_in = 0.0f;
    PSROIPoolBackward(grad_in, grad_out, bbox, mapping_channel, param_.spatial_scale, param_.output_dim);
  }

 private:
  PSROIPoolingParam param_;
};  // class PSROIPoolingOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(PSROIPoolingParam param);

#if DMLC_USE_CXX11
class PSROIPoolingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "maxidx"};
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
    return 1;
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(PSROIPool::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(PSROIPool::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    // mapping_channel: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.group_size, param_.group_size));
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.group_size, param_.group_size));
    return true;
  }

  OperatorProperty* Copy() const override {
    PSROIPoolingProp* psroi_pooling_sym = new PSROIPoolingProp();
    psroi_pooling_sym->param_ = this->param_;
    return psroi_pooling_sym;
  }

  std::string TypeString() const override {
    return "PSROIPooling";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[PSROIPool::kOut], in_data[PSROIPool::kBox], out_data[PSROIPool::kMaxIdx]};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  PSROIPoolingParam param_;
};  // class PSROIPoolingProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_PSROI_POOLING_INL_H_
