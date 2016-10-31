/*!
 * Copyright (c) 2016 by Contributors
 * \file scale-inl.h
 * \brief A scaling layer with inital scale, and adjusted with backprop
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_SCALE_INL_H_
#define MXNET_OPERATOR_SCALE_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
namespace scale_enum {
enum ScaleOpInputs {kData, kWeight};
enum ScaleOpOutputs {kOut};
enum ScaleOpResource {kTempSpace};
enum ScaleOpType {kInstance, kChannel, kSpatial};
}  // scale_enum

struct ScaleParam : public dmlc::Parameter<ScaleParam> {
  int mode;
  DMLC_DECLARE_PARAMETER(ScaleParam) {
    DMLC_DECLARE_FIELD(mode)
    .add_enum("instance", scale_enum::kInstance)
    .add_enum("spatial", scale_enum::kSpatial)
    .add_enum("channel", scale_enum::kChannel)
    .set_default(scale_enum::kInstance)
    .describe("Scaling Mode. If set to instance, this operator will use independent "
    "scale for each instance in the batch; this is the default mode. "
    "If set to channel, this operator will share scales cross channel at "
    "each position of each instance. If set to spatial, this operator shares scales "
    "in each channel.");
  }
};  // struct ScaleParam

template<typename xpu, typename DType>
class ScaleOp : public Operator {
 public:
  explicit ScaleOp(ScaleParam p) {
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
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    TShape orig_shape = in_data[scale_enum::kData].shape_;
    index_t nbatch = orig_shape[0];
    if (param_.mode == scale_enum::kInstance) {
      Shape<2> dshape = Shape2(orig_shape[0],
        orig_shape.ProdShape(1, orig_shape.ndim()));
      Tensor<xpu, 2, DType> data = in_data[scale_enum::kData]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> out = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 1, DType> weight = in_data[scale_enum::kWeight].get<xpu, 1, DType>(s);
      out = data * broadcast<0>(broadcast_keepdim(weight, 0, nbatch), out.shape_);
    } else if (param_.mode == scale_enum::kChannel) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = in_data[scale_enum::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> out = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 2, DType> weight = in_data[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(Shape2(1, dshape[2]), s);
      out = data * broadcast_with_axis(
        broadcast_keepdim(weight, 0, nbatch), 0, dshape[1]);
    } else if (param_.mode == scale_enum::kSpatial) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = in_data[scale_enum::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> out = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 2, DType> weight = in_data[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(Shape2(1, dshape[1]), s);
      out = data * broadcast_with_axis(
        broadcast_keepdim(weight, 0, nbatch), 1, dshape[2]);
    } else {
      LOG(FATAL) << "Unknown scaling mode.";
    }
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
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    TShape orig_shape = out_data[scale_enum::kOut].shape_;
    index_t nbatch = orig_shape[0];
    if (param_.mode == scale_enum::kInstance) {
      Shape<2> dshape = Shape2(orig_shape[0],
        orig_shape.ProdShape(1, orig_shape.ndim()));
      Tensor<xpu, 2, DType> data = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> grad_in = in_grad[scale_enum::kData]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> grad_out = out_grad[scale_enum::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 1, DType> wgrad_in = in_grad[scale_enum::kWeight].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> weight = in_data[scale_enum::kWeight].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> temp = ctx.requested[scale_enum::kTempSpace]
        .get_space_typed<xpu, 1, DType>(mshadow::Shape1(1), s);
      temp = sumall_except_dim<0>(reduce_keepdim<red::sum, false>(grad_out * data, 0));
      Assign(wgrad_in, req[scale_enum::kWeight], temp / weight);
      Assign(grad_in, req[scale_enum::kData], grad_out *
        broadcast<0>(broadcast_keepdim(weight, 0, nbatch), grad_out.shape_));
    } else if (param_.mode == scale_enum::kChannel) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_in = in_grad[scale_enum::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_out = out_grad[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> wshape = Shape2(1, dshape[2]);
      Tensor<xpu, 2, DType> wgrad_in = in_grad[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(wshape, s);
      Tensor<xpu, 2, DType> weight = in_data[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(wshape, s);
      Tensor<xpu, 2, DType> temp = ctx.requested[scale_enum::kTempSpace]
        .get_space_typed<xpu, 2, DType>(mshadow::Shape2(1, data.shape_[2]), s);
      temp = reduce_keepdim<red::sum, false>(
        reduce_with_axis<red::sum, false>(grad_out * data, 1), 0);
      Assign(wgrad_in, req[scale_enum::kWeight], temp / weight);
      Assign(grad_in, req[scale_enum::kWeight],
        grad_out * broadcast_with_axis(
        broadcast_keepdim(weight, 0, nbatch), 0, orig_shape[1]));
    } else if (param_.mode == scale_enum::kSpatial) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = out_data[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_in = in_grad[scale_enum::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_out = out_grad[scale_enum::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> wshape = Shape2(1, dshape[1]);
      Tensor<xpu, 2, DType> wgrad_in = in_grad[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(wshape, s);
      Tensor<xpu, 2, DType> weight = in_data[scale_enum::kWeight]
        .get_with_shape<xpu, 2, DType>(wshape, s);
      Tensor<xpu, 2, DType> temp = ctx.requested[scale_enum::kTempSpace]
        .get_space_typed<xpu, 2, DType>(mshadow::Shape2(1, data.shape_[1]), s);
      temp = reduce_keepdim<red::sum, false>(
        reduce_with_axis<red::sum, false>(grad_out * data, 2), 0);
      Assign(wgrad_in, req[scale_enum::kWeight], temp / weight);
      Assign(grad_in, req[scale_enum::kData],
        grad_out * broadcast_with_axis(
        broadcast_keepdim(weight, 0, nbatch), 1, dshape[2]));
    } else {
      LOG(FATAL) << "Unknown scaling mode";
    }
  }

 private:
  ScaleParam param_;
};  // class ScaleOp

template<typename xpu>
Operator* CreateOp(ScaleParam param, int dtype);

#if DMLC_USE_CXX11
class ScaleProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "scale"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
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
  CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
  const TShape &dshape = (*in_shape)[scale_enum::kData];
  if (dshape.ndim() ==  0) return false;
  if (param_.mode == scale_enum::kInstance) {
    CHECK_GE(dshape.ndim(), 2);
    SHAPE_ASSIGN_CHECK(*in_shape, scale_enum::kWeight, Shape1(1));
  } else if (param_.mode == scale_enum::kChannel) {
    CHECK_GE(dshape.ndim(), 3)
     << "At least 3 dimensions required in channel mode";
     SHAPE_ASSIGN_CHECK(*in_shape, scale_enum::kWeight,
       Shape1(dshape.ProdShape(2, dshape.ndim())));
  } else if (param_.mode == scale_enum::kSpatial) {
    CHECK_GE(dshape.ndim(), 3)
     << "At least 3 dimensions required in spatial mode";
     SHAPE_ASSIGN_CHECK(*in_shape, scale_enum::kWeight, Shape1(dshape[1]));
  }
  out_shape->clear();
  out_shape->push_back(dshape);
  return true;
  }



  OperatorProperty* Copy() const override {
  auto ptr = new ScaleProp();
  ptr->param_ = param_;
  return ptr;
  }

  std::string TypeString() const override {
  return "Scale";
  }

  std::vector<int> DeclareBackwardDependency(
  const std::vector<int> &out_grad,
  const std::vector<int> &in_data,
  const std::vector<int> &out_data) const override {
  return {out_grad[scale_enum::kOut], out_data[scale_enum::kOut], in_data[scale_enum::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
  return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
  return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
  LOG(FATAL) << "Not implemented";
  return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  ScaleParam param_;
};   // class ScaleProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SCALE_INL_H_
