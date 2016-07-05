/*!
 * Copyright (c) 2016 by Contributors
 * \file moon_output-inl.h
 * \brief
 *  This is the moon loss operator, which comes from the paper:
 *	Rudd E, G¨¹nther M, Boult T. MOON: A Mixed Objective Optimization Network for the Recognition of Facial Attributes[J].
 *  arXiv preprint arXiv:1603.07027, 2016.
 *  the moon loss operator is usually used in multi-binary-label application, which every binary label is +1 and -1;
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_MOON_OUTPUT_INL_H_
#define MXNET_OPERATOR_MOON_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace moonout_enum {
enum MoonOutputOpInputs {kData, kLabel};
enum MoonOutputOpOutputs {kOut};
}  // namespace moonout_enum

struct MoonOutputParam : public dmlc::Parameter<MoonOutputParam> {
  float grad_scale;
  std::string src_dist_path;
  DMLC_DECLARE_PARAMETER(MoonOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
	DMLC_DECLARE_FIELD(src_dist_path).set_default("src_dist.txt")
		.describe("the parameters file of src distribution");
  };
};

template<typename xpu, typename DType>
class MoonOutputOp : public Operator {
 public:
  explicit MoonOutputOp(MoonOutputParam param) : param_(param) {
	  std::ifstream ifs;
	  ifs.open(param_.src_dist_path.c_str(), std::ifstream::in);
	  float tmp;
	  while (ifs >> tmp) {
		  src_dist_.push_back(tmp);
	  }
	  ifs.close();
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
	using namespace mshadow;
	using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "MoonOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "MoonOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[moonout_enum::kData].FlatTo2D<xpu, DType>(s);
	Tensor<xpu, 2, DType> label = in_data[moonout_enum::kLabel].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[moonout_enum::kOut].FlatTo2D<xpu, DType>(s);
	CHECK_EQ(data.shape_, out.shape_) << "Moon: shape mismatch between input and output";
	CHECK_EQ(label.shape_, out.shape_) << "Moon: shape mismatch between label and output";
	CHECK_EQ(data.shape_[1], src_dist_.size()) << "Moon: shape mismatch between input channel and number parmaters in src_dist.txt";
	Assign(out, req[moonout_enum::kOut], F<mshadow::op::identity>(data));
	//out = data;
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
	Tensor<xpu, 2, DType> label = in_data[moonout_enum::kLabel].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[moonout_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = in_grad[moonout_enum::kData].FlatTo2D<xpu, DType>(s);
	MoonBackward(grad, out, label, src_dist_);
    grad *= DType(param_.grad_scale/label.size(1)); // normalize the gradient by number labels
  }

 private:
  MoonOutputParam param_;
  std::vector<float> src_dist_;
};  // class MoonOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(MoonOutputParam param, int dtype);

#if DMLC_USE_CXX11
class MoonOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, moonout_enum::kLabel,
                       Shape2(dshape[0], dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MoonOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MoonOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[moonout_enum::kLabel], out_data[moonout_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[moonout_enum::kOut], in_grad[moonout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[moonout_enum::kData], out_data[moonout_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  MoonOutputParam param_;
};  // class MoonOutputProp

class DeprecatedMoonProp : public MoonOutputProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    MoonOutputProp::param_.Init(kwargs);
  }

  std::string TypeString() const override {
    return "Moon";
  }
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MOON_OUTPUT_INL_H_
