/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_max_pool-inl.h
 * \brief
 * \author Ziheng Jiang
*/

#ifndef MXNET_OPERATOR_QUANTIZED_MAX_POOL_INL_H_
#define MXNET_OPERATOR_QUANTIZED_MAX_POOL_INL_H_

#include <mxnet/operator_util.h>
#include "../operator_common.h"
#include "../nn/pool.h"

namespace mxnet {
namespace op {

struct QuantizedMaxPoolParam : public dmlc::Parameter<QuantizedMaxPoolParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int layout;
  DMLC_DECLARE_PARAMETER(QuantizedMaxPoolParam) {
    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(stride)
    .set_default(TShape())
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pad)
    .set_default(TShape())
    .describe("pad for pooling: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(layout)
    .set_default(mshadow::kNCHW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NHWC", mshadow::kNHWC);
  }
};

template<typename xpu>
Operator* CreateOp(QuantizedMaxPoolParam param, int dtype);

class QuantizedMaxPoolProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    CHECK_EQ(param_.stride.ndim(), param_.kernel.ndim())
      << "stride and kernel should have the same length";
    CHECK_EQ(param_.pad.ndim(), param_.kernel.ndim())
      << "pad and kernel should have the same length";
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "min_range", "max_range"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "min_range", "max_range"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 3U);
    CHECK(!shape_is_none(in_shape->at(0)));
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4U)
        << "MaxPool: Input data should be 4D in "
        << "(batch, channel, y, x)";
    int N = -1, H = -1, W = -1, C = -1;
    if (param_.layout == mshadow::kNCHW) {
      N = 0, H = 2, W = 3, C = 1;
    } else if (param_.layout == mshadow::kNHWC) {
      N = 0, H = 1, W = 2, C = 3;
    } else {
      LOG(FATAL) << "not support other layout for now";
    }

    TShape oshape(4);
    CHECK_EQ(param_.kernel.ndim(), 2);

    CHECK(param_.kernel[0] <= dshape[H] + 2 * param_.pad[0])
        << "kernel size (" << param_.kernel[0]
        << ") exceeds input (" << dshape[H]
        << " padded to " << (dshape[H] + 2*param_.pad[0]) << ")";
    CHECK(param_.kernel[1] <= dshape[W] + 2 * param_.pad[1])
        << "kernel size (" << param_.kernel[1]
        << ") exceeds input (" << dshape[W]
        << " padded to " << (dshape[W] + 2*param_.pad[1]) << ")";

    // only support valid convention
    oshape[N] = dshape[N];
    oshape[C] = dshape[C];
    oshape[H] = 1 + (dshape[H] + 2 * param_.pad[0] - param_.kernel[0]) /
        param_.stride[0];
    oshape[W] = 1 + (dshape[W] + 2 * param_.pad[1] - param_.kernel[1]) /
        param_.stride[1];

    CHECK(shape_is_scalar(in_shape->at(1)));
    CHECK(shape_is_scalar(in_shape->at(2)));

    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 3U);
    CHECK_EQ((*in_type)[0], mshadow::kInt8)
      << "`dequantized_relu` only supports uint8 input for now";
    CHECK_EQ((*in_type)[1], mshadow::kFloat32)
      << "the second input of `dequantized_relu` should be a tensor "
      << "with type of float32";
    CHECK_EQ((*in_type)[2], mshadow::kFloat32)
      << "the third input of `dequantized_relu` should be a tensor "
      << "with type of float32";

    out_type->clear();
    out_type->push_back(mshadow::kInt8);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    QuantizedMaxPoolProp *prop_sym = new QuantizedMaxPoolProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "quantized_max_pool";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData],
            out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedMaxPoolParam param_;
};  // class QuantizedMaxPoolProp

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_
