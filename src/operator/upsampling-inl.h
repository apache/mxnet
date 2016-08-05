/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_UPSAMPLING_INL_H_
#define MXNET_OPERATOR_UPSAMPLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace up_enum {
enum UpSamplingOpInputs {kData, kWeight};
enum UpSamplingOpOutputs {kOut};
enum UpSamplingType {kNearest, kBilinear};
enum UpSamplingMultiInputMode {kConcat, kSum};
}  // namespace up_enum

struct UpSamplingParam : public dmlc::Parameter<UpSamplingParam> {
  index_t scale;
  index_t num_filter;
  int sample_type;
  int num_args;
  int multi_input_mode;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(UpSamplingParam) {
    DMLC_DECLARE_FIELD(scale)
    .set_range(1, 1000)
    .describe("Up sampling scale");
    DMLC_DECLARE_FIELD(num_filter)
    .describe("Input filter. Only used by bilinear sample_type.")
    .set_default(0);
    DMLC_DECLARE_FIELD(sample_type)
    .add_enum("nearest", up_enum::kNearest)
    .add_enum("bilinear", up_enum::kBilinear)
    .describe("upsampling method");
    DMLC_DECLARE_FIELD(multi_input_mode)
    .add_enum("concat", up_enum::kConcat)
    .add_enum("sum", up_enum::kSum)
    .set_default(up_enum::kConcat)
    .describe("How to handle multiple input. concat means concatenate upsampled "
    "images along the channel dimension. sum means add all images together, "
    "only available for nearest neighbor upsampling.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be upsampled. For nearest neighbor "
    "upsampling, this can be 1-N; the size of output will be"
    "(scale*h_0,scale*w_0) and all other inputs will be upsampled to the"
    "same size. For bilinear upsampling this must be 2; 1 input and 1 weight.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
    .describe("Tmp workspace for deconvolution (MB)");
  }
};  // struct UpSamplingParam

template<typename xpu, typename DType>
class UpSamplingNearestOp : public Operator {
 public:
  explicit UpSamplingNearestOp(UpSamplingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), param_.num_args);
    CHECK_EQ(out_data.size(), 1);
    if (req[up_enum::kOut] == kNullOp) {
      return;
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> out = out_data[up_enum::kOut].get<xpu, 4, DType>(s);
    if (param_.num_args > 1) {
      int begin = 0;
      for (int i = 0; i < param_.num_args; ++i) {
        Tensor<xpu, 4, DType> data = in_data[i].get<xpu, 4, DType>(s);
        int end = begin + data.size(1);
        int scale = out_data[up_enum::kOut].size(2)/in_data[i].size(2);
        if (param_.multi_input_mode == up_enum::kSum) {
          if (i == 0) {
            Assign(out, req[up_enum::kOut], upsampling_nearest(data, scale));
          } else {
            out += upsampling_nearest(data, scale);
          }
        } else {
          Assign(slice<1>(out, begin, end), req[up_enum::kOut], upsampling_nearest(data, scale));
        }
        begin = end;
      }
    } else {
      Tensor<xpu, 4, DType> data = in_data[up_enum::kData].get<xpu, 4, DType>(s);
      Assign(out, req[up_enum::kOut], upsampling_nearest(data, param_.scale));
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), param_.num_args);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad = out_grad[up_enum::kOut].get<xpu, 4, DType>(s);
    if (param_.num_args > 1) {
      int begin = 0;
      for (int i = 0; i < param_.num_args; ++i) {
        Tensor<xpu, 4, DType> input_grad = in_grad[i].get<xpu, 4, DType>(s);
        mshadow::Shape<2> in_shape = Shape2(input_grad.shape_[2], input_grad.shape_[3]);
        int end = begin + input_grad.size(1);
        int scale = grad.size(2)/in_shape[0];
        if (param_.multi_input_mode == up_enum::kSum) {
          Assign(input_grad, req[i],
                 pool<mshadow::red::sum>(grad,
                                         in_shape,
                                         scale,
                                         scale,
                                         scale,
                                         scale));
        } else {
          Assign(input_grad, req[i],
                 pool<mshadow::red::sum>(slice<1>(grad, begin, end),
                                         in_shape,
                                         scale,
                                         scale,
                                         scale,
                                         scale));
        }
        begin = end;
      }
    } else {
      Tensor<xpu, 4, DType> input_grad = in_grad[up_enum::kData].get<xpu, 4, DType>(s);
      mshadow::Shape<2> in_shape = Shape2(input_grad.shape_[2], input_grad.shape_[3]);
      Assign(input_grad, req[up_enum::kData],
             pool<mshadow::red::sum>(grad,
                                     in_shape,
                                     param_.scale,
                                     param_.scale,
                                     param_.scale,
                                     param_.scale));
    }
  }

 private:
  UpSamplingParam param_;
};  // class UpSamplingNearestOp

template<typename xpu>
Operator *CreateOp(UpSamplingParam param, int dtype);


#if DMLC_USE_CXX11
class UpSamplingProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.sample_type == up_enum::kNearest) {
      std::vector<std::string> ret;
      for (int i = 0; i < param_.num_args; ++i) {
        ret.push_back(std::string("arg") + static_cast<char>('0' + i));
      }
      return ret;
    } else {
      return {"data", "weight"};
    }
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_GE(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    TShape oshape = dshape;
    if (param_.sample_type == up_enum::kNearest) {
      CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
      oshape[1] = 0;
      for (auto& shape : *in_shape) {
        CHECK_EQ(shape.ndim(), 4) << \
          "UpSamplingNearest: Input data should be 4D in (batch, channel, y, x)";
        int oh = dshape[2]*param_.scale, ow = dshape[3]*param_.scale;
        CHECK_EQ(oh%shape[2], 0) << "UpSamplingNearest: input height of " << shape[2] << \
          "does not divide output height of " << oh;
        CHECK_EQ(ow%shape[3], 0) << "UpSamplingNearest: input weight of " << shape[3] << \
          "does not divide output width of " << ow;
        if (param_.multi_input_mode == up_enum::kSum) {
          CHECK(oshape[1] == 0 || oshape[1] == shape[1]) << \
            "Number of channels must be the same when multi_input_mode==sum";
          oshape[1] = shape[1];
        } else {
          oshape[1] += shape[1];
        }
      }
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
      CHECK_EQ(dshape.ndim(), 4) << \
        "UpSamplingBilinear: Input data should be 4D in (batch, channel, y, x)";
      if (dshape.ndim() ==  0) return false;
      int kernel = 2 * param_.scale - param_.scale % 2;
      SHAPE_ASSIGN_CHECK(*in_shape,
                         up_enum::kWeight,
                         mshadow::Shape4(dshape[1], 1, kernel, kernel));
      oshape = dshape;
    }
    oshape[2] = dshape[2] * param_.scale;
    oshape[3] = dshape[3] * param_.scale;
    out_shape->clear();
    out_shape->push_back(oshape);
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
    auto ptr = new UpSamplingProp();
    ptr->param_ = this->param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "UpSampling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.sample_type == up_enum::kNearest) {
      return {out_grad[up_enum::kOut]};
    } else {
      return {out_grad[up_enum::kOut], in_data[up_enum::kData], in_data[up_enum::kWeight]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.sample_type == up_enum::kNearest) {
      return {};
    } else {
      return {ResourceRequest::kTempSpace};
    }
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.sample_type == up_enum::kNearest) {
      return {};
    } else {
      return {ResourceRequest::kTempSpace};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;


 private:
  UpSamplingParam param_;
};  // class UpSamplingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_UPSAMPLING_INL_H_

