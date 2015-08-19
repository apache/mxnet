/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_POOLING_INL_H_
#define MXNET_OPERATOR_POOLING_INL_H_

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
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  int kernel_x;
  int kernel_y;
  int stride_x;
  int stride_y;
  int pad_x;
  int pad_y;
  int type;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    // TODO(bing) change to only set lower bound
    DMLC_DECLARE_FIELD(kernel_x).set_range(1, 10000);
    DMLC_DECLARE_FIELD(kernel_y).set_range(1, 10000);
    DMLC_DECLARE_FIELD(stride_x).set_range(1, 10000);
    DMLC_DECLARE_FIELD(stride_y).set_range(1, 10000);
    DMLC_DECLARE_FIELD(pad_x).set_default(0).set_range(0, 10000);
    DMLC_DECLARE_FIELD(pad_y).set_default(0).set_range(0, 10000);
    DMLC_DECLARE_FIELD(type).set_default(kMaxPooling)\
      .add_enum("max", kMaxPooling).add_enum("avg", kAvgPooling)\
      .add_enum("sum", kSumPooling);
  }
};

template<typename xpu, typename Reducer>
class PoolingOp : public Operator {
 public:
  explicit PoolingOp(PoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[kOut].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    // TODO(bing): dual stride in mshadow
    if (param_.type == kMaxPooling || param_.type == kSumPooling) {
      out = pool<Reducer>(pad(data, param_.pad_y, param_.pad_x),
                          out_shape,
                          param_.kernel_y,
                          param_.kernel_x,
                          param_.kernel_y);
    } else if (param_.type == kAvgPooling) {
      out = (1.0f / (param_.kernel_y * param_.kernel_x)) * \
            pool<Reducer>(pad(data, param_.pad_y, param_.pad_x),
                          out_shape,
                          param_.kernel_y,
                          param_.kernel_x,
                          param_.kernel_y);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    // TODO(bing): remove pad (0,0)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> output_data = out_data[kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[kData].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]);

    if (param_.type == kMaxPooling || param_.type == kSumPooling) {
      Assign(input_grad, req[kData],
             crop(unpool<Reducer>(pad(data, param_.pad_y, param_.pad_x),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.kernel_y,
                                  param_.kernel_x,
                                  param_.stride_y),
                  in_shape,
                  param_.pad_y,
                  param_.pad_x));
    } else if (param_.type == kAvgPooling) {
      Assign(input_grad, req[kData],
             (1.0f / param_.kernel_y / param_.kernel_x) *\
             crop(unpool<Reducer>(pad(data, param_.pad_y, param_.pad_x),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.kernel_y,
                                  param_.kernel_x,
                                  param_.stride_y),
                  in_shape,
                  param_.pad_y,
                  param_.pad_x));
    }
  }

 private:
  PoolingParam param_;
};  // class PoolingOp

template<typename xpu>
Operator* CreateOp(PoolingParam param);


#if DMLC_USE_CXX11
class PoolingProp : public OperatorProperty {
 public:
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.Init(kwargs);
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
      "Pooling: Input data should be 4D in (batch, channel, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    oshape[2] = std::min(dshape[2] + 2 * param_.pad_y - param_.kernel_y + param_.stride_y - 1,
                         dshape[2] + 2 * param_.pad_y - 1) / param_.stride_y + 1;
    oshape[3] = std::min(dshape[3] + 2 * param_.pad_x - param_.kernel_x + param_.stride_x - 1,
                         dshape[3] + 2 * param_.pad_x - 1) / param_.stride_x + 1;
    CHECK(oshape[2] > 0 && oshape[3] > 0) << "Pooling: kernel size exceed input";
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  virtual OperatorProperty* Copy() const {
    PoolingProp *prop_sym = new PoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  virtual std::string TypeString() const {
    return "Pooling";
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[kOut], in_data[kData], out_data[kOut]};
  }

  virtual std::vector<std::pair<int, int> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<int> &in_grad) const {
    return {{in_data[kData], in_grad[kData]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  PoolingParam param_;
};  // class PoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_
