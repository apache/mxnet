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

namespace pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};
}  // namespace pool_enum

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  bool global_pool;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. "
              "This is useful for input with different shape");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .describe("pad for pooling: (y, x) or (d, y, x)");
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
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "Not implmented";
    }
    Tensor<xpu, 4> data = in_data[pool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[pool_enum::kOut].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) {
      Assign(out,
             req[pool_enum::kOut],
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
    } else if (param_.pool_type == pool_enum::kAvgPooling) {
      Assign(out,
             req[pool_enum::kOut],
             (1.0f / (param_.global_pool ?
                      data.shape_[2] * data.shape_[3] :
                      param_.kernel[0] * param_.kernel[1])) * \
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
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
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    // TODO(bing): remove pad (0,0)
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "Not implmented";
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[pool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[pool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> output_data = out_data[pool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[pool_enum::kData].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]);

    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) {
      Assign(input_grad, req[pool_enum::kData],
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
    } else if (param_.pool_type == pool_enum::kAvgPooling) {
      Assign(input_grad, req[pool_enum::kData],
             (1.0f / param_.kernel[0] / param_.kernel[1]) *\
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
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
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 4) << "Pooling: Input data should be 4D in (batch, channel, y, x) "
                               << "Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
      } else {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) / param_.stride[0];
        oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) / param_.stride[1];
      }
      CHECK(oshape[2] > 0 && oshape[3] > 0) << "Pooling: kernel size exceed input";
      out_shape->clear();
      out_shape->push_back(oshape);
    } else if (param_.kernel.ndim() == 3) {
      CHECK_EQ(dshape.ndim(), 5) << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
        oshape[4] = 1;
      } else {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) / param_.stride[0];
        oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) / param_.stride[1];
        oshape[4] = 1 + (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) / param_.stride[2];
      }
      CHECK(oshape[2] > 0 && oshape[3] > 0 && oshape[4] > 0) << "Pooling: kernel size exceed input";
      out_shape->clear();
      out_shape->push_back(oshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingProp *prop_sym = new PoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Pooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData], out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[pool_enum::kData], in_grad[pool_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  PoolingParam param_;
};  // class PoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_

