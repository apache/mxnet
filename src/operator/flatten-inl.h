/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_FLATTEN_INL_H_
#define MXNET_OPERATOR_FLATTEN_INL_H_

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

enum FlattenOpInputs {kData};
enum FlattenOpOutputs {kOut};

template<typename xpu>
class FlattenOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[kOut].get<xpu, 4, real_t>(s);
    Assign(out, req[kOut], reshape(data, out.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad_out = out_grad[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_in = in_grad[kOut].get<xpu, 4, real_t>(s);
    Assign(grad_in, req[kData], reshape(grad_out, grad_in.shape_));
  }
};  // class FlattenOp

template<typename xpu>
Operator* CreateOp();

#if DMLC_USE_CXX11
class FlattenProp : public OperatorProperty {
 public:
  FlattenProp() {}

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {}

  virtual std::string TypeString() const {
    return "Flatten";
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(mshadow::Shape4(dshape[0], 1, 1, dshape[1] * dshape[2] * dshape[3]));
    return true;
  }

  virtual OperatorProperty* Copy() const {
    auto ptr = new FlattenProp();
    return ptr;
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[kOut]};
  }

  Operator* CreateOperator(Context ctx) const;
};  // class FlattenProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FLATTEN_INL_H_
