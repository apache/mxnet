
/*!
 * Copyright (c) 2017 by Contributors
 * \file crelu-inl.h
 * \brief concat relu
 * \author Yijie Zhuang
*/

#ifndef MXNET_OPERATOR_CONTRIB_CRELU_INL_H_
#define MXNET_OPERATOR_CONTRIB_CRELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../channel_op_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace concat_relu {
enum CReluOpInputs {kData};
enum CReluOpOutputs {kOut};
}

struct CReluParam : public dmlc::Parameter<CReluParam> {
  DMLC_DECLARE_PARAMETER(CReluParam) {}
};

template<typename xpu, typename DType>
class CReluOp : public Operator {
 public:
  explicit CReluOp(CReluParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2, DType> data = in_data[concat_relu::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[concat_relu::kOut].FlatTo2D<xpu, DType>(s);
    std::vector<Tensor<xpu, 2, DType> > concat_data(2);

    concat_data[0] = NewTensor<xpu, DType>(data.shape_, DType(0));
    concat_data[1] = NewTensor<xpu, DType>(data.shape_, DType(0));
    Copy(concat_data[0], data);
    Copy(concat_data[1], data);
    concat_data[0] = F<mshadow_op::relu>(concat_data[0]);
    concat_data[1] = F<mshadow_op::relu>(F<mshadow_op::negation>(concat_data[1]));

    Concatenate(concat_data, &out, 1, req[concat_relu::kOut]);
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    using namespace std;

    Tensor<xpu, 2, DType> m_in_grad = in_grad[concat_relu::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_data = in_data[concat_relu::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_grad = out_grad[concat_relu::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data[concat_relu::kOut].FlatTo2D<xpu, DType>(s);

    std::vector<Tensor<xpu, 2, DType> > concat_in_grad(2);
    concat_in_grad[0] = NewTensor<xpu, DType>(m_in_grad.shape_, DType(0));
    concat_in_grad[1] = NewTensor<xpu, DType>(m_in_grad.shape_, DType(0));

    m_out_grad = F<mshadow_op::relu_grad>(m_out_data) * m_out_grad;

    concat_in_grad[0] = slice<1>(m_out_grad, 0, m_out_grad.size(1)/2)
                        * F<mshadow_op::sign>(m_in_data);
    concat_in_grad[1] = slice<1>(m_out_grad, m_out_grad.size(1)/2, m_out_grad.size(1))
                        * F<mshadow_op::sign>(m_in_data);

    Assign(m_in_grad, req[concat_relu::kData], concat_in_grad[0] + concat_in_grad[1]);
  }

 private:
    CReluParam param_;
};  // class crelu

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CReluParam param, int dtype);

#if DMLC_USE_CXX11
class CReluProp : public OperatorProperty {
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
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    TShape dshape = in_shape->at(concat_relu::kData);
    if (dshape.ndim() == 0) return false;
    dshape[dshape.ndim()-1] = dshape[dshape.ndim()-1] * 2;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
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
    auto ptr = new CReluProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_CRelu";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[concat_relu::kOut], out_data[concat_relu::kOut], in_data[concat_relu::kData]};
#else
    return {out_grad[concat_relu::kOut], out_data[concat_relu::kOut], in_data[concat_relu::kData]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[concat_relu::kOut], in_grad[concat_relu::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[concat_relu::kData], out_data[concat_relu::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CReluParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_CRELU_INL_H_
