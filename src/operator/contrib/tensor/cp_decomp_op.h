/*!
 *  Copyright (c) 2016 by Contributors
 * \file cp_decomp_op.h
 * \brief Define operator for CP Decomposition
 * \author Jencir Lee
 */

#ifndef MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_OP_H_
#define MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_OP_H_

#include <mxnet/operator_util.h>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "../../mshadow_op.h"
#include "../../tensor/init_op.h"
#include "./cp_decomp.h"

namespace mxnet {
namespace op {

struct CPDecompParam : public dmlc::Parameter<CPDecompParam> {
  int k;

  DMLC_DECLARE_PARAMETER(CPDecompParam) {
    DMLC_DECLARE_FIELD(k).set_default(10)
    .describe("Rank of the CP Decomposition");
  }
};

template<typename xpu, int order, typename DType>
class CPDecompOp : public Operator {
 public:
  explicit CPDecompOp(CPDecompParam param) {
    this->k = param.k;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mxnet::op;
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), order + 1);

    Stream<xpu> *stream = ctx.get_stream<xpu>();

    // Input tensor
    const Tensor<xpu, order, DType> &ts = in_data[0].get<xpu, order, DType>
      (stream);

    // Output eigen-values vector
    Tensor<xpu, 1, DType> eigvals = out_data[0].FlatTo1D<xpu, DType>(stream);

    // Vector of output factor matrices, each one transposed
    std::vector<Tensor<xpu, 2, DType> > factors_T;
    for (int i = 1; i < static_cast<int>(out_data.size()); ++i)
      factors_T.push_back(out_data[i].FlatTo2D<xpu, DType>(stream));

    CPDecomp(eigvals, factors_T, ts, k);
  }

 private:
  int k;
};  // class CPDecompOp

#if DMLC_USE_CXX11
template <int order>
class CPDecompProp : public OperatorProperty {
 public:
  void Init
    (const std::vector<std::pair<std::string, std::string> >& kwargs)
    override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);

    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() != order) return false;

    out_shape->clear();
    out_shape->push_back(TShape(Shape1(param_.k)));
    for (int i = 0; i < order; ++i)
      out_shape->push_back(TShape(Shape2(param_.k, dshape[i])));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = in_type->at(0);

    if (dtype == -1) {
      LOG(FATAL) << "input type to dropout is not specified.";
      return false;
    }

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CPDecompProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "CPDecomp" + std::to_string(order) + "D";
  }

  /*
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout::kOut], out_data[dropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout::kOut], in_grad[dropout::kData]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom};
  }
  */

  int NumVisibleOutputs() const override {
    return order + 1;
  }

  int NumOutputs() const override {
    return order + 1;
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> outputs = {"eigvals"};
    for (int i = 0; i < order; ++i)
      outputs.push_back("factor_T_" + std::to_string(i));
    return outputs;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CPDecompParam param_;
};  // class DropoutProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_OP_H_
