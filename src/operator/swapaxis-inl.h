/*!
 * Copyright (c) 2015 by Contributors
 * \file swapaxis-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_OPERATOR_SWAPAXIS_INL_H_
#define MXNET_OPERATOR_SWAPAXIS_INL_H_

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

namespace swapaxisenum {
enum SwapAxisOpInputs {kData};
enum SwapAxisOpOutputs {kOut};
};


struct SwapAxisParam : public dmlc::Parameter<SwapAxisParam> {
  // use int for enumeration
  uint32_t dim1, dim2;
  DMLC_DECLARE_PARAMETER(SwapAxisParam) {
    DMLC_DECLARE_FIELD(dim1)
    .set_default(0)
    .describe("the first axis to be swapped.");
    DMLC_DECLARE_FIELD(dim2)
    .set_default(0)
    .describe("the second axis to be swapped.");
  }
};


template<typename xpu, typename DType>
class SwapAxisOp : public Operator {
 public:
  explicit SwapAxisOp(SwapAxisParam p) {
    CHECK_NE(p.dim1, p.dim2) << "dim1 can not be equal dim2.";
    this->param_ = p;
  }

  void Reshape2Five(mshadow::Shape<5> *inter_shape,
                    const mshadow::TShape &shape,
                    uint32_t dim1, uint32_t dim2) {
    using namespace mshadow;
    using namespace mshadow::expr;
    index_t ndim_in = shape.ndim();
    index_t si;

    if (dim1 > dim2) {
      std::swap(dim1, dim2);
    }

    for (si = 0; si < 5; si++) {
      (*inter_shape)[si] = 1;
    }
    // dim_0
    for (si = 0; si < dim1; si++) {
      (*inter_shape)[0] *= shape[si];
    }
    // dim_1
    (*inter_shape)[1] = shape[dim1];
    // dim_2
    for (si = dim1 + 1; si < dim2; si++) {
      (*inter_shape)[2] *= shape[si];
    }
    // dim_3
    (*inter_shape)[3] = shape[dim2];
    // dim_4
    for (si = dim2 + 1; si < ndim_in; si++) {
      (*inter_shape)[4] *= shape[si];
    }
  }

  void SwapAxis(mshadow::Stream<xpu> *s,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    uint32_t dim1 = param_.dim1;
    uint32_t dim2 = param_.dim2;

    TBlob data_in = in_data[swapaxisenum::kData];
    TBlob data_out = out_data[swapaxisenum::kData];

    TShape shape_in = data_in.shape_;
    TShape shape_out = data_out.shape_;

    Shape<5> inter_shape;

    Reshape2Five(&inter_shape, shape_in, dim1, dim2);

    Tensor<xpu, 5, DType> inter_data_in = data_in.get_with_shape<xpu, 5, DType>(inter_shape, s);

    Shape<5> inter_shape2 = inter_shape;
    std::swap(inter_shape2[1], inter_shape2[3]);

    Tensor<xpu, 5, DType> inter_data_out = data_out.get_with_shape<xpu, 5, DType>(inter_shape2, s);

    inter_data_out = swapaxis<3, 1>(inter_data_in);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    SwapAxis(s, in_data, out_data);
  }

  virtual void Backward(const OpContext &ctx,
                       const std::vector<TBlob> &out_grad,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &in_grad,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    SwapAxis(s, out_grad, in_grad);
  }

  SwapAxisParam param_;
};


template<typename xpu>
Operator* CreateOp(SwapAxisParam param, int dtype);


#if DMLC_USE_CXX11
class SwapAxisProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
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
    CHECK_EQ(in_shape->size(), 1);

    TShape &shape0 = (*in_shape)[swapaxisenum::kData];
    out_shape->clear();
    out_shape->push_back(shape0);
    TShape &shape1 = (*out_shape)[swapaxisenum::kOut];

    std::swap(shape1[param_.dim1], shape1[param_.dim2]);

    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "Input must have specified type";
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SwapAxisProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SwapAxis";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[swapaxisenum::kOut]};
  };

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SwapAxisParam param_;
};  // class SwapAxisProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SWAPAXIS_INL_H_


