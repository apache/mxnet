/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
  int dim1, dim2;
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
                    const mxnet::TShape &shape,
                    int dim1, int dim2) {
    using namespace mshadow;
    using namespace mshadow::expr;
    int ndim_in = shape.ndim();
    int si;

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
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req) {
    using namespace mshadow;
    using namespace mshadow::expr;

    TBlob data_in = in_data[swapaxisenum::kData];
    TBlob data_out = out_data[swapaxisenum::kData];
    OpReqType out_req = req[swapaxisenum::kData];

    mxnet::TShape shape_in = data_in.shape_;
    mxnet::TShape shape_out = data_out.shape_;
    int axis1 = param_.dim1;
    if (axis1 < 0) {
      axis1 += shape_in.ndim();
    }
    CHECK(axis1 >= 0 && axis1 < shape_in.ndim())
        << "axis1: axis " << param_.dim1 << " is out of bounds for array of ndim "
        << shape_in.ndim();

    int axis2 = param_.dim2;
    if (axis2 < 0) {
      axis2 += shape_in.ndim();
    }
    CHECK(axis2 >= 0 && axis2 < shape_in.ndim())
        << "axis2: axis " << param_.dim2 << " is out of bounds for array of ndim "
        << shape_in.ndim();

    if (shape_in.Size() == 0U) return;

    Shape<5> inter_shape;

    Reshape2Five(&inter_shape, shape_in, axis1, axis2);

    Tensor<xpu, 5, DType> inter_data_in = data_in.get_with_shape<xpu, 5, DType>(inter_shape, s);

    Shape<5> inter_shape2 = inter_shape;
    std::swap(inter_shape2[1], inter_shape2[3]);

    Tensor<xpu, 5, DType> inter_data_out = data_out.get_with_shape<xpu, 5, DType>(inter_shape2, s);

    if (out_req == kAddTo) {
        inter_data_out += swapaxis<3, 1>(inter_data_in);
    } else {
        inter_data_out = swapaxis<3, 1>(inter_data_in);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    SwapAxis(s, in_data, out_data, req);
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

    SwapAxis(s, out_grad, in_grad, req);
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

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1U);

    mxnet::TShape &shape0 = (*in_shape)[swapaxisenum::kData];
    if (!ndim_is_known(shape0)) return false;
    int axis1 = param_.dim1;
    if (axis1 < 0) {
      axis1 += shape0.ndim();
    }
    CHECK(axis1 >= 0 && axis1 < shape0.ndim())
        << "axis1: axis " << param_.dim1 << " is out of bounds for array of ndim " << shape0.ndim();

    int axis2 = param_.dim2;
    if (axis2 < 0) {
      axis2 += shape0.ndim();
    }
    CHECK(axis2 >= 0 && axis2 < shape0.ndim())
        << "axis2: axis " << param_.dim2 << " is out of bounds for array of ndim " << shape0.ndim();

    out_shape->clear();
    out_shape->push_back(shape0);
    mxnet::TShape &shape1 = (*out_shape)[swapaxisenum::kOut];

    std::swap(shape1[axis1], shape1[axis2]);

    return shape_is_known(*out_shape);
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
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

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SwapAxisParam param_;
};  // class SwapAxisProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SWAPAXIS_INL_H_
