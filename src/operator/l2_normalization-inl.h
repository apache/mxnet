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
 * Copyright (c) 2016 by Contributors
 * \file l2_normalization_op-inl.h
 * \brief instance l2 Normalization op
*/
#ifndef MXNET_OPERATOR_L2_NORMALIZATION_INL_H_
#define MXNET_OPERATOR_L2_NORMALIZATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace l2_normalization {
enum L2NormalizationOpInputs {kData};
enum L2NormalizationOpOutputs {kOut, kNorm};
enum L2NormalizationOpType {kInstance, kChannel, kSpatial};
enum L2NormalizationBackResource {kTempSpace};
}  // l2_normalization

struct L2NormalizationParam : public dmlc::Parameter<L2NormalizationParam> {
  float eps;
  int mode;
  DMLC_DECLARE_PARAMETER(L2NormalizationParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("instance", l2_normalization::kInstance)
    .add_enum("spatial", l2_normalization::kSpatial)
    .add_enum("channel", l2_normalization::kChannel)
    .set_default(l2_normalization::kInstance)
    .describe("Specify the dimension along which to compute L2 norm.");
  }
};

/**
 * \brief This is the implementation of l2 normalization operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class L2NormalizationOp : public Operator {
 public:
  explicit L2NormalizationOp(L2NormalizationParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[l2_normalization::kOut] == kNullOp) return;
    CHECK_EQ(req[l2_normalization::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    mxnet::TShape orig_shape = in_data[l2_normalization::kData].shape_;
    if (param_.mode == l2_normalization::kInstance) {
      Shape<2> dshape = Shape2(orig_shape[0],
        orig_shape.ProdShape(1, orig_shape.ndim()));
      Tensor<xpu, 2, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 1, DType> norm = out_data[l2_normalization::kNorm].get<xpu, 1, DType>(s);
      norm = sumall_except_dim<0>(F<mxnet::op::mshadow_op::square>(data));
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::plus, Req>, xpu>::Launch(
          s, norm.size(0), norm.dptr_, norm.dptr_, DType(param_.eps));
      });
      norm = F<mxnet::op::mshadow_op::square_root>(norm);
      out = data / broadcast<0>(norm, out.shape_);
    } else if (param_.mode == l2_normalization::kChannel) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[2]);
      Tensor<xpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<xpu, 2, DType>(norm_shape, s);
      norm = reduce_with_axis<red::sum, false>(F<mxnet::op::mshadow_op::square>(data), 1);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::plus, Req>, xpu>::Launch(
          s, norm.size(0) * norm.size(1), norm.dptr_, norm.dptr_, DType(param_.eps));
      });
      norm = F<mxnet::op::mshadow_op::square_root>(norm);
      out = data / broadcast_with_axis(norm, 0, orig_shape[1]);
    } else if (param_.mode == l2_normalization::kSpatial) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = in_data[l2_normalization::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> out = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[1]);
      Tensor<xpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<xpu, 2, DType>(norm_shape, s);
      norm = reduce_with_axis<red::sum, false>(F<mxnet::op::mshadow_op::square>(data), 2);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mxnet::op::mshadow_op::plus, Req>, xpu>::Launch(
          s, norm.size(0) * norm.size(1), norm.dptr_, norm.dptr_, DType(param_.eps));
      });
      norm = F<mxnet::op::mshadow_op::square_root>(norm);
      out = data / broadcast_with_axis(norm, 1, dshape[2]);
    } else {
      LOG(FATAL) << "Unexpected mode in l2 normalization";
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == 1U && in_grad.size() == 1U);
    CHECK_EQ(req.size(), 1U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    mxnet::TShape orig_shape = out_data[l2_normalization::kOut].shape_;
    if (param_.mode == l2_normalization::kInstance) {
      Shape<2> dshape = Shape2(orig_shape[0],
        orig_shape.ProdShape(1, orig_shape.ndim()));
      Tensor<xpu, 2, DType> data = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> grad_in = in_grad[l2_normalization::kData]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 2, DType> grad_out = out_grad[l2_normalization::kOut]
        .get_with_shape<xpu, 2, DType>(dshape, s);
      Tensor<xpu, 1, DType> norm = out_data[l2_normalization::kNorm].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> temp = ctx.requested[l2_normalization::kTempSpace]
        .get_space_typed<xpu, 1, DType>(mshadow::Shape1(data.shape_[0]), s);
      temp = sumall_except_dim<0>(grad_out * data);
      Assign(grad_in, req[l2_normalization::kData],
        (grad_out - data * broadcast<0>(temp, data.shape_)) /
        broadcast<0>(norm, data.shape_));
    } else if (param_.mode == l2_normalization::kChannel) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_in = in_grad[l2_normalization::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_out = out_grad[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[2]);
      Tensor<xpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<xpu, 2, DType>(norm_shape, s);
      Tensor<xpu, 2, DType> temp = ctx.requested[l2_normalization::kTempSpace]
        .get_space_typed<xpu, 2, DType>(mshadow::Shape2(data.shape_[0], data.shape_[2]), s);
      temp = reduce_with_axis<red::sum, false>(grad_out * data, 1);
      Assign(grad_in, req[l2_normalization::kData],
        (grad_out - data * broadcast_with_axis(temp, 0, orig_shape[1])) /
        broadcast_with_axis(norm, 0, orig_shape[1]));
    } else if (param_.mode == l2_normalization::kSpatial) {
      CHECK_GE(orig_shape.ndim(), 3);
      Shape<3> dshape = Shape3(orig_shape[0], orig_shape[1],
        orig_shape.ProdShape(2, orig_shape.ndim()));
      Tensor<xpu, 3, DType> data = out_data[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_in = in_grad[l2_normalization::kData]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Tensor<xpu, 3, DType> grad_out = out_grad[l2_normalization::kOut]
        .get_with_shape<xpu, 3, DType>(dshape, s);
      Shape<2> norm_shape = Shape2(dshape[0], dshape[1]);
      Tensor<xpu, 2, DType> norm = out_data[l2_normalization::kNorm]
        .get_with_shape<xpu, 2, DType>(norm_shape, s);
      Tensor<xpu, 2, DType> temp = ctx.requested[l2_normalization::kTempSpace]
        .get_space_typed<xpu, 2, DType>(mshadow::Shape2(data.shape_[0], data.shape_[1]), s);
      temp = reduce_with_axis<red::sum, false>(grad_out * data, 2);
      Assign(grad_in, req[l2_normalization::kData],
        (grad_out - data * broadcast_with_axis(temp, 1, dshape[2])) /
        broadcast_with_axis(norm, 1, dshape[2]));
    } else {
      LOG(FATAL) << "Unexpected mode in l2 normalization";
    }
  }

 protected:
  L2NormalizationParam param_;
};  // class L2NormalizationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(L2NormalizationParam param, int dtype);

#if DMLC_USE_CXX11
class L2NormalizationProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "norm"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = (*in_type)[0];
    type_assign(&dtype, (*out_type)[0]);
    type_assign(&dtype, (*out_type)[1]);

    TYPE_ASSIGN_CHECK(*in_type, 0, dtype);
    TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
    TYPE_ASSIGN_CHECK(*out_type, 1, dtype);
    return dtype != -1;
  }

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "L2Normalization layer only accepts data as input";
    const mxnet::TShape &dshape = (*in_shape)[l2_normalization::kData];
    // require data to be known
    if ((*in_shape)[l2_normalization::kData].ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    if (param_.mode == l2_normalization::kInstance) {
      out_shape->push_back(Shape1(dshape[0]));
    } else if (param_.mode == l2_normalization::kChannel) {
      CHECK_GE(dshape.ndim(), 3) << "At lease 3 dimensions required in channel mode";
      mxnet::TShape norm_shape = dshape;
      norm_shape[1] = 1;
      out_shape->push_back(norm_shape);
    } else if (param_.mode == l2_normalization::kSpatial) {
      CHECK_GE(dshape.ndim(), 3) << "At lease 3 dimensions required in spatial mode";
      out_shape->push_back(Shape2(dshape[0], dshape[1]));
    } else {
      return false;
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    L2NormalizationProp* norm_sym = new L2NormalizationProp();
    norm_sym->param_ = this->param_;
    return norm_sym;
  }

  std::string TypeString() const override {
    return "L2Normalization";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[l2_normalization::kOut],
      out_data[l2_normalization::kOut],
      out_data[l2_normalization::kNorm]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[l2_normalization::kOut], in_grad[l2_normalization::kData]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  L2NormalizationParam param_;
};  // class L2NormalizationSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_L2_NORMALIZATION_INL_H_
