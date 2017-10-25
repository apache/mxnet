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
 * \file pooling_mask_max_2d-inl.h
 * \brief
 * \author Pengfei Li
*/

#ifndef MXNET_OPERATOR_POOLING_MASK_MAX_2D_INL_H_
#define MXNET_OPERATOR_POOLING_MASK_MAX_2D_INL_H_

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

namespace pooling_mask_enum {
enum PoolingMaskOpInputs {kData};
enum PoolingMaskOpOutputs {kOut, kMask};
enum PoolingOpPadConventionType {kValid, kFull};
}  // namespace pooling_mask_enum

struct PoolingMaskParam : public dmlc::Parameter<PoolingMaskParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pooling_convention;
  DMLC_DECLARE_PARAMETER(PoolingMaskParam) {
    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for pooling: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(pooling_mask_enum::kValid)
    .add_enum("full", pooling_mask_enum::kFull)
    .add_enum("valid", pooling_mask_enum::kValid)
    .describe("Pooling convention to be applied.");
  }
};

template<typename xpu, typename DType>
class PoolingMaskOp : public Operator {
 public:
  explicit PoolingMaskOp(PoolingMaskParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<TBlob>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& out_data,
                       const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    pool_mask_forward(s, in_data[pooling_mask_enum::kData].dptr<DType>(), 
                      out_data[pooling_mask_enum::kOut].dptr<DType>(), 
                      out_data[pooling_mask_enum::kMask].dptr<int>(),
                      in_data[pooling_mask_enum::kData].shape_,
                      out_data[pooling_mask_enum::kOut].shape_,
                      param_.kernel, param_.stride, param_.pad);
  }

  virtual void Backward(const OpContext& ctx,
                        const std::vector<TBlob>& out_grad,
                        const std::vector<TBlob>& in_data,
                        const std::vector<TBlob>& out_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& in_grad,
                        const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 2U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad_in = in_grad[pooling_mask_enum::kData].get<xpu, 4, DType>(s);
    if (kWriteTo == req[pooling_mask_enum::kData]) {
        grad_in = 0.0f;
    }
    pool_mask_backward(s, in_grad[pooling_mask_enum::kData].dptr<DType>(),
                       out_grad[pooling_mask_enum::kData].dptr<DType>(),
                       out_data[pooling_mask_enum::kMask].dptr<int>(),
                       in_grad[pooling_mask_enum::kData].shape_,
                       out_grad[pooling_mask_enum::kOut].shape_,
                       param_.kernel, param_.stride, param_.pad);
  }
 private:
  PoolingMaskParam param_; 
  void pool_mask_forward(mshadow::Stream<cpu>* s, 
                        const DType* in_data, DType* out_data, int* mask,
                        const TShape& ishape, const TShape& oshape,
                        const TShape& kernel, const TShape& stride, const TShape& pad);

  void pool_mask_backward(mshadow::Stream<cpu>* s, 
                          DType* in_grad, const DType* out_grad, const int* mask,
                          const TShape& ishape, const TShape& oshape,
                          const TShape& kernel, const TShape& stride, const TShape& pad);

#if MXNET_USE_CUDA
  void pool_mask_forward(mshadow::Stream<gpu>* s, 
                        const DType* in_data, DType* out_data, int* mask,
                        const TShape& ishape, const TShape& oshape,
                        const TShape& kernel, const TShape& stride, const TShape& pad);
  void pool_mask_backward(mshadow::Stream<gpu>* s, 
                          DType* in_grad, const DType* out_grad, const int* mask,
                          const TShape& ishape, const TShape& oshape,
                          const TShape& kernel, const TShape& stride, const TShape& pad);
#endif  // MXNET_USE_CUDA  
};  // class PoolingMaskOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(PoolingMaskParam param, int dtype);

#if DMLC_USE_CXX11
class PoolingMaskProp : public OperatorProperty {
 public:

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 1) {
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D pooling not supported";
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
    CHECK_EQ(param_.stride.ndim(), param_.kernel.ndim())
      << "stride and kernel should have the same length";
    CHECK_EQ(param_.pad.ndim(), param_.kernel.ndim())
      << "pad and kernel should have the same length";
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
    return 2;
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 3U) << "Pooling: Input data should be  3D in (batch, channel, x)"
                                << " Or 4D in (batch, channel, y, x) "
                                << " Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
      
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
          << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
          << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
      CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
          << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
          << " padded to " << (dshape[3] + 2*param_.pad[1]) << ")";
      if (param_.pooling_convention == pooling_mask_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                            param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                            param_.stride[1];
      } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                            dshape[2] + 2 * param_.pad[0] -
                            param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                            dshape[3] + 2 * param_.pad[1] -
                            param_.kernel[1]) / param_.stride[1]));
      }
      
      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
      out_shape->push_back(oshape);  // save output shape
    } else {
      LOG(FATAL) << "Not Implemented.";
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];

    if (dtype == -1) {
      LOG(FATAL) << "Input type to pooling is not specified.";
      return false;
    }

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(mshadow::TypeFlag::kInt32);
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingMaskProp *prop_sym = new PoolingMaskProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "PoolingMask";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pooling_mask_enum::kOut], in_data[pooling_mask_enum::kData],
            out_data[pooling_mask_enum::kOut], out_data[pooling_mask_enum::kMask]};
  }
  /*
  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[pooling_mask_enum::kData], in_grad[pooling_mask_enum::kData]}};
  }
  */
  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  PoolingMaskParam param_;    
};  // class PoolingMaskProp
#endif
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_