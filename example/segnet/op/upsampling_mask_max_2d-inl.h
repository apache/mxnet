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
 * \file upsampling_mask_max_2d-inl.h
 * \brief
 * \author Pengfei Li
*/

#ifndef MXNET_OPERATOR_UPSAMPLING_MASK_MAX_2D_INL_H_
#define MXNET_OPERATOR_UPSAMPLING_MASK_MAX_2D_INL_H_

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

namespace upsampling_mask_enum {
enum UpSamplingMaskOpInputs {kData, kMask};
enum UpSamplingMaskOpOutputs {kOut};
}  // namespace upsampling_mask_enum

struct UpSamplingMaskParam : public dmlc::Parameter<UpSamplingMaskParam> {
  TShape out_shape;
  DMLC_DECLARE_PARAMETER(UpSamplingMaskParam) {
    DMLC_DECLARE_FIELD(out_shape)
    .enforce_nonzero()
    .describe("upsampling output size: (y, x)");
  }
};  // struct UpSamplingMaskParam

template<typename xpu, typename DType>
class UpSamplingMaskOp : public Operator {
 public:
  explicit UpSamplingMaskOp(UpSamplingMaskParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data_out = out_data[upsampling_mask_enum::kData].get<xpu, 4, DType>(s);
    data_out = 0.0f;
    upsample_mask_forward(s, in_data[upsampling_mask_enum::kData].dptr<DType>(), 
                      out_data[upsampling_mask_enum::kOut].dptr<DType>(), 
                      in_data[upsampling_mask_enum::kMask].dptr<int>(),
                      in_data[upsampling_mask_enum::kData].shape_,
                      out_data[upsampling_mask_enum::kOut].shape_);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 2U);
    CHECK_EQ(in_grad.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad_in = in_grad[upsampling_mask_enum::kData].get<xpu, 4, DType>(s);
    //if (kWriteTo == req[upsampling_mask_enum::kData]) {
        grad_in = 0.0f;
    //}
    upsample_mask_backward(s, in_grad[upsampling_mask_enum::kData].dptr<DType>(),
                       out_grad[upsampling_mask_enum::kData].dptr<DType>(),
                       in_data[upsampling_mask_enum::kMask].dptr<int>(),
                       in_grad[upsampling_mask_enum::kData].shape_,
                       out_grad[upsampling_mask_enum::kOut].shape_);
                          
  }

private:
  UpSamplingMaskParam param_; 
  void upsample_mask_forward(mshadow::Stream<cpu>* s, 
                        const DType* in_data, DType* out_data, int* mask,
                        const TShape& ishape, const TShape& oshape);

  void upsample_mask_backward(mshadow::Stream<cpu>* s, 
                          DType* in_grad, const DType* out_grad, const int* mask,
                          const TShape& ishape, const TShape& oshape);

#if MXNET_USE_CUDA
  void upsample_mask_forward(mshadow::Stream<gpu>* s, 
                        const DType* in_data, DType* out_data, int* mask,
                        const TShape& ishape, const TShape& oshape);
  void upsample_mask_backward(mshadow::Stream<gpu>* s, 
                          DType* in_grad, const DType* out_grad, const int* mask,
                          const TShape& ishape, const TShape& oshape);
#endif  // MXNET_USE_CUDA  
};  // class UnSamplingMaskOp

template<typename xpu>
Operator *CreateOp(UpSamplingMaskParam param, int dtype);

#if DMLC_USE_CXX11
class UpSamplingMaskProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "mask"};
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 2U);
    CHECK_EQ(param_.out_shape.ndim(), 2U);
    const TShape &dshape = (*in_shape)[0];
    const TShape &dshape1 = (*in_shape)[1];
    CHECK_EQ(dshape, dshape1);
    TShape oshape = dshape;
    oshape[2] = param_.out_shape[0];
    oshape[3] = param_.out_shape[1];
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // for (index_t i = 0; i < in_type->size(); ++i) {
    //   if ((*in_type)[i] == -1) {
    //     (*in_type)[i] = dtype;
    //   } else {
    //     UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
    //   }
    // }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new UpSamplingMaskProp();
    ptr->param_ = this->param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "UpSamplingMask";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[upsampling_mask_enum::kOut], in_data[upsampling_mask_enum::kMask]};
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
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  UpSamplingMaskParam param_;
};  // class UpSamplingMaskProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_UPSAMPLING_INL_H_