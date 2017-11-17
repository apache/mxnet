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
 * \file Ifft-inl.h
 * \brief
 * \author Chen Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_IFFT_INL_H_
#define MXNET_OPERATOR_CONTRIB_IFFT_INL_H_
#include <stdio.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

#if MXNET_USE_CUDA
#include <cufft.h>
#endif

namespace mxnet {
namespace op {
namespace ifft {
  enum ifftOpInputs {kData};  // input should represent complex
  enum ifftOpOutputs {kOut};  // output should be real
  enum ifftOpResource {kTempSpace};
}

struct IFFTParam : public dmlc::Parameter<IFFTParam> {
  int compute_size;  // the maximum size of sub-batch to be forwarded through cufft in one time
  DMLC_DECLARE_PARAMETER(IFFTParam){
    DMLC_DECLARE_FIELD(compute_size).set_default(128)
    .describe("Maximum size of sub-batch to be forwarded at one time");
  }
};

#if MXNET_USE_CUDA
template<typename xpu, typename DType>
class IFFTOp : public Operator {
 public:
  explicit IFFTOp(IFFTParam p) {
    this->param_ = p;
    init_cufft_ = false;
    dim_ = 0;
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

    if (!init_cufft_) {
      n_iffts = in_data[ifft::kData].shape_.ProdShape(0, in_data[ifft::kData].ndim()-1);
      // remember that input is complex
      dim_ = in_data[ifft::kData].shape_[in_data[ifft::kData].ndim()-1]/2;
      // stride_ in the number of complex numbers
      stride_ = param_.compute_size*dim_;

      init_cufft_ = true;

      num_compute = n_iffts/param_.compute_size;
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[ifft::kData].shape_;
    const TShape& oshape = out_data[ifft::kOut].shape_;
    Tensor<xpu, 2, DType> data = in_data[ifft::kData].get_with_shape<xpu, 2, DType>(
          Shape2(n_iffts, dim_*2), s);
    Tensor<xpu, 2, DType> out = out_data[ifft::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(n_iffts, dim_), s);
    // need temp space to store the intermediate complex matrices
    Tensor<xpu, 1, DType> workspace =
            ctx.requested[ifft::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(param_.compute_size*dim_*2), s);
    Tensor<xpu, 2, DType> complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(param_.compute_size, dim_*2), s);
    // start ifft
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param_.compute_size);
    for (size_t idx=0; idx < num_compute; ++idx) {
      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(data.dptr_ + 2*idx*stride_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(complex_data.dptr_);
      CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);

      Assign(out.Slice(idx*param_.compute_size, (idx+1)*param_.compute_size),
             req[ifft::kOut], complex_toreal(complex_data));
    }
    cufftDestroy(plan);
    // handle the remaining samples
    size_t remain_num = n_iffts - param_.compute_size*num_compute;
    if (remain_num > 0) {
      cufftHandle plan_remain;
      cufftPlanMany(&plan_remain, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0,
                    CUFFT_C2C, remain_num);

      complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(remain_num, dim_*2), s);

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(data.dptr_ + 2*num_compute*stride_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(complex_data.dptr_);
      CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);
        Assign(out.Slice(param_.compute_size*num_compute,
                         param_.compute_size*num_compute+remain_num),
             req[ifft::kOut], complex_toreal(complex_data));
      cufftDestroy(plan_remain);
    }
    // commenting this out to be consistant with caffe
    // out /= dim_;
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
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    const TShape& ishape = in_grad[ifft::kData].shape_;
    const TShape& oshape = out_grad[ifft::kOut].shape_;
    Tensor<xpu, 2, DType> gdata = in_grad[ifft::kData].get_with_shape<xpu, 2, DType>(
          Shape2(n_iffts, dim_*2), s);
    Tensor<xpu, 2, DType> grad = out_grad[ifft::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(n_iffts, dim_), s);
    // need temp space to pad the data into complex numbers due to cufft interface
    Tensor<xpu, 1, DType> workspace =
            ctx.requested[ifft::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(param_.compute_size*dim_*2), s);
    Tensor<xpu, 2, DType> complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(param_.compute_size, dim_*2), s);
    // start fft
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param_.compute_size);
    for (size_t idx = 0; idx < num_compute; ++idx) {
      complex_data = complex_pad_imag(grad.Slice(idx*param_.compute_size,
                                                 idx*param_.compute_size+param_.compute_size));

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(complex_data.dptr_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(gdata.dptr_ + 2*idx*stride_);
      CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
    }
    cufftDestroy(plan);

    // handle the remaining samples
    size_t remain_num = n_iffts - param_.compute_size*num_compute;
    if (remain_num > 0) {
      cufftHandle plan_remain;
      cufftPlanMany(&plan_remain, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0,
                    CUFFT_C2C, remain_num);
      complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                          Shape2(remain_num, dim_*2), s);
      complex_data = complex_pad_imag(grad.Slice(
          num_compute*param_.compute_size, num_compute*param_.compute_size+remain_num));

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(complex_data.dptr_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(gdata.dptr_ + 2*num_compute*stride_);
      CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
      cufftDestroy(plan_remain);
    }
    // commenting this out to be consistant with caffe
    // gdata /= dim_;
  }

 private:
  IFFTParam param_;
  int dim_, stride_, n_iffts;
  size_t num_compute;
  bool init_cufft_;
};  // class IFFTOp

#endif  // MXNET_USE_CUDA

// Declare Factory Function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(IFFTParam param, int dtype);

#if DMLC_USE_CXX11
class IFFTProp : public OperatorProperty {
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
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) <<"Input:[data]";
    const TShape &dshape = (*in_shape)[ifft::kData];
    // require data to be known
    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    if (dshape.ndim() == 4) {
      out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]/2));
    } else if (dshape.ndim() == 2) {
      out_shape->push_back(Shape2(dshape[0], dshape[1]/2));
    } else {
      return false;
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i=0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    IFFTProp* ifft_sym = new IFFTProp();
    ifft_sym->param_ = this->param_;
    return ifft_sym;
  }

  std::string TypeString() const override {
    return "_contrib_ifft";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[ifft::kOut], in_data[ifft::kData]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[ifft::kData], in_grad[ifft::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  IFFTParam param_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_IFFT_INL_H_
