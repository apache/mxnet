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
 * Copyright (c) 2017 by Contributors
 * \file bilinear_Sampler-inl.h
 * \brief
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_BILINEAR_SAMPLER_INL_H_
#define MXNET_OPERATOR_BILINEAR_SAMPLER_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace bs {
enum BilinearSamplerOpInputs {kData, kGrid};
enum BilinearSamplerOpOutputs {kOut, kTmp};
}

struct BilinearSamplerParam : public dmlc::Parameter<BilinearSamplerParam> {
  dmlc::optional<bool> cudnn_off;
  DMLC_DECLARE_PARAMETER(BilinearSamplerParam) {
    DMLC_DECLARE_FIELD(cudnn_off).set_default(dmlc::optional<bool>())
        .describe("whether to turn cudnn off");
  }
};

template<typename xpu, typename DType>
class BilinearSamplerOp : public Operator {
 public:
  explicit BilinearSamplerOp(BilinearSamplerParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[bs::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[bs::kOut].get<xpu, 4, DType>(s);

    BilinearSamplerForward(out, data, grid);
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
    CHECK_EQ(in_data.size(), 2U);
    CHECK_NE(req[bs::kData], kWriteInplace);
    CHECK_NE(req[bs::kGrid], kWriteInplace);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[bs::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> ggrid = in_grad[bs::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[bs::kOut].get<xpu, 4, DType>(s);
    if (req[bs::kData] == kNullOp && req[bs::kGrid] == kNullOp) {
      return;
    } else {
      if (req[bs::kData] == kWriteTo) {
        gdata = scalar<DType>(0.0f);
      }
      if (req[bs::kGrid] == kWriteTo) {
        ggrid = scalar<DType>(0.0f);
      }
      BilinearSamplerBackward(gdata, ggrid, grad, data, grid, req[bs::kData], req[bs::kGrid]);
    }
  }

 private:
  BilinearSamplerParam param_;
};  // class BilinearSamplerOp

template<typename xpu>
Operator* CreateOp(BilinearSamplerParam param, int dtype);

#if DMLC_USE_CXX11
class BilinearSamplerProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "grid"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp"};
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
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, grid]";
    const mxnet::TShape &dshape = (*in_shape)[bs::kData];
    const mxnet::TShape &lshape = (*in_shape)[bs::kGrid];
    if (!shape_is_known(dshape)) return false;
    CHECK_EQ(dshape.ndim(), 4U) \
        << "input data should be 4D in batch-num_filter-y-x";
    if (!shape_is_known(lshape)) return false;
    CHECK_EQ(lshape.ndim(), 4U) \
      << "Sampler grid should be 4D in batch-2-y-x";
    CHECK_EQ(dshape[0], lshape[0]);
    CHECK_EQ(lshape[1], 2U) << "incorrect grid shape[1], should be 2";
    // target height
    CHECK_GT(lshape[2], 0U) \
            << "incorrect grid_shape: " << lshape[2];
    // target width
    CHECK_GT(lshape[3], 0U) \
        << "incorrect grid_shape: " << lshape[3];
    out_shape->clear();
    // output_shape : (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3])
    out_shape->push_back(dshape);
    (*out_shape)[bs::kOut][2] = lshape[2];
    (*out_shape)[bs::kOut][3] = lshape[3];
    out_shape->push_back(Shape4(lshape[0], lshape[2], lshape[3], 2));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      int dtype = -1;
      for (int type : *in_type) {
        if (dtype == -1) {
          dtype = type;
        } else {
          CHECK(type == dtype ||
              type == -1) <<
                "Non-uniform data type in BilinearSampler";
        }
      }
      if (dtype == -1) {
        LOG(FATAL) << "Not enough information to infer type in BilinearSampler.";
        return false;
      }
      size_t nin = this->ListArguments().size();
      in_type->clear();
      for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);
      size_t naux = this->ListAuxiliaryStates().size();
      aux_type->clear();
      for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);
      size_t nout = this->ListOutputs().size();
      out_type->clear();
      for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
      return true;
    }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearSamplerProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BilinearSampler";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[bs::kOut],
            in_data[bs::kData],
            out_data[bs::kTmp],
            in_data[bs::kGrid]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BilinearSamplerParam param_;
};  // class BilinearSamplerProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BILINEAR_SAMPLER_INL_H_
