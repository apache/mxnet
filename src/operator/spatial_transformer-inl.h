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
 * \file spatial_transformer-inl.h
 * \brief
 *  Reproducing paper: aderberg M, Simonyan K, Zisserman A. "Spatial transformer networks"
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_SPATIAL_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_SPATIAL_TRANSFORMER_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./linalg.h"


namespace mxnet {
namespace op {

namespace st {
enum SpatialTransformerOpInputs {kData, kLoc};
enum SpatialTransformerOpOutputs {kOut, kGridDst, kGridSrc};
enum SpatialTransformerOpResource {kTempSpace};
enum SpatialTransformerTransformType {kAffine};
enum SpatialTransformerSamplerType {kBilinear};
}

struct SpatialTransformerParam : public dmlc::Parameter<SpatialTransformerParam> {
  TShape target_shape;
  int transform_type;
  int sampler_type;
  DMLC_DECLARE_PARAMETER(SpatialTransformerParam) {
    int shape[] = {0, 0};
    DMLC_DECLARE_FIELD(target_shape).set_default(TShape(shape, shape + 2))
        .describe("output shape(h, w) of spatial transformer: (y, x)");
    DMLC_DECLARE_FIELD(transform_type).add_enum("affine", st::kAffine)
        .describe("transformation type");
    DMLC_DECLARE_FIELD(sampler_type).add_enum("bilinear", st::kBilinear)
        .describe("sampling type");
  }
};

template<typename xpu, typename DType>
class SpatialTransformerOp : public Operator {
 public:
  explicit SpatialTransformerOp(SpatialTransformerParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[st::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[st::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grid_dst = out_data[st::kGridDst].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> grid_src = out_data[st::kGridSrc].get<xpu, 3, DType>(s);
    Shape<3> loc_shape = Shape3(data.size(0), 2, 3);
    Tensor<xpu, 3, DType> loc = in_data[st::kLoc].get_with_shape<xpu, 3, DType>(loc_shape, s);
    Tensor<cpu, 2, DType> workspace =
          ctx.requested[st::kTempSpace].get_host_space_typed<2, DType>(
          grid_dst.shape_);
    for (index_t i = 1; i <= workspace.size(1); i++) {
      // grid dst coordinate is (x, y, 1)
      workspace[0][i-1] = -1.0 + (i-1) % param_.target_shape[1] * 2.0 /
                          (param_.target_shape[1] - 1);
      workspace[1][i-1] = -1.0 + (i-1) / param_.target_shape[1] * 2.0 /
                          (param_.target_shape[0] - 1);
      workspace[2][i-1] = 1.0;
    }
    Copy(grid_dst, workspace, grid_dst.stream_);
    for (index_t batch = 0; batch < data.size(0); batch++) {
        if (param_.transform_type == st::kAffine) {
          // Legacy approach shown here for comparison:
          //    grid_src[batch] = dot(loc[batch], grid_dst);
          linalg_gemm(loc[batch], grid_dst, grid_src[batch], false, false, s);
        }
    }
    if (param_.sampler_type == st::kBilinear) {
      BilinearSamplingForward(out, data, grid_src);
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
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[st::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[st::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[st::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grid_dst = out_data[st::kGridDst].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> grid_src = out_data[st::kGridSrc].get<xpu, 3, DType>(s);
    Shape<3> loc_shape = Shape3(data.size(0), 2, 3);
    Tensor<xpu, 3, DType> gloc = in_grad[st::kLoc].get_with_shape<xpu, 3, DType>(loc_shape, s);
    gdata = 0.0;
    if (param_.sampler_type == st::kBilinear) {
      BilinearSamplingBackward(gdata, grid_src, grad, data);
    }
    for (index_t batch = 0; batch < data.size(0); batch++) {
        if (param_.transform_type == st::kAffine) {
          // Legacy approach shown here for comparison:
          //   gloc[batch] = dot(grid_src[batch], grid_dst.T());
          linalg_gemm(grid_src[batch], grid_dst, gloc[batch], false, true, s);
        }
    }
  }

 private:
  SpatialTransformerParam param_;
};  // class SpatialTransformerOp

template<typename xpu>
Operator* CreateOp(SpatialTransformerParam param, int dtype);

#if DMLC_USE_CXX11
class SpatialTransformerProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
      return {"data", "loc"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "grid_dst", "grid_src"};
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, loc]";
    CHECK_EQ(param_.transform_type, st::kAffine) << "only supports affine transform currently";
    CHECK_EQ(param_.sampler_type, st::kBilinear) << "only supports bilinear sampling currently";
    const TShape &dshape = (*in_shape)[st::kData];
    const TShape &lshape = (*in_shape)[st::kLoc];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4U) \
        << "input data should be 4D in batch-num_filter-y-x";
    if (lshape.ndim() ==  0) return false;
    CHECK_EQ(lshape.ndim(), 2U) \
        << "locolisation paramter should be 4D in batch-num_hidden";
    if (param_.transform_type == st::kAffine) {
      CHECK_EQ(lshape[1], 6U) << "incorrect locolisation network shape[1], should be 6";
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    CHECK_GT(param_.target_shape[0], 0U) \
        << "incorrect target_shape: " << param_.target_shape[0];
    CHECK_GT(param_.target_shape[1], 0U) \
        << "incorrect target_shape: " << param_.target_shape[1];
    (*out_shape)[st::kOut][2] = param_.target_shape[0];
    (*out_shape)[st::kOut][3] = param_.target_shape[1];
    out_shape->push_back(Shape2(3, param_.target_shape[0]*param_.target_shape[1]));
    out_shape->push_back(Shape3(dshape[0], 2, param_.target_shape[0]*param_.target_shape[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      int dtype = -1;
      for (size_t i = 0; i < in_type->size(); ++i) {
        if (dtype == -1) {
          dtype = in_type->at(i);
        } else {
          CHECK(in_type->at(i) == dtype ||
                in_type->at(i) == -1) <<
                "Non-uniform data type in SpatialTransformer";
        }
      }
      if (dtype == -1) {
        LOG(FATAL) << "Not enough information to infer type in SpatialTransformer.";
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
    auto ptr = new SpatialTransformerProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SpatialTransformer";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[st::kOut],
            out_data[st::kGridDst],
            out_data[st::kGridSrc],
            in_data[st::kData]
           };
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  #if CUDNN_MAJOR >= 5
  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }
  #endif

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SpatialTransformerParam param_;
};  // class SpatialTransformerProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SPATIAL_TRANSFORMER_INL_H_
