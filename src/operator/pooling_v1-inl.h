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
 * \file pooling_v1-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_POOLING_V1_INL_H_
#define MXNET_OPERATOR_POOLING_V1_INL_H_

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

namespace pool_v1_enum {
enum PoolingV1OpInputs {kData};
enum PoolingV1OpOutputs {kOut};
enum PoolingV1OpType {kMaxPooling, kAvgPooling, kSumPooling};
enum PoolingV1OpPadConventionType {kValid, kFull};
}  // namespace pool_v1_enum

struct PoolingV1Param : public dmlc::Parameter<PoolingV1Param> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  int pooling_convention;
  bool global_pool;
  DMLC_DECLARE_PARAMETER(PoolingV1Param) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. ");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_v1_enum::kMaxPooling)
    .add_enum("avg", pool_v1_enum::kAvgPooling)
    .add_enum("sum", pool_v1_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(pool_v1_enum::kValid)
    .add_enum("full", pool_v1_enum::kFull)
    .add_enum("valid", pool_v1_enum::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for pooling: (y, x) or (d, y, x)");
  }
};

template<typename xpu, typename Reducer, typename DType>
class PoolingV1Op : public Operator {
 public:
  explicit PoolingV1Op(PoolingV1Param p) {
    this->param_ = p;
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    }

    // reset padding size for global pooling
    TShape padding = param_.pad;
    if (param_.global_pool) {
      padding[0] = padding[1] = 0;
    }

    Tensor<xpu, 4, DType> data = in_data[pool_v1_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[pool_v1_enum::kOut].get<xpu, 4, DType>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    if (param_.pool_type == pool_v1_enum::kMaxPooling
        || param_.pool_type == pool_v1_enum::kSumPooling) {
      Assign(out,
             req[pool_v1_enum::kOut],
             pool<Reducer>(pad(data, padding[0], padding[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
    } else if (param_.pool_type == pool_v1_enum::kAvgPooling) {
      Assign(out,
             req[pool_v1_enum::kOut],
             scalar<DType>(1.0f / (param_.global_pool ?
                      data.shape_[2] * data.shape_[3] :
                      param_.kernel[0] * param_.kernel[1])) * \
             pool<Reducer>(pad(data, padding[0], padding[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    // TODO(bing): remove pad (0,0)
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    }

    // reset padding size for global pooling
    TShape padding = param_.pad;
    if (param_.global_pool) {
      padding[0] = padding[1] = 0;
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad = out_grad[pool_v1_enum::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[pool_v1_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[pool_v1_enum::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> input_grad = in_grad[pool_v1_enum::kData].get<xpu, 4, DType>(s);

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]);

    if (param_.pool_type == pool_v1_enum::kMaxPooling
        || param_.pool_type == pool_v1_enum::kSumPooling) {
      Assign(input_grad, req[pool_v1_enum::kData],
             crop(unpool<Reducer>(pad(data, padding[0], padding[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  padding[0],
                  padding[1]));
    } else if (param_.pool_type == pool_v1_enum::kAvgPooling) {
      Assign(input_grad, req[pool_v1_enum::kData],
             scalar<DType>(1.0f / (param_.global_pool ?
                      data.shape_[2] * data.shape_[3] :
                      param_.kernel[0] * param_.kernel[1])) * \
             crop(unpool<Reducer>(pad(data, padding[0], padding[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  padding[0],
                  padding[1]));
    }
  }

 private:
  PoolingV1Param param_;
};  // class PoolingV1Op

template<typename xpu>
Operator* CreateOp(PoolingV1Param param, int dtype);


#if DMLC_USE_CXX11
class PoolingV1Prop : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
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

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x) "
                               << "Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
      } else {
        CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
            << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
            << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
        CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
            << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
            << " padded to " << (dshape[3] + 2*param_.pad[1]) << ")";
        if (param_.pooling_convention == pool_v1_enum::kValid) {
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
      }
      out_shape->clear();
      out_shape->push_back(oshape);
    } else if (param_.kernel.ndim() == 3) {
      CHECK_EQ(dshape.ndim(), 5) << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
      CHECK_LE(param_.kernel[0], dshape[2] + 2 * param_.pad[0]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2]) << "kernel size exceeds input";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
        oshape[4] = 1;
      } else {
        if (param_.pooling_convention == pool_v1_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                              param_.stride[1];
          oshape[4] = 1 + (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) /
                              param_.stride[2];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[3] + 2 * param_.pad[1] -
                              param_.kernel[1]) / param_.stride[1]));
          oshape[4] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[4] + 2 * param_.pad[2] -
                              param_.kernel[2]) / param_.stride[2]));
        }
      }

      out_shape->clear();
      out_shape->push_back(oshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = (*in_type)[0];

    if (dtype == -1) {
      LOG(FATAL) << "Input type to pooling is not specified.";
      return false;
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingV1Prop *prop_sym = new PoolingV1Prop();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Pooling_v1";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_v1_enum::kOut], in_data[pool_v1_enum::kData],
            out_data[pool_v1_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[pool_v1_enum::kData], in_grad[pool_v1_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PoolingV1Param param_;
};  // class PoolingV1Prop
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_V1_INL_H_
