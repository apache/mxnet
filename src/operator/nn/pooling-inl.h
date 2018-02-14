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
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu, Jun Wu
*/

#ifndef MXNET_OPERATOR_NN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./pool.h"

namespace mxnet {
namespace op {

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  int pooling_convention;
  bool global_pool;
  bool cudnn_off;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. ");

    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn pooling and use MXNet pooling operator. ");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("Pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(pool_enum::kValid)
    .add_enum("full", pool_enum::kFull)
    .add_enum("valid", pool_enum::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.");
  }
};

template<typename xpu, typename DType>
class PoolingOp : public Operator {
 public:
  explicit PoolingOp(PoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<TBlob>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& out_data,
                       const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[pool_enum::kData].shape_;
    TShape padding = param_.pad;
    if (param_.global_pool) {
      for (index_t i = 0; i < padding.ndim(); i++) {
        padding[i] = 0;
      }
    }

    pool(s, in_data[pool_enum::kData].dptr<DType>(),
         in_data[pool_enum::kData].shape_,
         out_data[pool_enum::kOut].shape_,
         param_.global_pool?
           TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
           : param_.kernel,
         padding,
         param_.global_pool? TShape(param_.kernel.ndim()) : param_.stride,
         param_.pool_type,
         req[pool_enum::kOut],
         out_data[pool_enum::kOut].dptr<DType>());
  }

  virtual void Backward(const OpContext& ctx,
                        const std::vector<TBlob>& out_grad,
                        const std::vector<TBlob>& in_data,
                        const std::vector<TBlob>& out_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& in_grad,
                        const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[pool_enum::kData].shape_;
    TShape padding = param_.pad;
    if (param_.global_pool) {
      for (index_t i = 0; i < padding.ndim(); i++) {
        padding[i] = 0;
      }
    }

    unpool(s, out_grad[pool_enum::kOut].dptr<DType>(),
           in_data[pool_enum::kData].dptr<DType>(),
           out_data[pool_enum::kOut].dptr<DType>(),
           in_grad[pool_enum::kData].shape_,
           out_grad[pool_enum::kOut].shape_,
           param_.global_pool?
             TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
             : param_.kernel,
           padding,
           param_.global_pool? TShape(param_.kernel.ndim()) : param_.stride,
           param_.pool_type,
           req[pool_enum::kData],
           in_grad[pool_enum::kData].dptr<DType>());
  }

 private:
  PoolingParam param_;
};  // class PoolingOp

template<typename xpu>
Operator* CreateOp(PoolingParam param, int dtype);


#if DMLC_USE_CXX11
class PoolingProp : public OperatorProperty {
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
    if (param_.kernel.ndim() == 1) {
      CHECK_EQ(dshape.ndim(), 3U) << "Pooling: Input data should be 3D in (batch, channel, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
      } else {
        CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
            << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
            << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
        if (param_.pooling_convention == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
        }
      }
      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
    } else if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
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
        if (param_.pooling_convention == pool_enum::kValid) {
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
      out_shape->push_back(oshape);  // save output shape
    } else if (param_.kernel.ndim() == 3) {
      CHECK_EQ(dshape.ndim(), 5U)
        << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
      CHECK_LE(param_.kernel[0], dshape[2] + 2 * param_.pad[0]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2]) << "kernel size exceeds input";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
        oshape[4] = 1;
      } else {
        if (param_.pooling_convention == pool_enum::kValid) {
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
      out_shape->push_back(oshape);  // save output shape
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
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingProp *prop_sym = new PoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Pooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData],
            out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[pool_enum::kData], in_grad[pool_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PoolingParam param_;
};  // class PoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOLING_INL_H_
