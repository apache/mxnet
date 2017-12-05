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
 * \file torch_module-inl.h
 * \brief torch module operator
 * \author Min Lin
*/
#ifndef PLUGIN_TORCH_TORCH_CRITERION_INL_H_
#define PLUGIN_TORCH_TORCH_CRITERION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <stdio.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../../src/operator/operator_common.h"
#include "./torch_base.h"

namespace mxnet {
namespace op {
struct TorchCriterionParam : public dmlc::Parameter<TorchCriterionParam> {
  std::string lua_string;
  TShape label_shape;
  float grad_scale;
  DMLC_DECLARE_PARAMETER(TorchCriterionParam) {
    DMLC_DECLARE_FIELD(lua_string)
    .describe("lua string that is called to generate the torch criterion object");
    DMLC_DECLARE_FIELD(label_shape)
    .set_default(TShape())
    .enforce_nonzero()
    .describe("Shape of label (without batch size).");
    DMLC_DECLARE_FIELD(grad_scale)
    .set_default(1.0f)
    .describe("Scale the gradient by a float factor (a.k.a weight of this loss).");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class TorchCriterionOp : public Operator {
 private:
  TorchCriterionParam param_;
  TorchState* torchState_;
  int lua_reference_;

 public:
  explicit TorchCriterionOp(TorchCriterionParam p) {
    this->param_ = p;
    this->torchState_ = new TorchState();
    lua_State *L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    std::string exec = std::string("return ") + p.lua_string
      + TorchTensor::ModuleType(xpu::kDevMask);
    CHECK_EQ(luaL_loadstring(L, exec.c_str()), 0);
    int err = lua_pcall(L, 0, 1, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    // serialize
    this->lua_reference_ = lua_ref(L, LUA_REGISTRYINDEX);
  }

  ~TorchCriterionOp() {
    delete this->torchState_;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    lua_State *L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    torchState_->SetStream(s);
    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);
    // call forward
    // | self
    lua_getfield(L, -1, "forward");
    // | self | forward
    lua_pushvalue(L, -2);
    // | self | forward | self
    for (index_t i = 0; i < in_data.size(); ++i) {
      THGeneralTensor th = TorchTensor::TBlobToTHTensor(torchState_, in_data[i]);
      luaT_pushudata(L, th, TorchTensor::TensorType(in_data[i]));
    }
    // | self | forward | self | pred | label
    int err = lua_pcall(L, 3, 1, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    CHECK(lua_isnumber(L, -1)) << "Criterion must return a number";
    real_t loss = static_cast<real_t>(lua_tonumber(L, -1));
    lua_pop(L, 1);
    Tensor<xpu, 2> out = out_data[0].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[0], loss*param_.grad_scale);
    lua_pop(L, 1);
    CHECK_EQ(lua_gettop(L), 0);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    lua_State *L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req[0], kWriteTo) << "Torch Criterion only supports write to in_grad";
    CHECK_EQ(req[1], kNullOp) << "Torch Criterion cannot back prop to label";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    torchState_->SetStream(s);
    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);
    THGeneralTensor th = TorchTensor::TBlobToTHTensor(torchState_, in_grad[0]);
    luaT_pushudata(L, th, TorchTensor::TensorType(in_grad[0]));
    lua_setfield(L, -2, "gradInput");
    lua_getfield(L, -1, "backward");
    // | self | backward
    lua_pushvalue(L, -2);
    // | self | backward | self
    for (index_t i = 0; i < in_data.size(); ++i) {
      th = TorchTensor::TBlobToTHTensor(torchState_, in_data[i]);
      luaT_pushudata(L, th, TorchTensor::TensorType(in_data[i]));
    }
    // | self | forward | self | pred | label
    int err = lua_pcall(L, 3, 0, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    Tensor<xpu, 2> grad = in_grad[0].FlatTo2D<xpu, real_t>(s);
    grad *= param_.grad_scale * in_grad[0].shape_[0];
    lua_pop(L, 1);
    CHECK_EQ(lua_gettop(L), 0);
  }
};  // class TorchCriterionOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(TorchCriterionParam type);

#if DMLC_USE_CXX11
class TorchCriterionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  virtual std::vector<std::string> ListOutputs() const {
    return {"output"};
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
    CHECK_EQ(in_shape->size(), 2);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    std::vector<index_t> lshape;
    lshape.push_back(dshape[0]);
    lshape.insert(lshape.end(), param_.label_shape.data(),
      param_.label_shape.data() +  param_.label_shape.ndim());
    TShape shape(lshape.begin(), lshape.end());
    SHAPE_ASSIGN_CHECK(*in_shape, 1, shape);
    out_shape->clear();
    out_shape->push_back(Shape1(dshape[0]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TorchCriterionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "TorchCriterion";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep;
    dep.insert(dep.end(), in_data.begin(), in_data.end());
    // Ensure that the backward and forward cannot be called at the same time
    dep.insert(dep.end(), out_data.begin(), out_data.end());
    return dep;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  TorchCriterionParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_TORCH_TORCH_CRITERION_INL_H_
