/*!
 * Copyright (c) 2015 by Contributors
 * \file torch_module-inl.h
 * \brief torch module operator
 * \author Min Lin
*/
#ifndef PLUGIN_TORCH_TORCH_MODULE_INL_H_
#define PLUGIN_TORCH_TORCH_MODULE_INL_H_

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
struct TorchModuleParam : public dmlc::Parameter<TorchModuleParam> {
  std::string lua_string;
  uint32_t num_data;
  uint32_t num_params;
  uint32_t num_outputs;
  DMLC_DECLARE_PARAMETER(TorchModuleParam) {
    DMLC_DECLARE_FIELD(lua_string)
    .describe("lua string that is called to generate the torch module object");
    DMLC_DECLARE_FIELD(num_data)
    .describe("the number of input data");
    DMLC_DECLARE_FIELD(num_params)
    .describe("the number of parameters");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("the number of outputs");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class TorchModuleOp : public Operator {
 private:
  TorchModuleParam param_;
  TorchState* torchState_;
  int lua_reference_;

 public:
  explicit TorchModuleOp(TorchModuleParam p, TorchState* torchState) : torchState_(torchState) {
    this->param_ = p;
    lua_State* L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    std::string exec = std::string("return ") + p.lua_string
      + TorchTensor::ModuleType(xpu::kDevMask);
    CHECK_EQ(luaL_loadstring(L, exec.c_str()), 0);
    int err = lua_pcall(L, 0, 1, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    // Get number of parameters
    uint32_t param_num = 0;
    lua_getfield(L, -1, "parameters");
    lua_pushvalue(L, -2);
    CHECK_EQ(lua_pcall(L, 1, LUA_MULTRET, 0), 0);
    if (lua_gettop(L) == 1) {
      param_num = 0;
    } else {
      CHECK_EQ(lua_gettop(L), 3);
      param_num = lua_objlen(L, -2);
      lua_pop(L, 2);
    }
    CHECK_EQ(param_num, param_.num_params);
    // Free the parameters allocated by torch so it doesn't take up memory.
    if (param_.num_params != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, 1, 0);
      CHECK_EQ(err, 0);
      // iterate the parameters table to free tblobs inside
      lua_pushnil(L);
      while (lua_next(L, -2)) {
        CHECK(luaT_isudata(L, -1, TorchTensor::TensorType(xpu::kDevMask)));
        void* udata = luaT_toudata(L, -1, TorchTensor::TensorType(xpu::kDevMask));
        TorchTensor::FreeInternal(torchState_, static_cast<THGeneralTensor>(udata), xpu::kDevMask);
        lua_pop(L, 1);
      }
      lua_pop(L, 1);  // pop the parameter table
    }
    this->lua_reference_ = luaL_ref(L, LUA_REGISTRYINDEX);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    lua_State* L = torchState_->L;

    CHECK_EQ(lua_gettop(L), 0);
    CHECK_EQ(in_data.size(), param_.num_params + param_.num_data);
    CHECK_EQ(out_data.size(), param_.num_outputs);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    torchState_->SetStream(s);
    // Deserialize self table

    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);

    std::vector<THGeneralTensor> th_output =
      TorchTensor::TBlobVectorAsTable(torchState_, out_data.begin(),
                                      out_data.begin() + param_.num_outputs);
    // set the output field
    lua_setfield(L, -2, "output");
    // set the parameters
    if (param_.num_params != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, 1, 0);
      CHECK_EQ(err, 0);
      // iterate the parameters table to put tblobs inside
      lua_pushnil(L);
      std::vector<TBlob>::const_iterator it = in_data.begin() + param_.num_data;
      while (lua_next(L, -2)) {
        CHECK(luaT_isudata(L, -1, TorchTensor::TensorType(*it)));
        void* udata = luaT_toudata(L, -1, TorchTensor::TensorType(*it));
        TorchTensor::SetInternal(torchState_, static_cast<THGeneralTensor>(udata), *(it));
        it++;
        lua_pop(L, 1);
      }
      lua_pop(L, 1);  // pop the parameter table
    }
    // call updateOutput
    // | self
    lua_getfield(L, -1, "updateOutput");
    // | self | updateOutput
    lua_pushvalue(L, -2);
    // | self | updateOutput | self
    TorchTensor::TBlobVectorAsTable(torchState_, in_data.begin(),
                                    in_data.begin() + param_.num_data);
    // | self | updateOutput | self | inputs
    int err = lua_pcall(L, 2, 1, 0);  // doesn't need the output
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    TorchTensor::CheckOutput(torchState_, out_data.begin(), out_data.begin() + param_.num_outputs,
                             th_output.begin(), th_output.end());
    lua_pop(L, 2);
    CHECK_EQ(lua_gettop(L), 0);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    lua_State* L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    CHECK_EQ(in_data.size(), param_.num_params + param_.num_data);
    CHECK_EQ(out_data.size(), param_.num_outputs);
    CHECK_EQ(out_grad.size(), param_.num_outputs);
    CHECK_EQ(in_grad.size(), param_.num_params + param_.num_data);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    torchState_->SetStream(s);
    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);
    TorchTensor::TBlobVectorAsTable(torchState_, out_data.begin(), out_data.end());
    lua_setfield(L, -2, "output");
    std::vector<THGeneralTensor> th_grad =
      TorchTensor::TBlobVectorAsTable(torchState_, in_grad.begin(),
                                      in_grad.begin() + param_.num_data);
    lua_setfield(L, -2, "gradInput");
    if (param_.num_params != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, LUA_MULTRET, 0);
      CHECK_EQ(err, 0) << lua_tostring(L, -1);
      // iterate the parameters table to put tblobs inside
      lua_pushnil(L);
      std::vector<TBlob>::const_iterator it = in_data.begin() + param_.num_data;
      while (lua_next(L, -3)) {
        TorchTensor::SetInternal(
          torchState_,
          static_cast<THGeneralTensor>(luaT_toudata(L, -1, TorchTensor::TensorType(*it))),
          *it);
        it++;
        lua_pop(L, 1);
      }
      // iterate the grad of params
      lua_pushnil(L);
      it = in_grad.begin() + param_.num_data;;
      while (lua_next(L, -2)) {
        TorchTensor::SetInternal(
          torchState_,
          static_cast<THGeneralTensor>(luaT_toudata(L, -1, TorchTensor::TensorType(*it))),
          *it);
        it++;
        lua_pop(L, 1);
      }
      lua_pop(L, 2);  // pop the parameters
    }
    lua_getfield(L, -1, "zeroGradParameters");
    lua_pushvalue(L, -2);
    CHECK_EQ(lua_pcall(L, 1, 0, 0), 0);
    TorchTensor::TBlobVectorAsTable(torchState_, in_data.begin(),
                                    in_data.begin() + param_.num_data);
    TorchTensor::TBlobVectorAsTable(torchState_, out_grad.begin(), out_grad.end());
    // call
    lua_getfield(L, -3, "accGradParameters");
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushnumber(L, 1);
    int err = lua_pcall(L, 4, 0, 0);  // doesn't need the output
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    lua_getfield(L, -3, "updateGradInput");
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    err = lua_pcall(L, 3, 1, 0);  // doesn't need the output
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    TorchTensor::CheckOutput(torchState_, in_grad.begin(), in_grad.begin() + param_.num_data,
                             th_grad.begin(), th_grad.end());
    lua_pop(L, 4);
    CHECK_EQ(lua_gettop(L), 0);
  }
};  // class TorchModuleOp

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(TorchModuleParam type, TorchState* torchState);

#if DMLC_USE_CXX11
class TorchModuleProp : public OperatorProperty {
 protected:
  mutable std::vector<std::string> arguments_;
  mutable TorchState* torchState_;
  mutable int lua_reference_;

  void InitTorchState() const {
    this->torchState_ = new TorchState();
    lua_State* L = torchState_->L;
    std::string exec = std::string("return ") + param_.lua_string;
    CHECK_EQ(luaL_loadstring(L, exec.c_str()), 0);
    int err = lua_pcall(L, 0, LUA_MULTRET, 0);
    CHECK_EQ(lua_gettop(L), 1);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    lua_getfield(L, -1, "float");
    lua_pushvalue(L, -2);
    err = lua_pcall(L, 1, 1, 0);
    CHECK_EQ(err, 0);
    lua_reference_ = lua_ref(L, LUA_REGISTRYINDEX);
    lua_pop(L, 1);

    CHECK_EQ(lua_gettop(L), 0);
  }

 public:
  TorchModuleProp() : OperatorProperty(), torchState_(NULL), lua_reference_(-1) {
  }

  std::vector<std::string> ListArguments() const override {
    if (!torchState_) {
      InitTorchState();
    }
    lua_State* L = torchState_->L;

    if (arguments_.size() == 0) {
      for (uint32_t i = 0; i < param_.num_data; ++i) {
        std::string data = "data_" + std::to_string(i);
        arguments_.push_back(data);
      }
      std::string lua_code =
          "return function(module)\n"
          "          local params = module:parameters()\n"
          "          local dict = {}\n"
          "          if params == nil then\n"
          "             return {}\n"
          "          end\n"
          "          for id, p in ipairs(params) do\n"
          "             dict[p] = string.format('param_%d', id)\n"
          "          end\n"
          "          for key, value in pairs(module) do\n"
          "             if dict[value] then\n"
          "                dict[value] = key\n"
          "             end\n"
          "          end\n"
          "          local ret = {}\n"
          "          for _, p in ipairs(params) do\n"
          "             table.insert(ret, dict[p])\n"
          "          end\n"
          "          return ret\n"
          "end\n";
      luaL_loadstring(L, lua_code.c_str());
      int err = lua_pcall(L, 0, 1, 0);  // return the function
      CHECK_EQ(err, 0) << lua_tostring(L, -1);
      lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);
      err = lua_pcall(L, 1, 1, 0);  // call the function
      CHECK_EQ(err, 0) << lua_tostring(L, -1);
      lua_pushnil(L);
      while (lua_next(L, -2)) {
        arguments_.push_back(lua_tostring(L, -1));
        lua_pop(L, 1);
      }
      lua_pop(L, 1);
    }
    return arguments_;
  }

  virtual std::vector<std::string> ListOutputs() const {
    if (param_.num_outputs > 1) {
      std::vector<std::string> ret;
      std::string output = "output";
      for (uint32_t i = 0; i < param_.num_outputs; ++i) {
        ret.push_back(output + std::to_string(i));
      }
      return ret;
    } else {
      return {"output"};
    }
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
    if (torchState_ == nullptr) {
      this->InitTorchState();
    }
    lua_State* L = torchState_->L;

    CHECK_EQ(lua_gettop(L), 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);
    CHECK_EQ(in_shape->size(), param_.num_data + param_.num_params);
    CHECK_EQ(out_shape->size(), param_.num_outputs);
    CHECK_EQ(aux_shape->size(), 0);
    lua_getfield(L, -1, "updateOutput");
    lua_pushvalue(L, -2);  // self
    if (param_.num_data == 1) {
      THLongStorage* thshape = THLongStorage_newWithSize((*in_shape)[0].ndim());
      for (uint32_t i = 0; i < (*in_shape)[0].ndim(); ++i) {
        THLongStorage_set(thshape, i, (*in_shape)[0][i]);
      }
      THFloatTensor* in_data = THFloatTensor_newWithSize(thshape, NULL);
      THLongStorage_free(thshape);
      luaT_pushudata(L, in_data, TorchTensor::TensorType(mshadow::cpu::kDevMask));
    } else if (param_.num_data > 1) {
      lua_createtable(L, param_.num_data, 0);
      for (uint32_t data_index = 0; data_index < param_.num_data; ++data_index) {
        THLongStorage* thshape = THLongStorage_newWithSize((*in_shape)[data_index].ndim());
        for (uint32_t i = 0; i < (*in_shape)[data_index].ndim(); ++i) {
          THLongStorage_set(thshape, i, (*in_shape)[data_index][i]);
        }
        THFloatTensor* in_data = THFloatTensor_newWithSize(thshape, NULL);
        THLongStorage_free(thshape);
        luaT_pushudata(L, in_data, TorchTensor::TensorType(mshadow::cpu::kDevMask));
        lua_rawseti(L, -2, data_index);
      }
    }
    int err = lua_pcall(L, 2, 0, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    if (param_.num_params != 0) {
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, LUA_MULTRET, 0);
      CHECK_EQ(err, 0);
      CHECK_EQ(lua_gettop(L), 3);
      lua_pushnil(L);
      int index = param_.num_data;
      while (lua_next(L, -3)) {
        THFloatTensor* param = reinterpret_cast<THFloatTensor*>(luaT_toudata(L, -1,
          TorchTensor::TensorType(mshadow::cpu::kDevMask)));
        long int* size = param->size;  // NOLINT(*)
        (*in_shape)[index++] = TShape(size, size + THFloatTensor_nDimension(param));
        lua_pop(L, 1);
      }
      lua_pop(L, 2);
    }
    lua_getfield(L, -1, "output");
    if (param_.num_outputs == 0) {
    } else if (param_.num_outputs == 1) {
      THFloatTensor* output = reinterpret_cast<THFloatTensor*>(luaT_toudata(L, -1,
        TorchTensor::TensorType(mshadow::cpu::kDevMask)));
      long int* size = output->size;  // NOLINT(*)
      (*out_shape)[0] = TShape(size, size + THFloatTensor_nDimension(output));
    } else {
      for (uint32_t data_index = 0; data_index < param_.num_outputs; ++data_index) {
        lua_pushnil(L);
        int index = 0;
        while (lua_next(L, -2)) {
          THFloatTensor* out = reinterpret_cast<THFloatTensor*>(luaT_toudata(L, -1,
            TorchTensor::TensorType(mshadow::cpu::kDevMask)));
          long int* size = out->size;  // NOLINT(*)
          (*out_shape)[index++] = TShape(size, size + THFloatTensor_nDimension(out));
        }
      }
    }
    lua_pop(L, 2);
    CHECK_EQ(lua_gettop(L), 0);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TorchModuleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "TorchModule";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep;
    dep.insert(dep.end(), out_grad.begin(), out_grad.end());
    dep.insert(dep.end(), out_data.begin(), out_data.end());
    dep.insert(dep.end(), in_data.begin(), in_data.end());
    return dep;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  TorchModuleParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_TORCH_TORCH_MODULE_INL_H_
