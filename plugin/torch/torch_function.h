/*!
 *  Copyright (c) 2015 by Contributors
 * \file torch_function.h
 * \brief Torch interface.
 * \author Junyuan Xie
 */
#ifndef PLUGIN_TORCH_TORCH_FUNCTION_H_
#define PLUGIN_TORCH_TORCH_FUNCTION_H_
#include "./torch_base.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <algorithm>
#include <vector>

namespace mxnet {

template<typename xpu, typename OP>
void TorchRunOp(std::vector<NDArray> arr_in,
                std::vector<NDArray> arr_out,
                const std::map<std::string, std::string>& param,
                RunContext ctx) {
  TorchState* torchState = TorchState::ThreadSharedLuaState();
  torchState->SetStream(ctx.get_stream<xpu>());
  lua_State* L = torchState->L;

  lua_getglobal(L, "torch");
  lua_getfield(L, -1, OP::fname);
  int idx = 0;
  std::vector<NDArray> arr(arr_out.begin(), arr_out.end());
  arr.insert(arr.end(), arr_in.begin(), arr_in.end());
  std::string format = param.at("format");
  std::istringstream args(param.at("args"));
  for (size_t i = 0; i < format.size(); ++i) {
    std::string val;
    std::getline(args, val, ',');
    switch (format[i]) {
      case 'n': {
        CHECK(idx < arr.size()) << "Too few NDArray arguments for Torch." << OP::fname;
        luaT_pushudata(L,
                       TorchTensor::TBlobToTHTensor(torchState, arr[idx].data()),
                       TorchTensor::TensorType(arr[idx].data()));
        idx++;
        break;
      }
      case 'i':
        lua_pushinteger(L, std::stoi(val));
        break;
      case 'f':
        lua_pushnumber(L, std::stof(val));
        break;
      case 's':
        lua_pushstring(L, val.c_str());
        break;
      case 'b':
        lua_pushboolean(L, std::stoi(val));
        break;
      default:
        LOG(FATAL) << "Unknown argument type " << format[i] << " for Torch." << OP::fname;
    }
  }
  CHECK_EQ(lua_pcall(L, format.size(), 0, 0), 0) << "Lua Error: " << lua_tostring(L, -1);
}

template<typename OP>
void TorchOp(NDArray **u, real_t *s, NDArray **out,
             const std::map<std::string, std::string>& param) {
  std::vector<mshadow::TShape> shapes = OP::GetShape(u, param);
  CHECK_EQ(shapes.size(), OP::num_outputs)
    << "Too many output shapes for TorchOp " << OP::fname;
  Context ctx;
  int type_flag;
  if (OP::num_inputs) {
    ctx = u[0]->ctx();
    type_flag = u[0]->dtype();
    for (int i = 0; i < OP::num_inputs; ++i) {
      CHECK_EQ(ctx, u[i]->ctx()) << "Context of all oprands must be the same.";
      CHECK_EQ(type_flag, u[i]->dtype()) << "Data type of all oprands must be the same.";
    }
  } else {
    CHECK(param.count("ctx")) << "Must provide keyword argument ctx for TorchOp with 0 inputs";
    std::string str_ctx(param.at("ctx"));
    int id;
    char tmp[4];
    sscanf(str_ctx.c_str(), "%3s(%d)", tmp, &id);
    std::string dev(tmp);
    if (dev == "cpu") {
      ctx = Context::Create(Context::kCPU, id);
    } else if (dev == "gpu") {
      ctx = Context::Create(Context::kGPU, id);
    } else {
      LOG(FATAL) << "Unknown device type " << dev;
    }

    if (param.count("dtype")) {
      std::stringstream str_dtype(param.at("dtype"));
      str_dtype >> type_flag;
    } else {
      type_flag = mshadow::default_type_flag;
    }
  }
  std::vector<NDArray> arr_in, arr_out;
  std::vector<Engine::VarHandle> var_in, var_out, var_const;
  for (int i = 0; i < OP::num_inputs; ++i) {
    arr_in.push_back(*(u[i]));
    var_in.push_back(u[i]->var());
  }
  for (int i = 0; i < OP::num_outputs; ++i) {
    if (out[i]->is_none()) {
      *(out[i]) = NDArray(shapes[i], ctx, false, type_flag);
    }
    arr_out.push_back(*(out[i]));
    var_out.push_back(out[i]->var());
  }
  std::sort(var_in.begin(), var_in.end());
  var_in.resize(std::unique(var_in.begin(), var_in.end()) - var_in.begin());
  std::sort(var_out.begin(), var_out.end());
  var_out.resize(std::unique(var_out.begin(), var_out.end()) - var_out.begin());
  std::set_difference(var_in.begin(), var_in.end(), var_out.begin(), var_out.end(),
                      std::inserter(var_const, var_const.begin()));
  switch (ctx.dev_mask()) {
    case mshadow::cpu::kDevMask: {
      Engine::Get()->PushSync([arr_in, arr_out, param](RunContext rctx) {
        TorchRunOp<mshadow::cpu, OP>(arr_in, arr_out, param, rctx);
      }, ctx, var_const, var_out);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([arr_in, arr_out, param](RunContext rctx) {
        TorchRunOp<mshadow::gpu, OP>(arr_in, arr_out, param, rctx);
      }, ctx, var_const, var_out);
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

struct TorchFirstShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    return {u[0]->shape()};
  }
};

struct TorchConstructorShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    std::vector<index_t> shape;
    std::string format = param.at("format");
    std::istringstream args(param.at("args"));
    std::string val;
    std::getline(args, val, ',');
    CHECK_LE(format.size(), 5) << "Only support up to 4 dimensions.";
    for (size_t i = 1; i < format.size(); ++i) {
      CHECK_EQ(format[i], 'i') << "Only take integer arguments.";
      std::getline(args, val, ',');
      shape.push_back(std::stoi(val));
    }
    mshadow::TShape tshape(shape.begin(), shape.end());
    return {tshape};
  }
  static const int num_inputs = 0;
  static const int num_outputs = 1;
};

#define MXNET_REGISTER_TORCH_FUN(name, OP)                \
  MXNET_REGISTER_NDARRAY_FUN(name)                        \
  .set_function(TorchOp<OP>)                              \
  .set_num_use_vars(OP::num_inputs)                       \
  .set_num_mutate_vars(OP::num_outputs)                   \
  .set_type_mask(kAcceptEmptyMutateTarget)

#define MXNET_REGISTER_TORCH_UNARY_FUN(name, func)                            \
  struct TorchUnaryOpDesc_ ## name ## _ ## func : public TorchFirstShape {    \
    static constexpr const char* fname = #func;                               \
    static const int num_inputs = 1;                                          \
    static const int num_outputs = 1;                                         \
  };                                                                          \
  MXNET_REGISTER_TORCH_FUN(name, TorchUnaryOpDesc_ ## name ## _ ## func)      \
  .add_argument("x", "NDArray", "Input NDArray")

#define MXNET_REGISTER_TORCH_BINARY_FUN(name, func)                           \
  struct TorchBinaryOpDesc_ ## name ## _ ## func : public TorchFirstShape {   \
    static constexpr const char* fname = #func;                               \
    static const int num_inputs = 2;                                          \
    static const int num_outputs = 1;                                         \
  };                                                                          \
  MXNET_REGISTER_TORCH_FUN(name, TorchBinaryOpDesc_ ## name ## _ ## func)

#define MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(name, func)                  \
  MXNET_REGISTER_TORCH_BINARY_FUN(name, func)                                 \
  .add_argument("x1", "NDArray", "First Input NDArray")                       \
  .add_argument("x2", "NDArray", "Second Input NDArray")

#define MXNET_REGISTER_TORCH_TENARY_FUN(name, func)                           \
  struct TorchTenaryOpDesc_ ## name ## _ ## func : public TorchFirstShape {   \
    static constexpr const char* fname = #func;                               \
    static const int num_inputs = 3;                                          \
    static const int num_outputs = 1;                                         \
  };                                                                          \
  MXNET_REGISTER_TORCH_FUN(name, TorchTenaryOpDesc_ ## name ## _ ## func)

#define MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(name, func)                                  \
  struct TorchConstructorOpDesc_ ## name ## _ ## func : public TorchConstructorShape {    \
    static constexpr const char* fname = #func;                                           \
  };                                                                                      \
  MXNET_REGISTER_TORCH_FUN(name, TorchConstructorOpDesc_ ## name ## _ ## func)


}  // namespace mxnet
#endif  // PLUGIN_TORCH_TORCH_FUNCTION_H_
