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
 *  Copyright (c) 2015 by Contributors
 * \file torch_base.h
 * \brief Torch interface.
 * \author Junyuan Xie
 */
#ifndef PLUGIN_TORCH_TORCH_BASE_H_
#define PLUGIN_TORCH_TORCH_BASE_H_
#include <mxnet/base.h>

extern "C" {
#include <lua.h>
#include <luaT.h>
#include <lualib.h>
#include <TH/THStorage.h>
#include <TH/THTensor.h>
}

#if MXNET_USE_CUDA
extern "C" {
#include <THC/THCStorage.h>
#include <THC/THCTensor.h>
#include <THC/THCTensorCopy.h>
}
#endif  // MXNET_USE_CUDA

#include <vector>

namespace mxnet {

class TorchState {
 public:
  lua_State* L;
  TorchState();
  static TorchState* ThreadSharedLuaState();

#if MXNET_USE_CUDA
  THCState* CudaState() {
    lua_getglobal(L, "cutorch");
    CHECK(!lua_isnil(L, -1));
    lua_getfield(L, -1, "_state");
    CHECK(!lua_isnil(L, -1));
    THCState* state = reinterpret_cast<THCState*>(lua_touserdata(L, -1));
    lua_pop(L, 2);
    return state;
  }
#endif  // MXNET_USE_CUDA

  template<typename xpu>
  void SetStream(mshadow::Stream<xpu>* s);

  void PrintState() {
    int i;
    int top = lua_gettop(L);
    LOG(INFO) << "Stack height: " << top;
    for (i = 1; i <= top; i++) {  /* repeat for each level */
      int t = lua_type(L, i);
      switch (t) {
        case LUA_TSTRING:  /* strings */
          LOG(INFO) << i << ": '" << lua_tostring(L, i) << "'";
          break;
        case LUA_TBOOLEAN:  /* booleans */
          LOG(INFO) << i << ": " << (lua_toboolean(L, i) ? "true" : "false");
          break;
        case LUA_TNUMBER:  /* numbers */
          LOG(INFO) << i << ": " << lua_tonumber(L, i);
          break;
        default:  /* other values */
          LOG(INFO) << i << ": " << lua_typename(L, t);
          break;
      }
    }
  }

  int Deserialize(THCharStorage* chunk) {  // read only to the chunk
    CHECK_NE(chunk, NULL);
    lua_getglobal(L, "Deserialize");
    luaT_pushudata(L, chunk, "torch.CharStorage");
    THCharStorage_retain(chunk);  // keep it because read only
    int err = lua_pcall(L, 1, 1, 0);
    CHECK_EQ(err, 0);
    return 1;
  }

  int Serialize(THCharStorage** chunk) {
    lua_getglobal(L, "Serialize");
    lua_pushvalue(L, -2);
    int err = lua_pcall(L, 1, 1, 0);
    CHECK_EQ(err, 0) << "Serialize failed " << lua_tostring(L, -1);
    THCharStorage_free(*chunk);  // free the original
    *chunk = reinterpret_cast<THCharStorage*>(luaT_toudata(L, -1, "torch.CharStorage"));
    THCharStorage_retain(*chunk);  // keep the chunk even when lua side deletes
    lua_pop(L, 2);
    return 0;
  }
};

typedef void* THGeneralTensor;
typedef void* THGeneralStorage;

class TorchTensor {
 public:
  static const char* TensorType(int dev_mask) {
    switch (dev_mask) {
      case cpu::kDevMask:
        return "torch.FloatTensor";
      case gpu::kDevMask:
        return "torch.CudaTensor";
      default:
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        return NULL;
    }
  }

  static const char* ModuleType(int dev_mask) {
    switch (dev_mask) {
      case cpu::kDevMask:
        return ":float()";
      case gpu::kDevMask:
        return ":cuda()";
      default:
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        return NULL;
    }
  }

  static const char* TensorType(TBlob data) {
    return TensorType(data.dev_mask());
  }

  static const char* ModuleType(TBlob data) {
    return TensorType(data.dev_mask());
  }

  static THGeneralTensor TBlobToTHTensor(TorchState* torchState, TBlob data) {
    size_t size = data.Size();
    THGeneralTensor tensor = NULL;
    THLongStorage* thshape = THLongStorage_newWithSize(data.ndim());
    for (int i = 0; i < data.ndim(); ++i) {
      THLongStorage_set(thshape, i, data.shape_[i]);
    }
    CHECK_EQ(data.type_flag_, mshadow::kFloat32) << "Torch Interface only support float32";
    switch (data.dev_mask()) {
      case cpu::kDevMask: {
        THFloatStorage* storage = THFloatStorage_newWithData(static_cast<real_t*>(data.dptr_),
                                                             size);
        THFloatStorage_clearFlag(storage, TH_STORAGE_FREEMEM);
        tensor = (THGeneralTensor)THFloatTensor_newWithStorage(storage, 0, thshape, NULL);
        THFloatStorage_free(storage);
        break;
      }
#if MXNET_USE_CUDA
      case gpu::kDevMask: {
        THCState* state = torchState->CudaState();
        THCudaStorage* storage = THCudaStorage_newWithData(state, static_cast<real_t*>(data.dptr_),
                                                           size);
        // a bug in cutorch
        THFloatStorage_clearFlag(reinterpret_cast<THFloatStorage*>(storage), TH_STORAGE_FREEMEM);
        tensor = (THGeneralTensor)THCudaTensor_newWithStorage(state, storage, 0, thshape, NULL);
        THCudaStorage_free(state, storage);
        break;
      }
#endif
      default:
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
    }
    THLongStorage_free(thshape);

    return tensor;
  }

  static void FreeInternal(TorchState* torchState, THGeneralTensor tensor, int dev_mask) {
    switch (dev_mask) {
      case cpu::kDevMask: {
        THFloatStorage* original = static_cast<THFloatTensor*>(tensor)->storage;
        THFloatStorage_free(original);
        break;
      }
#if MXNET_USE_CUDA
      case gpu::kDevMask: {
        THCState* state = torchState->CudaState();
        THCudaStorage* original = static_cast<THCudaTensor*>(tensor)->storage;
        THCudaStorage_free(state, original);
        break;
      }
#endif
      default:
        LOG(FATAL) << "Unknown device type " << dev_mask;
    }
  }

  static void SetInternal(TorchState* torchState, THGeneralTensor tensor, const TBlob& blob) {
    size_t size = blob.Size();
    switch (blob.dev_mask()) {
      case cpu::kDevMask: {
        THFloatStorage* storage = THFloatStorage_newWithData(static_cast<real_t*>(blob.dptr_),
                                                             size);
        THFloatStorage_clearFlag(storage, TH_STORAGE_FREEMEM);
        THFloatStorage* original = static_cast<THFloatTensor*>(tensor)->storage;
        static_cast<THFloatTensor*>(tensor)->storage = storage;
        THFloatStorage_free(original);
        break;
      }
#if MXNET_USE_CUDA
      case gpu::kDevMask: {
        THCState* state = torchState->CudaState();
        THCudaStorage* storage = THCudaStorage_newWithData(state,
                                                           static_cast<real_t*>(blob.dptr_),
                                                           size);
        // TODO(min): torch bug Cuda version not implemented
        THFloatStorage_clearFlag(reinterpret_cast<THFloatStorage*>(storage), TH_STORAGE_FREEMEM);
        THCudaStorage* original = static_cast<THCudaTensor*>(tensor)->storage;
        static_cast<THCudaTensor*>(tensor)->storage = storage;
        THCudaStorage_free(state, original);
        break;
      }
#endif
      default:
        LOG(FATAL) << "Unknown device type " << blob.dev_mask();
    }
  }

  static std::vector<THGeneralTensor> TBlobVectorAsTable(
    TorchState* torchState,
    const std::vector<TBlob>::const_iterator begin,
    const std::vector<TBlob>::const_iterator end) {
    lua_State* L = torchState->L;
    std::vector<THGeneralTensor> res;
    int num = end - begin;
    if (num > 1) {
      lua_createtable(L, num, 0);
      int index = 1;
      for (std::vector<TBlob>::const_iterator it = begin; it != end; ++it) {
        THGeneralTensor th = TorchTensor::TBlobToTHTensor(torchState, *it);
        res.push_back(th);
        luaT_pushudata(L, th, TorchTensor::TensorType(*it));
        lua_rawseti(L, -2, index++);
      }
    } else if (num == 0) {
      lua_pushnil(L);
    } else {
      THGeneralTensor th = TorchTensor::TBlobToTHTensor(torchState, *begin);
      res.push_back(th);
      luaT_pushudata(L, th, TorchTensor::TensorType(*begin));
    }
    return res;
  }

  static void CopyIfDifferent(TorchState* torchState, TBlob dst, THGeneralTensor th_dst) {
    lua_State* L = torchState->L;
    if (luaT_isudata(L, -1, TorchTensor::TensorType(cpu::kDevMask))) {
      CHECK_EQ(dst.dev_mask(), cpu::kDevMask) << "Device type mismatch.";
      THFloatTensor* src = static_cast<THFloatTensor*>(
        luaT_toudata(L, -1, TorchTensor::TensorType(cpu::kDevMask)));
      if (src->storage != static_cast<THFloatTensor*>(th_dst)->storage) {
        THFloatTensor_copy(static_cast<THFloatTensor*>(th_dst), src);
      }
#if MXNET_USE_CUDA
    } else if (luaT_isudata(L, -1, TorchTensor::TensorType(gpu::kDevMask))) {
      CHECK_EQ(dst.dev_mask(), gpu::kDevMask) << "Device type mismatch.";
      THCudaTensor* src = static_cast<THCudaTensor*>(
        luaT_toudata(L, -1, TorchTensor::TensorType(gpu::kDevMask)));
      if (src->storage != static_cast<THCudaTensor*>(th_dst)->storage) {
        THCudaTensor_copy(torchState->CudaState(), static_cast<THCudaTensor*>(th_dst), src);
      }
#endif  // MXNET_USE_CUDA
    } else {
      LOG(FATAL) << "Unsupported Torch tensor type " << luaT_typename(L, -1);
    }
  }

  static void CheckOutput(TorchState* torchState,
                          std::vector<TBlob>::const_iterator begin,
                          std::vector<TBlob>::const_iterator end,
                          std::vector<THGeneralTensor>::const_iterator th_begin,
                          std::vector<THGeneralTensor>::const_iterator th_end) {
    lua_State* L = torchState->L;
    int num = end - begin;
    CHECK_EQ(th_end - th_begin, num);
    if (num == 0) {
    } else if (num == 1) {
      CopyIfDifferent(torchState, *begin, *th_begin);
    } else {
      CHECK(lua_istable(L, -1));
      lua_pushnil(L);
      for (; begin != end; ++begin, ++th_begin) {
        CHECK(lua_next(L, -2));
        CopyIfDifferent(torchState, *begin, *th_begin);
        lua_pop(L, 1);
      }
      lua_pop(L, 1);
    }
  }
};

}  // namespace mxnet
#endif  // PLUGIN_TORCH_TORCH_BASE_H_
