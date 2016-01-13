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
}
#endif  // MXNET_USE_CUDA

#include <vector>

namespace mxnet {

class TorchState {
 public:
  static lua_State* LuaState();

#if MXNET_USE_CUDA
  static THCState* CudaState() {
    lua_State* L = TorchState::LuaState();
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
  static void SetStream(mshadow::Stream<xpu>* s);

  static int Deserialize(THCharStorage* chunk) {  // read only to the chunk
    CHECK_NE(chunk, NULL);
    lua_State* L = LuaState();
    lua_getglobal(L, "Deserialize");
    luaT_pushudata(L, chunk, "torch.CharStorage");
    THCharStorage_retain(chunk);  // keep it because read only
    int err = lua_pcall(L, 1, 1, 0);
    CHECK_EQ(err, 0);
    return 1;
  }

  static int Serialize(THCharStorage** chunk) {
    lua_State* L = LuaState();
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

  static void PrintState() {
    lua_State* L = LuaState();
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
    return TensorType(data.dev_mask_);
  }

  static const char* ModuleType(TBlob data) {
    return TensorType(data.dev_mask_);
  }

  static THGeneralTensor TBlobToTHTensor(TBlob data) {
    size_t size = data.Size();
    THGeneralTensor tensor = NULL;
    THLongStorage* thshape = THLongStorage_newWithSize(data.ndim());
    for (int i = 0; i < data.ndim(); ++i) {
      THLongStorage_set(thshape, i, data.shape_[i]);
    }
    CHECK_EQ(data.type_flag_, mshadow::kFloat32) << "Torch Interface only support float32";
    switch (data.dev_mask_) {
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
        THCState* state = TorchState::CudaState();
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

  static void FreeInternal(THGeneralTensor tensor, int dev_mask) {
    switch (dev_mask) {
      case cpu::kDevMask: {
        THFloatStorage* original = static_cast<THFloatTensor*>(tensor)->storage;
        THFloatStorage_free(original);
        break;
      }
#if MXNET_USE_CUDA
      case gpu::kDevMask: {
        THCState* state = TorchState::CudaState();
        THCudaStorage* original = static_cast<THCudaTensor*>(tensor)->storage;
        THCudaStorage_free(state, original);
        break;
      }
#endif
      default:
        LOG(FATAL) << "Unknown device type " << dev_mask;
    }
  }

  static void SetInternal(THGeneralTensor tensor, const TBlob& blob) {
    size_t size = blob.Size();
    switch (blob.dev_mask_) {
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
        THCState* state = TorchState::CudaState();
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
        LOG(FATAL) << "Unknown device type " << blob.dev_mask_;
    }
  }

  static void TBlobVectorAsTable(const std::vector<TBlob>::const_iterator begin,
                         const std::vector<TBlob>::const_iterator end) {
    lua_State* L = TorchState::LuaState();
    int num = end - begin;
    if (num > 1) {
      lua_createtable(L, num, 0);
      int index = 1;
      for (std::vector<TBlob>::const_iterator it = begin; it != end; ++it) {
        THGeneralTensor th = TorchTensor::TBlobToTHTensor(*it);
        luaT_pushudata(L, th, TorchTensor::TensorType(*it));
        lua_rawseti(L, -2, index++);
      }
    } else if (num == 0) {
      lua_pushnil(L);
    } else {
      luaT_pushudata(L, TorchTensor::TBlobToTHTensor(*begin), TorchTensor::TensorType(*begin));
    }
  }
};

}  // namespace mxnet
#endif  // PLUGIN_TORCH_TORCH_BASE_H_
