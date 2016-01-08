/*!
 * Copyright (c) 2016 by Contributors
 * \file torch_base.cc
 * \brief torch_state
 * \author Junyuan Xie
*/
#include "./torch_base.h"

namespace mxnet {
lua_State* TorchState::LuaState() {
  thread_local lua_State* state = NULL;
  if (!state) {
    state = luaL_newstate();
    luaL_openlibs(state);
    luaL_loadstring(state,
                    "require 'torch'\n"
                    "require 'nn'\n"
#if MXNET_USE_CUDA
                    "require 'cutorch'\n"
                    "require 'cunn'\n"
#if MXNET_USE_CUDNN
                    "require 'cudnn'\n"
#endif  // MXNET_USE_CUDNN
#endif  // MXNET_USE_CUDA
                    "local ss = require 'threads.sharedserialize'\n"
                    "Serialize, Deserialize = ss.save, ss.load\n");
    int err = lua_pcall(state, 0, 0, 0);
    CHECK_EQ(err, 0) << lua_tostring(state, -1);
  }
  return state;
}

template<>
void TorchState::SetStream(mshadow::Stream<mshadow::cpu>* s) {
  return;
}

#if MXNET_USE_CUDA
template<>
void TorchState::SetStream(mshadow::Stream<mshadow::gpu>* s) {
  TorchState::CudaState()->currentStream = mshadow::Stream<gpu>::GetStream(s);
}
#endif  // MXNET_USE_CUDA
}  // namespace mxnet
