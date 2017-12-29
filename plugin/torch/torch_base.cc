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
 * \file torch_base.cc
 * \brief torch_state
 * \author Junyuan Xie
*/
#include "./torch_base.h"

namespace mxnet {
TorchState::TorchState() {
  this->L = luaL_newstate();

  luaL_openlibs(L);
  luaL_loadstring(L,
                  "require 'torch'\n"
                  "require 'nn'\n"
#if MXNET_USE_CUDA
                  "require 'cutorch'\n"
                  "require 'cunn'\n"
#if MXNET_USE_CUDNN
                  "require 'cudnn'\n"
#endif  // MXNET_USE_CUDNN
#endif  // MXNET_USE_CUDA
                  ); // NOLINT(*)
  int err = lua_pcall(L, 0, 0, 0);
  CHECK_EQ(err, 0) << lua_tostring(L, -1);
}

TorchState* TorchState::ThreadSharedLuaState() {
  thread_local TorchState* state = nullptr;
  if (!state) {
    state = new TorchState();
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
  CudaState()->currentStream = mshadow::Stream<gpu>::GetStream(s);
}
#endif  // MXNET_USE_CUDA
}  // namespace mxnet
