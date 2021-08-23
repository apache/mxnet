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
 * \file cudnn_algoreg.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./cudnn_algoreg-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include <sstream>
#include <unordered_map>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
template<>
CuDNNAlgoReg<ConvolutionParam> *CuDNNAlgoReg<ConvolutionParam>::Get() {
  static CuDNNAlgoReg<ConvolutionParam> inst;
  return &inst;
}

template<>
CuDNNAlgoReg<DeconvolutionParam> *CuDNNAlgoReg<DeconvolutionParam>::Get() {
  static CuDNNAlgoReg<DeconvolutionParam> inst;
  return &inst;
}
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet
