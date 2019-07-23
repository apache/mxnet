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
 * \file cuda_utils.cc
 * \brief Common CUDA utilities.
 */

#include <mxnet/base.h>
#include <mshadow/base.h>
#include "cuda_utils.h"

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {

int get_load_type(size_t N) {
  using namespace mshadow;
  if (N % 8 == 0) {
    return kFloat64;
  } else if (N % 4 == 0) {
    return kFloat32;
  } else if (N % 2 == 0) {
    return kFloat16;
  } else {
    return kInt8;
  }
}
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
