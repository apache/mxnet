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
 * Copyright (c) 2020 by Contributors
 * \file cuda_rtc.h
 * \brief Common CUDA utilities for
 *        runtime compilation.
 */

#ifndef MXNET_COMMON_CUDA_RTC_H_
#define MXNET_COMMON_CUDA_RTC_H_

#include "mxnet/base.h"
#include "mxnet/op_attr_types.h"

#if MXNET_USE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <string>
#include <vector>

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

namespace util {

/*! \brief Convert OpReqType to string.
 *  \param req to convert
 */
std::string to_string(OpReqType req);

}  // namespace util

int GetMaxSupportedArch();

extern std::mutex lock;

/*! \brief Compile and get the GPU kernel. Uses cache in order to
 *         eliminate the overhead of compilation.
 *  \param parameters of the kernel (e.g. values of the template arguments, types used)
 *  \param kernel_name name of the kernel
 *  \param code used for compilation of the kernel if not found in cache
 *  \param dev_id id of the device which the kernel will be launched on
 */
CUfunction get_function(const std::string &parameters,
                        const std::string &kernel_name,
                        const std::string &code,
                        int dev_id);

/*! \brief Launch a GPU kernel.
 *  \param function to launch
 *  \param grid_dim grid dimensions
 *  \param block_dim block dimensions
 *  \param shared_mem_bytes amount of dynamic shared memory needed by the kernel
 *  \param stream used for launching the kernel
 *  \param args arguments of the kernel
 */
void launch(CUfunction function,
            const dim3 grid_dim,
            const dim3 block_dim,
            unsigned int shared_mem_bytes,
            mshadow::Stream<gpu> *stream,
            std::vector<const void*> *args);

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_H_
