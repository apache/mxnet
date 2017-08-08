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
 * \file ndarray_function_cpu.cc
 * \brief CPU Implementation of ndarray function.
 */

// this will be invoked by gcc and compile CPU version
#include "./ndarray_function.h"
#include "./ndarray_function-inl.h"

namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo1D<cpu, DType>(),
                      from.FlatTo1D<cpu, DType>());
    } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
            to->FlatTo1D<cpu, DType>() =
                mshadow::expr::tcast<DType>(from.FlatTo1D<cpu, SrcDType>());
        })
    }
  })
}
}  // namespace ndarray
}  // namespace mxnet
