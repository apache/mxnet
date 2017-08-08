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
 * \brief GPU Implementation of ndarray function.
 */

// this will be invoked by nvcc and compile GPU version
#include <dmlc/logging.h>
#include "./ndarray_function.h"
#include "./ndarray_function-inl.h"

namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    mshadow::Copy(to->FlatTo1D<gpu, DType>(),
                  from.FlatTo1D<cpu, DType>(),
                  ctx.get_stream<gpu>());
  });
}

template<>
void Copy<gpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    mshadow::Copy(to->FlatTo1D<cpu, DType>(),
                  from.FlatTo1D<gpu, DType>(),
                  ctx.get_stream<gpu>());
  });
}

template<>
void Copy<gpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  if (from_ctx.dev_id == to_ctx.dev_id) {
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
    MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
      if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo1D<gpu, DType>(s),
                      from.FlatTo1D<gpu, DType>(s),
                      s);
      } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
          to->FlatTo1D<gpu, DType>(s) =
            mshadow::expr::tcast<DType>(from.FlatTo1D<gpu, SrcDType>(s));
        })
      }
    })
  } else {
    CHECK(from.CheckContiguous() && to->CheckContiguous())
      << "copy across only support continugous memory";
    CHECK_EQ(to->type_flag_, from.type_flag_)
      << "Source and target must have the same data type when copying across devices.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK(s != NULL) << "need stream in GPU context";
    cudaMemcpyPeerAsync(to->dptr_,
                        to_ctx.dev_id,
                        from.dptr_,
                        from_ctx.dev_id,
                        from.shape_.Size() * mshadow::mshadow_sizeof(to->type_flag_),
                        s->stream_);
  }
}
}  // namespace ndarray
}  // namespace mxnet
