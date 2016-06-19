/*!
 *  Copyright (c) 2015 by Contributors
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
    mshadow::Copy(to->FlatTo2D<gpu, DType>(),
                  from.FlatTo2D<cpu, DType>(),
                  static_cast<mshadow::Stream<gpu>*>(ctx.stream));
  });
}

template<>
void Copy<gpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    mshadow::Copy(to->FlatTo2D<cpu, DType>(),
                  from.FlatTo2D<gpu, DType>(),
                  static_cast<mshadow::Stream<gpu>*>(ctx.stream));
  });
}

template<>
void Copy<gpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  if (from_ctx.dev_id == to_ctx.dev_id) {
    mshadow::Stream<gpu>* s = static_cast<mshadow::Stream<gpu>*>(ctx.stream);
    MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
      if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo2D<gpu, DType>(s),
                      from.FlatTo2D<gpu, DType>(s),
                      s);
      } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
          to->FlatTo2D<gpu, DType>(s) =
            mshadow::expr::tcast<DType>(from.FlatTo2D<gpu, SrcDType>(s));
        })
      }
    })
  } else {
    CHECK(from.CheckContiguous() && to->CheckContiguous())
      << "copy across only support continugous memory";
    CHECK_EQ(to->type_flag_, from.type_flag_)
      << "Source and target must have the same data type when copying across devices.";
    mshadow::Stream<gpu> *s = static_cast<mshadow::Stream<gpu>*>(ctx.stream);
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
