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
  mshadow::Copy(to->FlatTo2D<gpu, real_t>(),
                from.FlatTo2D<cpu, real_t>(),
                static_cast<mshadow::Stream<gpu>*>(ctx.stream));
}

template<>
void Copy<gpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  mshadow::Copy(to->FlatTo2D<cpu, real_t>(),
                from.FlatTo2D<gpu, real_t>(),
                static_cast<mshadow::Stream<gpu>*>(ctx.stream));
}

template<>
void Copy<gpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  if (from_ctx.dev_id == to_ctx.dev_id) {
     mshadow::Copy(to->FlatTo2D<gpu, real_t>(),
                   from.FlatTo2D<gpu, real_t>(),
                   static_cast<mshadow::Stream<gpu>*>(ctx.stream));
   } else {
     CHECK(from.CheckContiguous() && to->CheckContiguous())
         << "copy across only support continugous memory";
     mshadow::Stream<gpu> *s = static_cast<mshadow::Stream<gpu>*>(ctx.stream);
     CHECK(s != NULL) << "need stream in GPU context";
     cudaMemcpyPeerAsync(to->dptr_,
                         to_ctx.dev_id,
                         from.dptr_,
                         from_ctx.dev_id,
                         from.shape_.Size() * sizeof(real_t),
                         s->stream_);
  }
}
}  // namespace ndarray
}  // namespace mxnet
