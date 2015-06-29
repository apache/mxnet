// this will be invoked by gcc and compile CPU version
#include "./narray_op.h"
#include "./narray_op-inl.h"

namespace mxnet {
namespace narray {
template<>
void Copy<cpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  mshadow::Copy(to->FlatTo2D<cpu, real_t>(),
                from.FlatTo2D<cpu, real_t>());
}
}  // namespace narray
}  // namespace mxnet
