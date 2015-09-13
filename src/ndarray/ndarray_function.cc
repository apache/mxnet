/*!
 *  Copyright (c) 2015 by Contributors
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
  mshadow::Copy(to->FlatTo2D<cpu, real_t>(),
                from.FlatTo2D<cpu, real_t>());
}
}  // namespace ndarray
}  // namespace mxnet
