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
