/*!
 * Copyright (c) 2015 by Contributors
 * \file cast.cu
 * \brief
 * \author Junyuan Xie
*/
#include <vector>

#include "./cast-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(CastParam param, std::vector<int> *in_type) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH((*in_type)[0], SrcDType, {
    MSHADOW_TYPE_SWITCH(param.dtype, DstDType, {
        op = new CastOp<gpu, SrcDType, DstDType>();
    })
  })
  return op;
}
}  // namespace op
}  // namespace mxnet

