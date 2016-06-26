/*!
 * Copyright (c) 2015 by Contributors
 * \file support_vector_machine.cu
 * \brief
 * \author Jonas Amaro
*/

#include "./support_vector_machine-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SupportVectorMachineParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SupportVectorMachineOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

