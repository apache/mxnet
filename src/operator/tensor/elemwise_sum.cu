/*!
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.cu
 * \brief elementwise sum operator
*/
#include "./elemwise_sum.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(ElementWiseSum)
.set_attr<FCompute>("FCompute<gpu>", ElementWiseSumCompute<gpu>);

}  // namespace op
}  // namespace mxnet
