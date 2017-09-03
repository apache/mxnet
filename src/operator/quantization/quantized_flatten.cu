/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantized_flatten.cu
 * \brief
 */
#include "./quantized_flatten-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(quantized_flatten)
.set_attr<FCompute>("FCompute<gpu>", QuantizedFlattenCompute<gpu>);

}  // namespace op
}  // namespace mxnet
