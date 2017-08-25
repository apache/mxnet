/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cu
 * \brief
 */
#include "./quantize-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_quantize)
.set_attr<FCompute>("FCompute<gpu>", QuantizeCompute<gpu>);

}  // namespace op
}  // namespace mxnet
