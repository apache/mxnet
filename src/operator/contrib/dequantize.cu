/*!
 *  Copyright (c) 2017 by Contributors
 * \file dequantize.cu
 * \brief
 */
#include "./dequantize-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_dequantize)
.set_attr<FCompute>("FCompute<gpu>", DequantizeCompute<gpu>);

}  // namespace op
}  // namespace mxnet
