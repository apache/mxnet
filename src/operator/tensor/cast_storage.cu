/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage.cu
 * \brief GPU Implementation of cast_storage operator.
 */
#include "./cast_storage-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(cast_storage)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", CastStorageComputeEx<gpu>);

}  // namespace op
}  // namespace mxnet
