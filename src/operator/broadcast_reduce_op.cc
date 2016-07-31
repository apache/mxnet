/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast reduce op
 */
// this will be invoked by gcc and compile CPU version
#include "./broadcast_reduce_op-inl.h"
namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);

}  // namespace op
}  // namespace mxnet
