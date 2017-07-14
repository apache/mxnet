/*!
 *  Copyright (c) 2017 by Contributors
 * \file box_nms.cu
 * \brief Non-maximum suppression for bounding boxes, GPU version
 * \author Joshua Zhang
 */
#include "./box_nms-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSBackward<gpu>);
}  // namespace op
}  // namespace mxnet
