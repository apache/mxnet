/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);

MXNET_OPERATOR_REGISTER_REDUCE_AXES(sum)
.attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.attr<FGradient>("FGradient", [](const nnvm::NodePtr& n,
                                 const std::vector<nnvm::NodeEntry>& ograds) {

  });

NNVM_REGISTER_OP(_backward_sum)
.attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu, )
}  // namespace op
}  // namespace mxnet