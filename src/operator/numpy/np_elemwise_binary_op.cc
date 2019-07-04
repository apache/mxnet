#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_logaddexp2)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"x1", "x2"};
})
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::logaddexp2>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryBroadcastComputeSparseEx<cpu, mshadow_op::logaddexp2>)
.set_attr<FInferStorageType>("FInferStorageType", BinaryBroadcastMulStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_np_logaddexp2"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  })
.add_argument("x1", "NDArray-or-Symbol", "Input ndarray")
.add_argument("x2", "NDArray-or-Symbol", "Input ndarray");


NNVM_REGISTER_OP(_backward_np_logaddexp2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu,   \
                                     mshadow_op::logaddexp2_grad_left,   \
                                     mshadow_op::logaddexp2_grad_right>);
}  // namespace op
}  // namespace mxnet