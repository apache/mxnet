#include "./np_arctan2_op.h"
#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu,Arctan2OpForward>);
NNVM_REGISTER_OP(_backward_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu,Arctan2OpBackward>);

}
}

