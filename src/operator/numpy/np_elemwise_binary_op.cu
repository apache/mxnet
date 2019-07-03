#include "./np_arctan2_op.h"
#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_np_arctan2)
.set_attr<FCompute>("FCompute<gpu>", Arctan2OpForward<gpu>);
NNVM_REGISTER_OP(_backward_np_arctan2)
.set_attr<FCompute>("FCompute<gpu>", Arctan2OpBackward<gpu>);

}
}

