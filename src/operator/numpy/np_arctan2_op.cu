#include "./np_arctan2_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_np_arctan2)
.set_attr<FCompute>("FCompute<gpu>", Arctan2OpForward<gpu>);
NNVM_REGISTER_OP(_np_backward_arctan2)
.set_attr<FCompute>("FCompute<gpu>", Arctan2OpBackward<gpu>);

}
}

