#include "np_around_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_around)
.set_attr<FCompute>("FCompute<gpu>", AroundOpForward<gpu>);

}  // namespace op
}  // namespace mxnet