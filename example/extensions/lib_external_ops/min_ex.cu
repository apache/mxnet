#include "./min_ex-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(min_ex)
.set_attr<FCompute>("FCompute<gpu>", MinExForward<gpu>);

}  // namespace op
}  // namespace mxnet
