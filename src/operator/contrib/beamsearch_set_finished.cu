#include "./beamsearch_set_finished-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_beamsearch_set_finished)
.set_attr<FCompute>("FCompute<gpu>", BeamsearchSetFinishedForward<gpu>);

NNVM_REGISTER_OP(_contrib_beamsearch_noop_grad)
.set_attr<FCompute>("FCompute<gpu>", NoopGrad<gpu>);
}
}
