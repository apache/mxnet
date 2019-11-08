#include "./np_diag_op-inl.h"

namespace mxnet {
    namespace op {

        NNVM_REGISTER_OP(_np_diag)
        .set_attr<FCompute>("FCompute<gpu>", NumpyDiagOpForward<gpu>);

        NNVM_REGISTER_OP(_np_backward_diag)
        .set_attr<FCompute>("FCompute<gpu>",  NumpyDiagOpBackward<gpu>);

    }  // namespace op
}  // namespace mxnet
