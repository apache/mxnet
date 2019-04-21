#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(__name$, __kernel$)     \
NNVM_REGISTER_OP(__name$)                                               \
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, __kernel$>)  \

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_negative, mshadow_op::negation);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_reciprocal, mshadow_op::reciprocal);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_absolute, mshadow_op::abs);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sign, mshadow_op::sign);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_rint, mshadow_op::rint);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_ceil, mshadow_op::ceil);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_floor, mshadow_op::floor);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_trunc, mshadow_op::trunc);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_fix, mshadow_op::fix);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_square, mshadow_op::square);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sqrt, mshadow_op::square_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_cbrt, mshadow_op::cube_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_exp, mshadow_op::exp);

NNVM_REGISTER_OP(_numpy_log)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::LogCompute<gpu, mshadow_op::log>);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log10, mshadow_op::log10);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log2, mshadow_op::log2);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log1p, mshadow_op::log1p);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_expm1, mshadow_op::expm1);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_logical_not, mshadow_op::nt);

}
}
