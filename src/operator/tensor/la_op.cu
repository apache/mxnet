/*!
 * Copyright (c) 2017 by Contributors
 * \file la_op.cu
 * \brief GPU-Operators for advanced linear algebra.
 */
#include "./la_op.h"
#include "./la_op_inline.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(linalg_gemm)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 3, 1, gemm>);

NNVM_REGISTER_OP(_backward_linalg_gemm)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 4, 3, gemm_backward>);

NNVM_REGISTER_OP(linalg_gemm2)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 2, 1, gemm2>);

NNVM_REGISTER_OP(_backward_linalg_gemm2)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 3, 2, gemm2_backward>);

NNVM_REGISTER_OP(linalg_trmm)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 2, 1, trmm>);

NNVM_REGISTER_OP(_backward_linalg_trmm)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 4, 2, trmm_backward>);

NNVM_REGISTER_OP(linalg_trsm)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 2, 1, trsm>);

NNVM_REGISTER_OP(_backward_linalg_trsm)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 4, 2, trsm_backward>);

NNVM_REGISTER_OP(linalg_sumlogdiag)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 0, 1, 1, sumlogdiag>);

NNVM_REGISTER_OP(_backward_linalg_sumlogdiag)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, sumlogdiag_backward>);

NNVM_REGISTER_OP(linalg_potri)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, potri>);

NNVM_REGISTER_OP(_backward_linalg_potri)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 3, 1, potri_backward>);

#if MXNET_USE_CUSOLVER == 1

NNVM_REGISTER_OP(linalg_potrf)
.set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, potrf>);

NNVM_REGISTER_OP(_backward_linalg_potrf)
.set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, potrf_backward>);

#endif

}  // namespace op
}  // namespace mxnet
