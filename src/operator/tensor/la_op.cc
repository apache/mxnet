/*!
 * Copyright (c) 2017 by Contributors
 * \file la_op.cc
 * \brief CPU-Operators for advanced linear algebra.
 */
#include "./la_op.h"
#include "./la_op_inline.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LaMatrixMacParam);
DMLC_REGISTER_PARAMETER(LaMatrixMultParam);
DMLC_REGISTER_PARAMETER(LaTriangMatrixMultParam);

NNVM_REGISTER_OP(linalg_gemm)
.describe(R"code(Performs general matrix multiplication and accumulation.
Input are three tensors *A*, *B*, *C* each of dimension *n >= 2* and each
having the same shape on the leading *n-2* dimensions. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\ , *B*\ :sub:`i`\ , *C*\ :sub:`i` be the matrices given by the last *2* dimensions.
The operator performs the BLAS3 function *gemm*

   *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ ) + *beta* \* *C*\ :sub:`i`

on all such triples of matrices. Here *alpha* and *beta* are scalar operator parameters and *op()*
is either the identity or the matrix transposition.

In case of *n=2*, a single *gemm* function is performed on the matrices *A*, *B*, *C*.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix multiply-add
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
           = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]

   // Batch matrix multiply-add
   A = [[[1.0, 1.0]], [[0.1, 0.1]]]
   B = [[[1.0, 1.0]], [[0.1, 0.1]]]
   C = [[[10.0]], [[0.01]]]
   linalg_gemm(A, B, C, transpose_b = 1, alpha = 2.0 , beta = 10.0)
           = [[[104.0]], [[0.14]]]
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaMatrixMacParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B", "C"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaMatrixMultMacOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{2, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 3, 1, gemm>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_gemm"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("C", "NDArray-or-Symbol", "Tensor of input matrices")
.add_arguments(LaMatrixMacParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_gemm)
.set_num_inputs(4)
.set_num_outputs(3)
.set_attr_parser(ParamParser<LaMatrixMacParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{1, 0}, {2, 1}, {3, 2}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 3, gemm_backward>);

NNVM_REGISTER_OP(linalg_gemm2)
.describe(R"code(Performs general matrix multiplication.
Input are two tensors *A*, *B* each of dimension *n >= 2* and each
having the same shape on the leading *n-2* dimensions. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2* dimensions.
The operator performs the BLAS3 function *gemm* (restricted to two arguments)

   *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *op*\ (*B*\ :sub:`i`\ )

on all such pairs of matrices. Here *alpha* is a scalar operator parameter and *op()* is either
the identity or the matrix transposition.

In case of *n=2*, a single *gemm* function is performed on the matrices *A*, *B*.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix multiply
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0)
            = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

   // Batch matrix multiply
   A = [[[1.0, 1.0]], [[0.1, 0.1]]]
   B = [[[1.0, 1.0]], [[0.1, 0.1]]]
   linalg_gemm2(A, B, transpose_b = 1, alpha = 2.0 )
           = [[[4.0]], [[0.04 ]]]
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaMatrixMultMacOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 2, 1, gemm2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_gemm2"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of input matrices")
.add_arguments(LaMatrixMultParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_gemm2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LaMatrixMultParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{1, 0}, {2, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 3, 2, gemm2_backward>);

NNVM_REGISTER_OP(linalg_potrf)
.describe(R"code(Performs Cholesky factorization of a symmetric positive-definite matrix.
Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
The operator performs the Cholesky factorization (LAPACK function *potrf*)
on each *A*\ :sub:`i`\ ,
i.e. it computes a lower triangular matrix *U*\ :sub:`i` such that

   *A*\ :sub:`i`\  = *U*\ :sub:`i`\  \* *U*\ :sub:`i`\ \ :sup:`T`

for all such matrices. The matrices *A*\ :sub:`i` must be all symmetric and positive-definite.
The resulting matrices *U*\ :sub:`i` will contain zeros in the upper triangle
apart from the diagonal.

In case of *n=2*, a single Cholesky factorization is performed on the matrix *A*.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix factorization
   A = [[4.0, 1.0], [1.0, 4.25]]
   linalg_potrf(A) = [[2.0, 0], [0.5, 2.0]]

   // Batch matrix factorization
   A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
   linalg_potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, potrf>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_potrf"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices to be decomposed");

NNVM_REGISTER_OP(_backward_linalg_potrf)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 2, 1, potrf_backward>);


NNVM_REGISTER_OP(linalg_potri)
.describe(R"code(Performs matrix inversion from a Cholesky factorization.
Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
The operator assumes that each *A*\ :sub:`i` is the Cholesky factorization of some symmetric
positive-definite matrix *B*\ :sub:`i` given as a lower triangular matrix
(so *A* is the output of a prior call to operator *linalg_potrf*). The operator computes the
inverse of each *B*\ :sub:`i` from this decomposition, i.e

   *out*\ :sub:`i` = *B*\ :sub:`i`\ \ :sup:`-1`

for all such matrices.

In case of *n=2*, the operation is performed on the matrix *A* itself.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix inverse
   A = [[2.0, 0], [0.5, 2.0]]
   linalg_potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]

   // Batch matrix inverse
   A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
   linalg_potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
                  [[0.06641, -0.01562], [-0.01562, 0,0625]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, potri>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_linalg_potri"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of lower triangular matrices");

NNVM_REGISTER_OP(_backward_linalg_potri)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 3, 1, potri_backward>);

NNVM_REGISTER_OP(linalg_trmm)
.describe(R"code(Performs multiplication with a triangular matrix.
Input are two tensors *A*, *B* each of dimension *n >= 2* and each
having the same shape on the leading *n-2* dimensions. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2* dimensions.
The operator performs the BLAS3 function *trmm*

   *out*\ :sub:`i` = *alpha* \* *op*\ (*A*\ :sub:`i`\ ) \* *B*\ :sub:`i`

or

   *out*\ :sub:`i` = *alpha* \* *B*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ )

on all such pairs of matrices. Here *alpha* is a scalar operator parameter,  *op()* is either
the identity or the matrix transposition (depending on the parameter *transpose*) and the
order of matrix multiplication depends on the parameter *rightside*.
All matrices *A*\ :sub:`i` must be lower triangular.

In case of *n=2*, a single *trmm* function is performed on the matrices *A*, *B*.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix multiply
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   linalg_trmm(A, B, alpha = 2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

   // Batch matrix multiply
   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
   B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
   linalg_trmm(A, B, alpha = 2.0 ) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
                                  [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaTriangMatrixMultOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{1, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 2, 1, trmm>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_linalg_trmm"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of lower triangular matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of matrices")
.add_arguments(LaTriangMatrixMultParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_trmm)
.set_num_inputs(4)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 2, trmm_backward>);

NNVM_REGISTER_OP(linalg_trsm)
.describe(R"code(Solves matrix equations involving a triangular matrix. 
Input are two tensors *A*, *B* each of dimension *n >= 2* and each
having the same shape on the leading *n-2* dimensions. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\ , *B*\ :sub:`i`\  be the matrices given by the last *2* dimensions.
The operator performs the BLAS3 function *trsm*, i.e. it solves the equation

   *op*\ (*A*\ :sub:`i`\ ) \* *X*\ :sub:`i` = *alpha* \* *B*\ :sub:`i`

or

   *X*\ :sub:`i` \* *op*\ (*A*\ :sub:`i`\ ) = *alpha* \* *B*\ :sub:`i`

on all such pairs of matrices. Here *alpha* is a scalar operator parameter,  *op()* is either
the identity or the matrix transposition (depending on the parameter *transpose*) and the
order of multiplication on the left depends on the parameter *rightside*.
All matrices *A*\ :sub:`i` must be lower triangular.

In case of *n=2*, a single *trsm* function is performed on the matrices *A*, *B*.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix solve
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
   linalg_trsm(A, B, alpha = 0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

   // Batch matrix solve
   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
   B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
   linalg_trsm(A, B, alpha = 0.5 ) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                  [[2.0, 2.0, 2.0 ], [2.0, 2.0, 2.0]]]
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaTriangMatrixMultOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{1, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 2, 1, trsm>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_linalg_trsm"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of lower triangular matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of matrices")
.add_arguments(LaTriangMatrixMultParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_trsm)
.set_num_inputs(4)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 2, trsm_backward>);

NNVM_REGISTER_OP(linalg_sumlogdiag)
.describe(R"code(Computes the sum of the logarithms of all diagonal elements in a matrix.
Input is a tensor *A* of dimension *n >= 2*. For every *n-2* dimensional index *i* let
*A*\ :sub:`i`\  be the matrix given by the last *2* dimensions.
The operator performs a reduction of each such matrix to a scalar by summing up the logarithms
of all diagonal elements. All matrices must be square and all diagonal elements must be positive.

In case of *n=2*, *A* represents a single matrix on which the reduction will be performed.

.. note:: The operator does only support float32 and float64 data types and provides
          proper backward gradients.

Examples::

   // Single matrix reduction
   A = [[1.0, 1.0], [1.0, 7.0]]
   linalg_sumlogdiag(A) = [1.9459]

   // Batch matrix reduction
   A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
   linalg_sumlogdiag(A) = [1.9459, 3.9318]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaReduceShape<2>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaReduceForward<cpu, 2, sumlogdiag>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_sumlogdiag"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrices");

NNVM_REGISTER_OP(_backward_linalg_sumlogdiag)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaReduceBackward<cpu, 2, sumlogdiag_backward>);

}  // namespace op
}  // namespace mxnet
