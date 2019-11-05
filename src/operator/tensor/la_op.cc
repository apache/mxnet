/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file la_op.cc
 * \brief CPU implementation of Operators for advanced linear algebra.
 */

#include "./la_op.h"
#include "./la_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LaMatrixMacParam);
DMLC_REGISTER_PARAMETER(LaMatrixMultParam);
DMLC_REGISTER_PARAMETER(LaCholeskyParam);
DMLC_REGISTER_PARAMETER(LaTriangMatrixMultParam);
DMLC_REGISTER_PARAMETER(LaDiagParam);
DMLC_REGISTER_PARAMETER(LaTrianParam);
DMLC_REGISTER_PARAMETER(LaSyrkParam);

NNVM_REGISTER_OP(_linalg_gemm)
.add_alias("linalg_gemm")
.describe(R"code(Performs general matrix multiplication and accumulation.
Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, the BLAS3 function *gemm* is performed:

   *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*

Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or
matrix transposition (depending on *transpose_a*, *transpose_b*).

If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices
are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis*
parameter. By default, the trailing two dimensions will be used for matrix encoding.

For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes
calls. For example let *A*, *B*, *C* be 5 dimensional tensors. Then gemm(*A*, *B*, *C*, axis=1) is equivalent
to the following without the overhead of the additional swapaxis operations::

    A1 = swapaxes(A, dim1=1, dim2=3)
    B1 = swapaxes(B, dim1=1, dim2=3)
    C = swapaxes(C, dim1=1, dim2=3)
    C = gemm(A1, B1, C)
    C = swapaxis(C, dim1=1, dim2=3)

When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix multiply-add
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)
           = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]

   Batch matrix multiply-add
   A = [[[1.0, 1.0]], [[0.1, 0.1]]]
   B = [[[1.0, 1.0]], [[0.1, 0.1]]]
   C = [[[10.0]], [[0.01]]]
   gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)
           = [[[104.0]], [[0.14]]]
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaMatrixMacParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B", "C"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaMatrixMultMacOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{2, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpGemmForward<cpu, 2, 2, 3, 1, gemm>)
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
  { return std::vector<std::pair<int, int> >{{2, 1}, {3, 2}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpGemmBackward<cpu, 2, 2, 4, 3, gemm_backward>);

NNVM_REGISTER_OP(_linalg_gemm2)
.add_alias("linalg_gemm2")
.describe(R"code(Performs general matrix multiplication.
Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, the BLAS3 function *gemm* is performed:

   *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)

Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix
transposition (depending on *transpose_a*, *transpose_b*).

If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices
are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis*
parameter. By default, the trailing two dimensions will be used for matrix encoding.

For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes
calls. For example let *A*, *B* be 5 dimensional tensors. Then gemm(*A*, *B*, axis=1) is equivalent to
the following without the overhead of the additional swapaxis operations::

    A1 = swapaxes(A, dim1=1, dim2=3)
    B1 = swapaxes(B, dim1=1, dim2=3)
    C = gemm2(A1, B1)
    C = swapaxis(C, dim1=1, dim2=3)

When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix multiply
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   gemm2(A, B, transpose_b=True, alpha=2.0)
            = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

   Batch matrix multiply
   A = [[[1.0, 1.0]], [[0.1, 0.1]]]
   B = [[[1.0, 1.0]], [[0.1, 0.1]]]
   gemm2(A, B, transpose_b=True, alpha=2.0)
           = [[[4.0]], [[0.04 ]]]
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaMatrixMultMacOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpGemmForward<cpu, 2, 2, 2, 1, gemm2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_gemm2"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of input matrices")
.add_arguments(LaMatrixMultParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_gemm2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LaMatrixMultParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{2, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpGemmBackward<cpu, 2, 2, 3, 2, gemm2_backward>);

NNVM_REGISTER_OP(_linalg_potrf)
.add_alias("linalg_potrf")
.describe(R"code(Performs Cholesky factorization of a symmetric positive-definite matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, the Cholesky factor *B* of the symmetric, positive definite matrix *A* is
computed. *B* is triangular (entries of upper or lower triangle are all zero), has
positive diagonal entries, and:

  *A* = *B* \* *B*\ :sup:`T`  if *lower* = *true*
  *A* = *B*\ :sup:`T` \* *B*  if *lower* = *false*

If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix factorization
   A = [[4.0, 1.0], [1.0, 4.25]]
   potrf(A) = [[2.0, 0], [0.5, 2.0]]

   Batch matrix factorization
   A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
   potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaCholeskyParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, potrf>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_potrf"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices to be decomposed");

NNVM_REGISTER_OP(_backward_linalg_potrf)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaCholeskyParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 2, 1, potrf_backward>);


NNVM_REGISTER_OP(_linalg_potri)
.add_alias("linalg_potri")
.describe(R"code(Performs matrix inversion from a Cholesky factorization.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* is a triangular matrix (entries of upper or lower triangle are all zero)
with positive diagonal. We compute:

  *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1` if *lower* = *true*
  *out* = *A*\ :sup:`-1` \* *A*\ :sup:`-T` if *lower* = *false*

In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix
*B* (obtained by *potrf*), then

  *out* = *B*\ :sup:`-1`

If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

.. note:: Use this operator only if you are certain you need the inverse of *B*, and
          cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution
          (*trsm*). The latter is numerically much safer, and also cheaper.

Examples::

   Single matrix inverse
   A = [[2.0, 0], [0.5, 2.0]]
   potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]

   Batch matrix inverse
   A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
   potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
               [[0.06641, -0.01562], [-0.01562, 0,0625]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaCholeskyParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, potri>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_linalg_potri"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of lower triangular matrices");

NNVM_REGISTER_OP(_backward_linalg_potri)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaCholeskyParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 3, 1, potri_backward>);

NNVM_REGISTER_OP(_linalg_trmm)
.add_alias("linalg_trmm")
.describe(R"code(Performs multiplication with a lower triangular matrix.
Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, *A* must be triangular. The operator performs the BLAS3 function
*trmm*:

   *out* = *alpha* \* *op*\ (*A*) \* *B*

if *rightside=False*, or

   *out* = *alpha* \* *B* \* *op*\ (*A*)

if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
identity or the matrix transposition (depending on *transpose*).

If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single triangular matrix multiply
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

   Batch triangular matrix multiply
   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
   B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
   trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
                            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaTriangMatrixMultOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{1, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 2, 1, trmm>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_trmm"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of lower triangular matrices")
.add_argument("B", "NDArray-or-Symbol", "Tensor of matrices")
.add_arguments(LaTriangMatrixMultParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_trmm)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 3, 2, trmm_backward>);

NNVM_REGISTER_OP(_linalg_trsm)
.add_alias("linalg_trsm")
.describe(R"code(Solves matrix equation involving a lower triangular matrix.
Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, *A* must be triangular. The operator performs the BLAS3 function
*trsm*, solving for *out* in:

   *op*\ (*A*) \* *out* = *alpha* \* *B*

if *rightside=False*, or

   *out* \* *op*\ (*A*) = *alpha* \* *B*

if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
identity or the matrix transposition (depending on *transpose*).

If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix solve
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
   trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

   Batch matrix solve
   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
   B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
   trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTriangMatrixMultParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A", "B"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaTriangMatrixMultOpShape)
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
  { return std::vector<std::pair<int, int> >{{0, 1}, {1, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 2, trsm_backward>);

NNVM_REGISTER_OP(_linalg_sumlogdiag)
.add_alias("linalg_sumlogdiag")
.describe(R"code(Computes the sum of the logarithms of the diagonal elements of a square matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* must be square with positive diagonal entries. We sum the natural
logarithms of the diagonal elements, the result has shape (1,).

If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all
inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix reduction
   A = [[1.0, 1.0], [1.0, 7.0]]
   sumlogdiag(A) = [1.9459]

   Batch matrix reduction
   A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
   sumlogdiag(A) = [1.9459, 3.9318]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaReduceShape<2>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 0, 1, 1, sumlogdiag>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_sumlogdiag"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrices");

NNVM_REGISTER_OP(_backward_linalg_sumlogdiag)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{1, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 2, 1, sumlogdiag_backward>);

NNVM_REGISTER_OP(_linalg_extractdiag)
.add_alias("linalg_extractdiag")
.describe(R"code(Extracts the diagonal entries of a square matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, then *A* represents a single square matrix which diagonal elements get extracted as a 1-dimensional tensor.

If *n>2*, then *A* represents a batch of square matrices on the trailing two dimensions. The extracted diagonals are returned as an *n-1*-dimensional tensor.

.. note:: The operator supports float32 and float64 data types only.

Examples::

    Single matrix diagonal extraction
    A = [[1.0, 2.0],
         [3.0, 4.0]]

    extractdiag(A) = [1.0, 4.0]

    extractdiag(A, 1) = [2.0]

    Batch matrix diagonal extraction
    A = [[[1.0, 2.0],
          [3.0, 4.0]],
         [[5.0, 6.0],
          [7.0, 8.0]]]

    extractdiag(A) = [[1.0, 4.0],
                      [5.0, 8.0]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaDiagParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaDiagTrianShape<true, true>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 1, 1, 1, copydiag>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_linalg_extractdiag"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrices")
.add_arguments(LaDiagParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_extractdiag)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaDiagParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 1, 2, 1, 1, copydiag>);

NNVM_REGISTER_OP(_linalg_makediag)
.add_alias("linalg_makediag")
.describe(R"code(Constructs a square matrix with the input as diagonal.
Input is a tensor *A* of dimension *n >= 1*.

If *n=1*, then *A* represents the diagonal entries of a single square matrix. This matrix will be returned as a 2-dimensional tensor.
If *n>1*, then *A* represents a batch of diagonals of square matrices. The batch of diagonal matrices will be returned as an *n+1*-dimensional tensor.

.. note:: The operator supports float32 and float64 data types only.

Examples::

    Single diagonal matrix construction
    A = [1.0, 2.0]

    makediag(A)    = [[1.0, 0.0],
                      [0.0, 2.0]]

    makediag(A, 1) = [[0.0, 1.0, 0.0],
                      [0.0, 0.0, 2.0],
                      [0.0, 0.0, 0.0]]

    Batch diagonal matrix construction
    A = [[1.0, 2.0],
         [3.0, 4.0]]

    makediag(A) = [[[1.0, 0.0],
                    [0.0, 2.0]],
                   [[3.0, 0.0],
                    [0.0, 4.0]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaDiagParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaDiagTrianShape<true, false>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 1, 2, 1, 1, copydiag>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_linalg_makediag"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of diagonal entries")
.add_arguments(LaDiagParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_makediag)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaDiagParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 1, 1, 1, copydiag>);

NNVM_REGISTER_OP(_linalg_extracttrian)
.add_alias("linalg_extracttrian")
.describe(R"code(Extracts a triangular sub-matrix from a square matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, then *A* represents a single square matrix from which a triangular sub-matrix is extracted as a 1-dimensional tensor.

If *n>2*, then *A* represents a batch of square matrices on the trailing two dimensions. The extracted triangular sub-matrices are returned as an *n-1*-dimensional tensor.

The *offset* and *lower* parameters determine the triangle to be extracted:

- When *offset = 0* either the lower or upper triangle with respect to the main diagonal is extracted depending on the value of parameter *lower*.
- When *offset = k > 0* the upper triangle with respect to the k-th diagonal above the main diagonal is extracted. 
- When *offset = k < 0* the lower triangle with respect to the k-th diagonal below the main diagonal is extracted. 

.. note:: The operator supports float32 and float64 data types only.

Examples::

    Single triagonal extraction
    A = [[1.0, 2.0],
         [3.0, 4.0]]

    extracttrian(A) = [1.0, 3.0, 4.0]
    extracttrian(A, lower=False) = [1.0, 2.0, 4.0]
    extracttrian(A, 1) = [2.0]
    extracttrian(A, -1) = [3.0]

    Batch triagonal extraction
    A = [[[1.0, 2.0],
          [3.0, 4.0]],
         [[5.0, 6.0],
          [7.0, 8.0]]]

    extracttrian(A) = [[1.0, 3.0, 4.0],
                       [5.0, 7.0, 8.0]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTrianParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaDiagTrianShape<false, true>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 1, 1, 1, copytrian>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_linalg_extracttrian"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrices")
.add_arguments(LaTrianParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_extracttrian)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTrianParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 1, 2, 1, 1, copytrian>);

NNVM_REGISTER_OP(_linalg_maketrian)
.add_alias("linalg_maketrian")
.describe(R"code(Constructs a square matrix with the input representing a specific triangular sub-matrix.
This is basically the inverse of *linalg.extracttrian*. Input is a tensor *A* of dimension *n >= 1*.

If *n=1*, then *A* represents the entries of a triangular matrix which is lower triangular if *offset<0* or *offset=0*, *lower=true*. The resulting matrix is derived by first constructing the square
matrix with the entries outside the triangle set to zero and then adding *offset*-times an additional 
diagonal with zero entries to the square matrix. 

If *n>1*, then *A* represents a batch of triangular sub-matrices. The batch of corresponding square matrices is returned as an *n+1*-dimensional tensor.

.. note:: The operator supports float32 and float64 data types only.

Examples::

    Single  matrix construction
    A = [1.0, 2.0, 3.0]

    maketrian(A)              = [[1.0, 0.0],
                                 [2.0, 3.0]]

    maketrian(A, lower=false) = [[1.0, 2.0],
                                 [0.0, 3.0]]

    maketrian(A, offset=1)    = [[0.0, 1.0, 2.0],
                                 [0.0, 0.0, 3.0],
                                 [0.0, 0.0, 0.0]]
    maketrian(A, offset=-1)   = [[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 3.0, 0.0]]

    Batch matrix construction
    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

    maketrian(A)           = [[[1.0, 0.0],
                               [2.0, 3.0]],
                              [[4.0, 0.0],
                               [5.0, 6.0]]]

    maketrian(A, offset=1) = [[[0.0, 1.0, 2.0],
                               [0.0, 0.0, 3.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 4.0, 5.0],
                               [0.0, 0.0, 6.0],
                               [0.0, 0.0, 0.0]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTrianParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaDiagTrianShape<false, false>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 1, 2, 1, 1, copytrian>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_linalg_maketrian"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of triangular matrices stored as vectors")
.add_arguments(LaTrianParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_maketrian)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaTrianParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 1, 1, 1, copytrian>);

NNVM_REGISTER_OP(_linalg_syrk)
.add_alias("linalg_syrk")
.describe(R"code(Multiplication of matrix with its transpose.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, the operator performs the BLAS3 function *syrk*:

  *out* = *alpha* \* *A* \* *A*\ :sup:`T`

if *transpose=False*, or

  *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*

if *transpose=True*.

If *n>2*, *syrk* is performed separately on the trailing two dimensions for all
inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix multiply
   A = [[1., 2., 3.], [4., 5., 6.]]
   syrk(A, alpha=1., transpose=False)
            = [[14., 32.],
               [32., 77.]]
   syrk(A, alpha=1., transpose=True)
            = [[17., 22., 27.],
               [22., 29., 36.],
               [27., 36., 45.]]

   Batch matrix multiply
   A = [[[1., 1.]], [[0.1, 0.1]]]
   syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaSyrkParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaSyrkShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, syrk>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_linalg_syrk"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices")
.add_arguments(LaSyrkParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linalg_syrk)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaSyrkParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 2, 1, syrk_backward>);

NNVM_REGISTER_OP(_linalg_gelqf)
.add_alias("linalg_gelqf")
.describe(R"code(LQ factorization for general matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*
must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ
factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so
that:

   *A* = *L* \* *Q*

Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,
and *Q* is row-orthonormal, meaning that

   *Q* \* *Q*\ :sup:`T`

is equal to the identity matrix of shape *(x, x)*.

If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all
inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single LQ factorization
   A = [[1., 2., 3.], [4., 5., 6.]]
   Q, L = gelqf(A)
   Q = [[-0.26726124, -0.53452248, -0.80178373],
        [0.87287156, 0.21821789, -0.43643578]]
   L = [[-3.74165739, 0.],
        [-8.55235974, 1.96396101]]

   Batch LQ factorization
   A = [[[1., 2., 3.], [4., 5., 6.]],
        [[7., 8., 9.], [10., 11., 12.]]]
   Q, L = gelqf(A)
   Q = [[[-0.26726124, -0.53452248, -0.80178373],
         [0.87287156, 0.21821789, -0.43643578]],
        [[-0.50257071, -0.57436653, -0.64616234],
         [0.7620735, 0.05862104, -0.64483142]]]
   L = [[[-3.74165739, 0.],
         [-8.55235974, 1.96396101]],
        [[-13.92838828, 0.],
         [-19.09768702, 0.52758934]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaLQFactShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 2, gelqf>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_gelqf"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices to be factorized");

NNVM_REGISTER_OP(_backward_linalg_gelqf)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 1, gelqf_backward>);

NNVM_REGISTER_OP(_linalg_syevd)
.describe(R"code(Eigendecomposition for symmetric matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* must be symmetric, of shape *(x, x)*. We compute the eigendecomposition,
resulting in the orthonormal matrix *U* of eigenvectors, shape *(x, x)*, and the
vector *L* of eigenvalues, shape *(x,)*, so that:

   *U* \* *A* = *diag(L)* \* *U*

Here:

   *U* \* *U*\ :sup:`T` = *U*\ :sup:`T` \* *U* = *I*

where *I* is the identity matrix. Also, *L(0) <= L(1) <= L(2) <= ...* (ascending order).

If *n>2*, *syevd* is performed separately on the trailing two dimensions of *A* (batch
mode). In this case, *U* has *n* dimensions like *A*, and *L* has *n-1* dimensions.

.. note:: The operator supports float32 and float64 data types only.

.. note:: Derivatives for this operator are defined only if *A* is such that all its
          eigenvalues are distinct, and the eigengaps are not too small. If you need
          gradients, do not apply this operator to matrices with multiple eigenvalues.

Examples::

   Single symmetric eigendecomposition
   A = [[1., 2.], [2., 4.]]
   U, L = syevd(A)
   U = [[0.89442719, -0.4472136],
        [0.4472136, 0.89442719]]
   L = [0., 5.]

   Batch symmetric eigendecomposition
   A = [[[1., 2.], [2., 4.]],
        [[1., 2.], [2., 5.]]]
   U, L = syevd(A)
   U = [[[0.89442719, -0.4472136],
         [0.4472136, 0.89442719]],
        [[0.92387953, -0.38268343],
         [0.38268343, 0.92387953]]]
   L = [[0., 5.],
        [0.17157288, 5.82842712]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", LaEigFactShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpForwSyevd<cpu, syevd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_syevd"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices to be factorized");

NNVM_REGISTER_OP(_backward_linalg_syevd)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackwSyevd<cpu, syevd_backward>);

NNVM_REGISTER_OP(_linalg_inverse)
.add_alias("linalg_inverse")
.describe(R"code(Compute the inverse of a matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* is a square matrix. We compute:

  *out* = *A*\ :sup:`-1`

If *n>2*, *inverse* is performed separately on the trailing two dimensions
for all inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   Single matrix inverse
   A = [[1., 4.], [2., 3.]]
   inverse(A) = [[-0.6, 0.8], [0.4, -0.2]]

   Batch matrix inverse
   A = [[[1., 4.], [2., 3.]],
        [[1., 3.], [2., 4.]]]
   inverse(A) = [[[-0.6, 0.8], [0.4, -0.2]],
                 [[-2., 1.5], [1., -0.5]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", InverseShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, inverse>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_inverse"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix");

NNVM_REGISTER_OP(_backward_linalg_inverse)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int> >{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 2, 1, inverse_backward>);

NNVM_REGISTER_OP(_linalg_det)
.add_alias("linalg_det")
.describe(R"code(Compute the determinant of a matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* is a square matrix. We compute:

  *out* = *det(A)*

If *n>2*, *det* is performed separately on the trailing two dimensions
for all inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.
.. note:: There is no gradient backwarded when A is non-invertible (which is
          equivalent to det(A) = 0) because zero is rarely hit upon in float
          point computation and the Jacobi's formula on determinant gradient
          is not computationally efficient when A is non-invertible.

Examples::

   Single matrix determinant
   A = [[1., 4.], [2., 3.]]
   det(A) = [-5.]

   Batch matrix determinant
   A = [[[1., 4.], [2., 3.]],
        [[2., 3.], [1., 4.]]]
   det(A) = [-5., 5.]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(3)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
  return 1; })
.set_attr<mxnet::FInferShape>("FInferShape", DetShape<1>)
.set_attr<nnvm::FInferType>("FInferType", DetType<1>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpDetForward<cpu, 1, det>)
.set_attr<nnvm::FGradient>("FGradient", ReduceDetGrad<1>{"_backward_linalg_det"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix");

NNVM_REGISTER_OP(_backward_linalg_det)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpDetBackward<cpu, 1, det_backward>);

NNVM_REGISTER_OP(_linalg_slogdet)
.add_alias("linalg_slogdet")
.describe(R"code(Compute the sign and log of the determinant of a matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* is a square matrix. We compute:

  *sign* = *sign(det(A))*
  *logabsdet* = *log(abs(det(A)))*

If *n>2*, *slogdet* is performed separately on the trailing two dimensions
for all inputs (batch mode).

.. note:: The operator supports float32 and float64 data types only.
.. note:: The gradient is not properly defined on sign, so the gradient of
          it is not backwarded.
.. note:: No gradient is backwarded when A is non-invertible. Please see
          the docs of operator det for detail.

Examples::

   Single matrix signed log determinant
   A = [[2., 3.], [1., 4.]]
   sign, logabsdet = slogdet(A)
   sign = [1.]
   logabsdet = [1.609438]

   Batch matrix signed log determinant
   A = [[[2., 3.], [1., 4.]],
        [[1., 2.], [2., 4.]],
        [[1., 2.], [4., 3.]]]
   sign, logabsdet = slogdet(A)
   sign = [1., 0., -1.]
   logabsdet = [1.609438, -inf, 1.609438]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(4)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
  return 2; })
.set_attr<mxnet::FInferShape>("FInferShape", DetShape<2>)
.set_attr<nnvm::FInferType>("FInferType", DetType<2>)
.set_attr<FCompute>("FCompute<cpu>", LaOpDetForward<cpu, 2, slogdet>)
.set_attr<nnvm::FGradient>("FGradient", ReduceDetGrad<2>{"_backward_linalg_slogdet"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix");

NNVM_REGISTER_OP(_backward_linalg_slogdet)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpDetBackward<cpu, 2, slogdet_backward>);

}  // namespace op
}  // namespace mxnet
