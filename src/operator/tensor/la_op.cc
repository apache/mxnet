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

If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   // Single matrix multiply-add
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)
           = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]

   // Batch matrix multiply-add
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
  { return std::vector<std::pair<int, int> >{{2, 1}, {3, 2}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 4, 3, gemm_backward>);

NNVM_REGISTER_OP(_linalg_gemm2)
.add_alias("linalg_gemm2")
.describe(R"code(Performs general matrix multiplication.
Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, the BLAS3 function *gemm* is performed:

   *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)

Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix
transposition (depending on *transpose_a*, *transpose_b*).

If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   // Single matrix multiply
   A = [[1.0, 1.0], [1.0, 1.0]]
   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
   gemm2(A, B, transpose_b=True, alpha=2.0)
            = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

   // Batch matrix multiply
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
  { return std::vector<std::pair<int, int> >{{2, 1}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LaOpBackward<cpu, 2, 2, 3, 2, gemm2_backward>);

NNVM_REGISTER_OP(_linalg_potrf)
.add_alias("linalg_potrf")
.describe(R"code(Performs Cholesky factorization of a symmetric positive-definite matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, the Cholesky factor *L* of the symmetric, positive definite matrix *A* is
computed. *L* is lower triangular (entries of upper triangle are all zero), has
positive diagonal entries, and:

  *A* = *L* \* *L*\ :sup:`T`

If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
(batch mode).

.. note:: The operator supports float32 and float64 data types only.

Examples::

   // Single matrix factorization
   A = [[4.0, 1.0], [1.0, 4.25]]
   potrf(A) = [[2.0, 0], [0.5, 2.0]]

   // Batch matrix factorization
   A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
   potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
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


NNVM_REGISTER_OP(_linalg_potri)
.add_alias("linalg_potri")
.describe(R"code(Performs matrix inversion from a Cholesky factorization.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, *A* is a lower triangular matrix (entries of upper triangle are all zero)
with positive diagonal. We compute:

  *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1`

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

   // Single matrix inverse
   A = [[2.0, 0], [0.5, 2.0]]
   potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]

   // Batch matrix inverse
   A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
   potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
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

NNVM_REGISTER_OP(_linalg_trmm)
.add_alias("linalg_trmm")
.describe(R"code(Performs multiplication with a lower triangular matrix.
Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
on the leading *n-2* dimensions.

If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
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

   // Single triangular matrix multiply
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
   trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

   // Batch triangular matrix multiply
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
.set_attr<nnvm::FInferShape>("FInferShape", LaTriangMatrixMultOpShape)
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

If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
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

   // Single matrix solve
   A = [[1.0, 0], [1.0, 1.0]]
   B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
   trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

   // Batch matrix solve
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

   // Single matrix reduction
   A = [[1.0, 1.0], [1.0, 7.0]]
   sumlogdiag(A) = [1.9459]

   // Batch matrix reduction
   A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
   sumlogdiag(A) = [1.9459, 3.9318]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaReduceShape<2>)
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

   // Single matrix multiply
   A = [[1., 2., 3.], [4., 5., 6.]]
   syrk(A, alpha=1., transpose=False)
            = [[14., 32.],
               [32., 77.]]
   syrk(A, alpha=1., transpose=True)
            = [[17., 22., 27.],
               [22., 29., 36.],
               [27., 36., 45.]]

   // Batch matrix multiply
   A = [[[1., 1.]], [[0.1, 0.1]]]
   syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaSyrkParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", LaSyrkShape)
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

   // Single LQ factorization
   A = [[1., 2., 3.], [4., 5., 6.]]
   Q, L = gelqf(A)
   Q = [[-0.26726124, -0.53452248, -0.80178373],
        [0.87287156, 0.21821789, -0.43643578]]
   L = [[-3.74165739, 0.],
        [-8.55235974, 1.96396101]]

   // Batch LQ factorization
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
.set_attr<nnvm::FInferShape>("FInferShape", LaLQFactShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
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

   // Single symmetric eigendecomposition
   A = [[1., 2.], [2., 4.]]
   U, L = syevd(A)
   U = [[0.89442719, -0.4472136],
        [0.4472136, 0.89442719]]
   L = [0., 5.]

   // Batch symmetric eigendecomposition
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
.set_attr<nnvm::FInferShape>("FInferShape", LaEigFactShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
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

}  // namespace op
}  // namespace mxnet
