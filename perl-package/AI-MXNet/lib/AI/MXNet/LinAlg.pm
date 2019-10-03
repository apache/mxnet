# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

package AI::MXNet::LinAlg;
use strict;
use warnings;
use AI::MXNet::NS;
use AI::MXNet::LinAlg::Symbol;
use AI::MXNet::LinAlg::NDArray;

=head1 NAME

    AI::MXNet::LinAlg - Linear Algebra routines for NDArray and Symbol.
=cut

=head1 DESCRIPTION

    The Linear Algebra API, provides imperative/symbolic linear algebra tensor operations on CPU/GPU.

    mx->linalg-><sym|nd>->gemm  Performs general matrix multiplication and accumulation.
    mx->linalg-><sym|nd>->gemm2 Performs general matrix multiplication.
    mx->linalg-><sym|nd>->potrf Performs Cholesky factorization of a symmetric positive-definite matrix.
    mx->linalg-><sym|nd>->potri Performs matrix inversion from a Cholesky factorization.
    mx->linalg-><sym|nd>->trmm  Performs multiplication with a lower triangular matrix.
    mx->linalg-><sym|nd>->trsm  Solves matrix equation involving a lower triangular matrix.
    mx->linalg-><sym|nd>->sumlogdiag    Computes the sum of the logarithms of the diagonal elements of a square matrix.
    mx->linalg-><sym|nd>->syrk  Multiplication of matrix with its transpose.
    mx->linalg-><sym|nd>->gelqf LQ factorization for general matrix.
    mx->linalg-><sym|nd>->syevd Eigendecomposition for symmetric matrix.
    L<NDArray Python Docs|https://mxnet.apache.org/api/python/ndarray/linalg.html>
    L<Symbol Python Docs|https://mxnet.apache.org/api/python/symbol/linalg.html>

    Examples:

    ## NDArray
    my $A = mx->nd->array([[1.0, 1.0], [1.0, 1.0]]);
    my $B = mx->nd->array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    ok(almost_equal(
        mx->nd->linalg->gemm2($A, $B, transpose_b=>1, alpha=>2.0)->aspdl,
        pdl([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    ));

    ## Symbol
    my $sym_gemm2 = mx->sym->linalg->gemm2(
        mx->sym->var('A'),
        mx->sym->var('B'),
        transpose_b => 1,
        alpha => 2.0
    );
    my $A = mx->nd->array([[1.0, 1.0], [1.0, 1.0]]);
    my $B = mx->nd->array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    ok(almost_equal(
        $sym_gemm2->eval(args => { A => $A, B => $B })->[0]->aspdl,
        pdl([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    ));

=cut

sub sym    { 'AI::MXNet::LinAlg::Symbol'  }
sub symbol { 'AI::MXNet::LinAlg::Symbol'  }
sub nd     { 'AI::MXNet::LinAlg::NDArray' }
sub ndarray { 'AI::MXNet::LinAlg::NDArray' }

1;
