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
 *  Copyright (c) 2017 by Contributors
 *  \file krprod.h
 *  \brief Core function for Khatri-Rao product
 *  \author Jencir Lee, Chris Swierczewski
 */
#ifndef MXNET_OPERATOR_CONTRIB_KRPROD_H_
#define MXNET_OPERATOR_CONTRIB_KRPROD_H_
#include <algorithm>
#include <utility>
#include <vector>
#include "mshadow/tensor.h"
#include "../operator_common.h"
#include "../c_lapack_api.h"


namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;

/*!
 * \brief Computes row-wise Kronecker product
 *
 * Given input matrices, this function computes the Kronecker product
 * row-wise. E.g. if the input matrices  are of shape (3, 2), (3, 4),
 * (3, 5), the result matrix will be of shape (3, 2 * 4 * 5), which is
 * (3, 40).
 *
 * \param out result matrix
 * \param ts_arr vector of input matrices
 */
template <typename DType>
inline void row_wise_kronecker
  (Tensor<cpu, 2, DType> out,
  const std::vector<Tensor<cpu, 2, DType> > &ts_arr) {
  CHECK_GE(ts_arr.size(), 1) << "The input matrices must be non-empty.";

  // Check all input and output matrices have the same number of rows
  // and the output matrix has the right number of columns
  int nrows = static_cast<int>(out.size(0));
  int ncols = 1;
  for (auto & ts : ts_arr) {
    CHECK_EQ(nrows, static_cast<int>(ts.size(0)))
      << "All input and output matrices must have the same number of rows.";
    ncols *= ts.size(1);
  }
  CHECK_EQ(ncols, static_cast<int>(out.size(1)));

  // Create an intermediate space of the same shape as out
  //
  // Suppose storage stores the result at step i-1, we'd
  // compute and store the result into out for step i;
  // we then proceed to compute and store the result in storage
  // for step i+1 and so on and so forth, by alternating using
  // storage and out to store the given variable and the result variable
  Tensor<cpu, 2, DType> storage(out.shape_);
  AllocSpace(&storage);

  // Pointers to the given variable and result variable
  // We exchange what given and result point to at every step
  Tensor<cpu, 2, DType> *given = &storage,
    *result = &out, *tmp;

  // Compute each intermediate row-wise Kronecker product
  storage = 1;
  ncols = 1;
  for (auto & ts : ts_arr) {
    expr::BLASEngine<cpu, DType>::SetStream
      (result->stream_);

    // Compute the current row-wise Kronecker product
    *result = 0;
    for (int i = 0; i < nrows; ++i) {
      // BLAS signature
      //
      // dger(
      //   m : ts.size(1), length of each row of current matrix
      //   n : ncols, length of each row of previous result
      //   alpha : 1, scaling to the outer product of x and y
      //   x : ts[i].dptr_, current row of current matrix
      //   incx : 1, as each element in the row is contiguous
      //   y : (*given)[i].dptr_, current row of the given variable
      //   incy : 1, as each element in the row is contiguous
      //   a : (*result)[i].dptr_, current row of the result variable
      //   lda : ts.size(1), as the outer product is stored as one row
      //         which occupies contiguous memory, and as BLASEngine::ger()
      //         assumes column-major matrix, lda has to be precisely
      //         the length of x, i.e. ts[i].size(1)
      // )
      expr::BLASEngine<cpu, DType>::ger
        (result->stream_,
        ts.size(1), ncols, 1,
        ts[i].dptr_, 1,
        (*given)[i].dptr_, 1,
        (*result)[i].dptr_, ts.size(1));
    }
    ncols *= ts.size(1);

    tmp = given;
    given = result;
    result = tmp;
  }

  // If the final result is stored in storage,
  // copy its value to out
  if (given != &out)
    Copy(out, storage);

  FreeSpace(&storage);
}

/*!
 * \brief Khatri-Rao product
 *
 * \param out result matrix
 * \param ts_arr vector of input matrices
 */
template <typename DType>
inline void khatri_rao
  (Tensor<cpu, 2, DType> out,
  const std::vector<Tensor<cpu, 2, DType> > &ts_arr) {
  CHECK_GE(ts_arr.size(), 1) << "The input matrices must be non-empty.";

  // Check all input and output matrices have the same number
  // of columns and the output matrix has the right number of rows
  int ncols = static_cast<int>(out.size(1));
  int nrows = 1;
  for (auto & ts : ts_arr) {
    CHECK_EQ(ncols, static_cast<int>(ts.size(1)))
      << "All input and output matrices must have the same number of columns.";
    nrows *= ts.size(0);
  }
  CHECK_EQ(nrows, static_cast<int>(out.size(0)));

  // Change the layout of matrices to column-major
  Tensor<cpu, 2, DType> out_t(Shape2(out.size(1), out.size(0)));
  AllocSpace(&out_t);
  flip<cpu, DType>(out.size(0), out.size(1), out_t.dptr_, out_t.stride_,
    out.dptr_, out.stride_);

  std::vector<Tensor<cpu, 2, DType> > ts_t_arr;
  for (int i = 0; i < static_cast<int>(ts_arr.size()); ++i) {
    ts_t_arr.emplace_back(Shape2(ts_arr[i].size(1), ts_arr[i].size(0)));
    AllocSpace(&ts_t_arr[i]);
    flip<cpu, DType>(ts_arr[i].size(0), ts_arr[i].size(1), ts_t_arr[i].dptr_,
      ts_t_arr[i].stride_, ts_arr[i].dptr_, ts_arr[i].stride_);
  }

  // Perform row-wise Kronecker product
  row_wise_kronecker(out_t, ts_t_arr);

  // Change the layout of result matrix back to row-major
  flip<cpu, DType>(out.size(1), out.size(0), out.dptr_, out.stride_,
    out_t.dptr_, out_t.stride_);

  FreeSpace(&out_t);
  for (auto &ts_t : ts_t_arr)
    FreeSpace(&ts_t);
}

/*!
 * \brief Moore-Penrose pseudoinverse of the Khatri-Rao product
 *
 * Given input matrices A_1, ..., A_n, of shape (l_1, k), ..., (l_n, k) respectively, the pseudoinverse of the Khatri-Rao product is
 *
 *   pinv(A_1 khatri-rao A_2 khatri-rao ... khatri-rao A_n) =
 *     ((A_1^T A_1) hadamard-dot ... hadamard-dot (A_n^T A_n))
 *     (A_1 khatri-rao ... khatri-rao A_n)^T
 *
 * As the first term of the r.h.s is a square matrix, the result is always of the same shape as the transpose of the Khatri-Rao product of the input matrices. The input argument ts_arr could contain the original input matrices, or transposed ones.
 *
 * \param out result matrix
 * \param ts_arr vector of input matrices
 * \param input_transposed if every input matrices is transposed
 */
template <typename DType>
inline void inv_khatri_rao
  (Tensor<cpu, 2, DType> out,
  const std::vector<Tensor<cpu, 2, DType> > &ts_arr,
  bool input_transposed = false) {
  CHECK_GE(ts_arr.size(), 1) << "Input tensor array must be non-empty";

  // Initialise the Hadamard product to eye(k)
  // where k is the number of "factors"
  int k = out.size(0);
  Tensor<cpu, 2, DType> hadamard_prod(Shape2(k, k));
  AllocSpace(&hadamard_prod);
  hadamard_prod = 1;

  // Note that out is of the same shape as the transpose of
  // the Khatri-Rao product
  //
  // When input is transposed, we could first put the transpose of
  // the Khatri-Rao product in out, then call the linear solver, which
  // will update the out's content to the final result;
  //
  // If the input is not transposed, we need to create an intermediate
  // tensor to store the Khatri-Rao product, call the linear solver with
  // MXNET_LAPACK_COL_MAJOR as the matrix layout, and transpose
  // the final result into out

  int info;
  if (input_transposed) {
    row_wise_kronecker(out, ts_arr);
    for (auto &ts : ts_arr)
      hadamard_prod *= implicit_dot(ts, ts.T());

    info = MXNET_LAPACK_posv<DType>(MXNET_LAPACK_ROW_MAJOR, 'U',
      k, out.size(1), hadamard_prod.dptr_, hadamard_prod.stride_,
      out.dptr_, out.stride_);
  } else {
    Tensor<cpu, 2, DType> kr(Shape2(out.size(1), out.size(0)));
    AllocSpace(&kr);
    khatri_rao(kr, ts_arr);

    for (auto &ts : ts_arr)
      hadamard_prod *= implicit_dot(ts.T(), ts);

    info = MXNET_LAPACK_posv<DType>(MXNET_LAPACK_COL_MAJOR, 'U',
      k, out.size(1), hadamard_prod.dptr_, hadamard_prod.stride_,
      kr.dptr_, kr.stride_);

    flip<cpu, DType>(out.size(1), out.size(0), out.dptr_, out.stride_,
      kr.dptr_, kr.stride_);
    FreeSpace(&kr);
  }

  FreeSpace(&hadamard_prod);
  if (info != 0)
    LOG(FATAL) << "The linear solver in inv_prod() returns " << info;
}


template<typename xpu, typename DType>
inline void KhatriRaoCompute_(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &out_data) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;

  Stream<xpu> *stream = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> out = out_data[0].get<xpu, 2, DType>(stream);
  std::vector<Tensor<xpu, 2, DType> > ts_arr(in_data.size());
  std::transform(in_data.begin(), in_data.end(), ts_arr.begin(),
                 [&stream](TBlob blob) -> Tensor<xpu, 2, DType> {
                   return blob.get<xpu, 2, DType>(stream);
                 });
  khatri_rao(out, ts_arr);
}


template<typename xpu>
inline void KhatriRaoCompute(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      KhatriRaoCompute_<xpu, DType>(attrs, ctx, inputs, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_KRPROD_H_
