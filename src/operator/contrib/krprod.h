/*!
 *  Copyright (c) 2017 by Contributors
 *  \file krprod.h
 *  \brief Core function for Khatri-Rao product
 *  \author Jencir Lee
 */
#ifndef MXNET_OPERATOR_CONTRIB_KRPROD_H_
#define MXNET_OPERATOR_CONTRIB_KRPROD_H_
#include <vector>
#include "mshadow/tensor.h"

namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;

/*!
 * \brief Computes Khatri-Rao product
 *
 * As MXNet matrices are row-major, this implementation computes
 * Khatri-Rao product row by row, rather than column by column.
 * E.g. if the input matrices are of shape (3, 2), (3, 4), (3, 5),
 * the result matrix will be of shape (3, 2 * 4 * 5), which is (3, 40).
 *
 * \param out result matrix
 * \param ts_arr vector of input matrices
 */
template <typename DType>
inline void krprod
  (Tensor<cpu, 2, DType> out,
  const std::vector<Tensor<cpu, 2, DType> > &ts_arr) {
  // If no input matrix, return all-one vector
  if (ts_arr.empty()) {
    CHECK_EQ(1, out.size(1)) << "The output matrix must have width 1.";
    out = 1;
    return;
  }

  // Check all input and output matrices have the same height
  // and the output matrix has the right width
  int ts_height = static_cast<int>(out.size(0));
  int krprod_length = 1;
  for (auto & ts : ts_arr) {
    CHECK_EQ(ts_height, static_cast<int>(ts.size(0)))
      << "All input and output matrices must have the same height.";
    krprod_length *= ts.size(1);
  }
  CHECK_EQ(krprod_length, static_cast<int>(out.size(1)));

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

  // Compute each intermediate Khatri-Rao product
  storage = 1;
  krprod_length = 1;
  for (auto & ts : ts_arr) {
    expr::BLASEngine<cpu, DType>::SetStream
      (result->stream_);

    // Compute the current Khatri-Rao product, row by row
    *result = 0;
    for (int i = 0; i < ts_height; ++i) {
      // BLAS signature
      //
      // dger(
      //   m : ts.size(1), length of each row of current matrix
      //   n : krprod_length, length of each row of previous result
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
        ts.size(1), krprod_length, 1,
        ts[i].dptr_, 1,
        (*given)[i].dptr_, 1,
        (*result)[i].dptr_, ts.size(1));
    }
    krprod_length *= ts.size(1);

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
 * \brief Convenience function for Khatri-Rao product for two input matrices
 *
 * As MXNet matrices are row-major, this implementation computes
 * Khatri-Rao product row by row, rather than column by column.
 * E.g. if the input matrices are of shape (3, 2), (3, 4), (3, 5),
 * the result matrix will be of shape (3, 2 * 4 * 5), which is (3, 40).
 *
 * \param out result matrix
 * \param in1 first input matrix
 * \param in2 second input matrix
 */
template <typename DType>
inline void krprod
  (Tensor<cpu, 2, DType> out,
  const Tensor<cpu, 2, DType> &in1,
  const Tensor<cpu, 2, DType> &in2) {
  krprod(out, std::vector<Tensor<cpu, 2, DType> > {in1, in2});
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_KRPROD_H_
