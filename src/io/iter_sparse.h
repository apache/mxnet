/*!
 *  Copyright (c) 2017 by Contributors
 * \file iter_sparse.h
 * \brief mxnet sparse data iterator
 */
#ifndef MXNET_IO_ITER_SPARSE_H_
#define MXNET_IO_ITER_SPARSE_H_

#include <mxnet/io.h>
#include <mxnet/ndarray.h>

namespace mxnet {
/*!
 * \brief iterator type
 * \param DType data type
 */
template<typename DType>
class SparseIIterator : public IIterator<DType> {
 public:
  /*! \brief storage type of the data or label */
  virtual const NDArrayStorageType GetStorageType(bool is_data) const = 0;
  /*! \brief shape of the data or label */
  virtual const TShape GetShape(bool is_data) const = 0;
};  // class SparseIIterator

}  // namespace mxnet
#endif  // MXNET_IO_ITER_SPARSE_H_
