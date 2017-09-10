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
