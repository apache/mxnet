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
 *  Copyright (c) 2015 by Contributors
 * \file io.h
 * \brief mxnet io data structure and data iterator
 */
#ifndef MXNET_IO_H_
#define MXNET_IO_H_

#include <dmlc/data.h>
#include <dmlc/registry.h>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include "./base.h"
#include "./ndarray.h"

namespace mxnet {
/*!
 * \brief iterator type
 * \tparam DType data type
 */
template<typename DType>
class IIterator : public dmlc::DataIter<DType> {
 public:
  /*!
   * \brief set the parameters and init iter
   * \param kwargs key-value pairs
   */
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;
  /*! \brief reset the iterator */
  virtual void BeforeFirst(void) = 0;
  /*! \brief move to next item */
  virtual bool Next(void) = 0;
  /*! \brief get current data */
  virtual const DType &Value(void) const = 0;
  /*! \brief constructor */
  virtual ~IIterator(void) {}
  /*! \brief store the name of each data, it could be used for making NDArrays */
  std::vector<std::string> data_names;
  /*! \brief set data name to each attribute of data */
  inline void SetDataName(const std::string data_name) {
    data_names.push_back(data_name);
  }
};  // class IIterator

/*! \brief a single data instance */
struct DataInst {
  /*! \brief unique id for instance */
  unsigned index;
  /*! \brief content of data */
  std::vector<TBlob> data;
  /*! \brief extra data to be fed to the network */
  std::string extra_data;
};  // struct DataInst

/*!
 * \brief DataBatch of NDArray, returned by Iterator
 */
struct DataBatch {
  /*! \brief content of dense data, if this DataBatch is dense */
  std::vector<NDArray> data;
  /*! \brief index of image data */
  std::vector<uint64_t> index;
  /*! \brief extra data to be fed to the network */
  std::string extra_data;
  /*! \brief num of example padded to batch */
  int num_batch_padd;
};  // struct DataBatch

/*! \brief typedef the factory function of data iterator */
typedef std::function<IIterator<DataBatch> *()> DataIteratorFactory;
/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct DataIteratorReg
    : public dmlc::FunctionRegEntryBase<DataIteratorReg,
                                        DataIteratorFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Iterators
//--------------------------------------------------------------
/*!
 * \brief Macro to register Iterators
 *
 * \code
 * // example of registering a mnist iterator
 * REGISTER_IO_ITE(MNISTIter)
 * .describe("Mnist data iterator")
 * .set_body([]() {
 *     return new PrefetcherIter(new MNISTIter());
 *   });
 * \endcode
 */
#define MXNET_REGISTER_IO_ITER(name)                                    \
  DMLC_REGISTRY_REGISTER(::mxnet::DataIteratorReg, DataIteratorReg, name)
}  // namespace mxnet
#endif  // MXNET_IO_H_
