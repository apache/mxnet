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
 * \file sampler.h
 * \brief The data sampler for access elements in dataset.
 */

#ifndef MXNET_IO_SAMPLER_H_
#define MXNET_IO_SAMPLER_H_

#include "dmlc/data.h"

#include <vector>
#include <string>
#include <utility>

namespace mxnet {
namespace io {
template<typename DType>
class DatasetSampler : public dmlc::DataIter<DType> {
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
  virtual ~DatasetSampler(void) {}
};  // class DatasetSampler

struct BatchSample {
    std::vector<int64_t> samples;
    std::size_t batch_size;
};  // struct BatchSample

/*! \brief typedef the factory function of data sampler */
typedef std::function<DatasetSampler<BatchSample> *()> DataSamplerFactory;
/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct DataSamplerReg
    : public dmlc::FunctionRegEntryBase<DataSamplerReg,
                                        DataSamplerFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Iterators
//--------------------------------------------------------------
/*!
 * \brief Macro to register Samplers
 *
 * \code
 * // example of registering a mnist iterator
 * MXNET_REGISTER_IO_SAMPLER(RandomSampler)
 * .describe("Random sampler")
 * .set_body([]() {
 *     return new RandomSampler();
 *   });
 * \endcode
 */
#define MXNET_REGISTER_IO_SAMPLER(name)                                    \
  DMLC_REGISTRY_REGISTER(::mxnet::DataSamplerReg, DataSamplerReg, name)
}  // namespace io
}  // namespace mxnet

#endif  // MXNET_IO_SAMPLER_H_