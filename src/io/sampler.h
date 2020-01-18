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
class IndexSampler : public dmlc::DataIter<int64_t> {
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
  virtual const int64_t &Value(void) const = 0;
  /*! \brief deconstructor */
  virtual ~IndexSampler(void) {}
};  // class IndexSampler

typedef std::vector<int64_t> BatchSample;

class BatchSampler : public dmlc::DataIter<BatchSample> {
 public:
  /*! \brief deconstructor */
  virtual ~BatchSampler(void) {
  }
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
  virtual const BatchSample &Value(void) const = 0;
};  // class BatchSampler

/*! \brief typedef the factory function of data sampler */
typedef std::function<BatchSampler *()> BatchSamplerFactory;
/*!
 * \brief Registry entry for BatchSampler factory functions.
 */
struct BatchSamplerReg
    : public dmlc::FunctionRegEntryBase<BatchSamplerReg,
                                        BatchSamplerFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Samplers
//--------------------------------------------------------------
/*!
 * \brief Macro to register Samplers
 *
 * \code
 * // example of registering a Random Sampler
 * MXNET_REGISTER_IO_SAMPLER(RandomBatchSampler)
 * .describe("Random batch sampler")
 * .set_body([]() {
 *     return new BatchSampler(new RandomSampler);
 *   });
 * \endcode
 */
#define MXNET_REGISTER_IO_BATCH_SAMPLER(name)                                    \
  DMLC_REGISTRY_REGISTER(::mxnet::io::BatchSamplerReg, BatchSamplerReg, name)
}  // namespace io
}  // namespace mxnet

#endif  // MXNET_IO_SAMPLER_H_