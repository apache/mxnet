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
 * \file batchify.h
 * \brief Mini-batch data combination functions.
 */

#ifndef MXNET_IO_BATCHIFY_H_
#define MXNET_IO_BATCHIFY_H_
#include <dmlc/registry.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace io {

class BatchifyFunction {
  public:
    /*! \brief Destructor */
    virtual ~BatchifyFunction(void) {};
    /*! \brief The batchify logic */
    virtual std::vector<NDArray> Batchify(
        std::vector<std::vector<NDArray> > inputs,
        std::vector<int> keep_dim) = 0;
};  // class BatchifyFunction

/*! \brief typedef the factory function of data sampler */
typedef std::function<BatchifyFunction *()> BatchifyFunctionFactory;
/*!
 * \brief Registry entry for DataSampler factory functions.
 */
struct BatchifyFunctionReg
    : public dmlc::FunctionRegEntryBase<BatchifyFunctionReg,
                                        BatchifyFunctionFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Batchify Function
//--------------------------------------------------------------
/*!
 * \brief Macro to register Batchify Functions
 *
 * \code
 * // example of registering a Batchify Function
 * MXNET_REGISTER_IO_BATCHIFY_FUNCTION(StackBatchify)
 * .describe("Stack Batchify Function")
 * .set_body([]() {
 *     return new StackBatchify();
 *   });
 * \endcode
 */
#define MXNET_REGISTER_IO_BATCHIFY_FUNCTION(name)                                    \
  DMLC_REGISTRY_REGISTER(::mxnet::io::BatchifyFunctionReg, BatchifyFunctionReg, name)
}  // namespace io
}  // namespace mxnet

#endif  // MXNET_IO_BATCHIFY_H_