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
 * \file kvstore_utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_KVSTORE_KVSTORE_UTILS_H_
#define MXNET_KVSTORE_KVSTORE_UTILS_H_

#include <dmlc/logging.h>
#include <mxnet/ndarray.h>
#include <mxnet/resource.h>
#include <utility>
#include <vector>

namespace mxnet {
namespace kvstore {


/*!
 * \brief sort and get unique values.
 */
template<typename xpu>
void UniqueImpl(const Resource& rsc, mshadow::Stream<xpu> *s,
                NDArray *out, nnvm::dim_t size);

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_UTILS_H_
