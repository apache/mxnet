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
 * \file api_registry.h
 * \brief This file contains utilities related to
 *  the MXNet's global function registry.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_API_REGISTRY_H_
#define MXNET_API_REGISTRY_H_

#include <string>
#include <utility>
#include "runtime/registry.h"

namespace mxnet {
/*!
 * \brief Register an API function globally.
 * It simply redirects to MXNET_REGISTER_GLOBAL
 *
 * \code
 *   MXNET_REGISTER_API(MyPrint)
 *   .set_body([](MXNetArgs args, MXNetRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#define MXNET_REGISTER_API(OpName) MXNET_REGISTER_GLOBAL(OpName)

}  // namespace mxnet
#endif  // MXNET_API_REGISTRY_H_
