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
 * \file runtime/ndarray.h
 * \brief A device-independent managed NDArray abstraction.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_NDARRAY_H_
#define MXNET_RUNTIME_NDARRAY_H_

namespace mxnet {
namespace runtime {

/*!
 * \brief The type trait indicates subclass of TVM's NDArray.
 *  For irrelavant classes, code = -1.
 *  For TVM NDArray itself, code = 0.
 *  All subclasses of NDArray should override code > 0.
 */
template<typename T>
struct array_type_info {
  /*! \brief the value of the traits */
  static const int code = -1;
};

}  // namespace runtime
}  // namespace mxnet
#endif  // MXNET_RUNTIME_NDARRAY_H_
