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
 * \file src/api/ndarary_handle.cc
 * \brief Implementations of NDArrayHandle
 */
#include <mxnet/runtime/ndarray_handle.h>
#include <mxnet/runtime/object.h>
#include <mxnet/runtime/registry.h>
#include <mxnet/runtime/packed_func.h>

namespace mxnet {
namespace runtime {

MXNET_REGISTER_GLOBAL("ndarray_handle._GetNDArrayHandleValue")
.set_body([](MXNetArgs args, MXNetRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& handle = Downcast<NDArrayHandle>(obj);
  *rv = handle->value;
});

MXNET_REGISTER_OBJECT_TYPE(NDArrayHandleObj);

}  // namespace runtime
}  // namespace mxnet
