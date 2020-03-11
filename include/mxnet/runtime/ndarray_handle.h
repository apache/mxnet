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
 * \file ndarray_handle.h
 * \brief NDArray handle types
 */
#ifndef MXNET_RUNTIME_NDARRAY_HANDLE_H_
#define MXNET_RUNTIME_NDARRAY_HANDLE_H_
#include <mxnet/ndarray.h>
#include <mxnet/runtime/object.h>

namespace mxnet {

class NDArrayHandleObj : public Object {
 public:
  /*! \brief the Internal value. */
  NDArray* value;

  static constexpr const char* _type_key = "MXNet.NDArrayHandle";
  MXNET_DECLARE_FINAL_OBJECT_INFO(NDArrayHandleObj, Object)
};

class NDArrayHandle : public ObjectRef {
 public:
  explicit NDArrayHandle(NDArray* value) {
    runtime::ObjectPtr<NDArrayHandleObj> node = make_object<NDArrayHandleObj>();
    node->value = value;
    data_ = std::move(node);
  }
  MXNET_DEFINE_OBJECT_REF_METHODS(NDArrayHandle, ObjectRef, NDArrayHandleObj)
};

};  // namespace mxnet

#endif  // MXNET_RUNTIME_NDARRAY_HANDLE_H_
