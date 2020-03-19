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
 * \file src/runtime/container.cc
 * \brief Implementations of common plain old data (POD) containers.
 */
// Acknowledgement: This file originates from incubator-tvm
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/memory.h>
#include <mxnet/runtime/object.h>
#include <mxnet/runtime/registry.h>

namespace mxnet {
namespace runtime {

MXNET_REGISTER_GLOBAL("container._GetADTTag")
.set_body([](MXNetArgs args, MXNetRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.tag());
});

MXNET_REGISTER_GLOBAL("container._GetADTSize")
.set_body([](MXNetArgs args, MXNetRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.size());
});


MXNET_REGISTER_GLOBAL("container._GetADTFields")
.set_body([](MXNetArgs args, MXNetRetValue* rv) {
  ObjectRef obj = args[0];
  int idx = args[1];
  const auto& adt = Downcast<ADT>(obj);
  CHECK_LT(idx, adt.size());
  *rv = adt[idx];
});

MXNET_REGISTER_GLOBAL("container._ADT")
.set_body([](MXNetArgs args, MXNetRetValue* rv) {
  int itag = args[0];
  size_t tag = static_cast<size_t>(itag);
  std::vector<ObjectRef> fields;
  for (int i = 1; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *rv = ADT(tag, fields);
});

MXNET_REGISTER_OBJECT_TYPE(ADTObj);

}  // namespace runtime

}  // namespace mxnet
