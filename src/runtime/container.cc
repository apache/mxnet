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
#include <mxnet/runtime/container_ext.h>
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

MXNET_REGISTER_GLOBAL("container._Map")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  CHECK_EQ(args.size() % 2, 0);
  std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> data;
  for (int i = 0; i < args.num_args; i += 2) {
    ObjectRef k =
        String::CanConvertFrom(args[i]) ? args[i].operator String() : args[i].operator ObjectRef();
    ObjectRef v;
    if (args[i + 1].type_code() == kNDArrayHandle) {
      mxnet::NDArray *array = args[i + 1].operator mxnet::NDArray*();
      v = NDArrayHandle(array);
    } else {
      v = args[i + 1];
    }
    data.emplace(std::move(k), std::move(v));
  }
  *rv = Map<ObjectRef, ObjectRef>(data);
});

MXNET_REGISTER_GLOBAL("container._MapSize")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  CHECK_EQ(args[0].type_code(), kObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  CHECK(ptr->IsInstance<MapObj>());
  auto* n = static_cast<const MapObj*>(ptr);
  *rv = static_cast<int64_t>(n->size());
});

MXNET_REGISTER_GLOBAL("container._MapGetItem")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  CHECK_EQ(args[0].type_code(), kObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  CHECK(ptr->IsInstance<MapObj>());

  auto* n = static_cast<const MapObj*>(ptr);
  auto it = n->find(String::CanConvertFrom(args[1]) ? args[1].operator String()
                                                    : args[1].operator ObjectRef());
  CHECK(it != n->end()) << "cannot find the corresponding key in the Map";
  *rv = (*it).second;
});

MXNET_REGISTER_GLOBAL("container._MapItems")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  CHECK_EQ(args[0].type_code(), kObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  auto* n = static_cast<const MapObj*>(ptr);
  std::vector<ObjectRef> rkvs;
  for (const auto& kv : *n) {
    if (kv.first->IsInstance<StringObj>()) {
      rkvs.push_back(Downcast<String>(kv.first));
    } else {
      rkvs.push_back(kv.first);
    }
    rkvs.push_back(kv.second);
  }
  *rv = ADT(0, rkvs.begin(), rkvs.end());
});

MXNET_REGISTER_GLOBAL("container._MapCount")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  CHECK_EQ(args[0].type_code(), kObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  CHECK(ptr->IsInstance<MapObj>());
  const MapObj* n = static_cast<const MapObj*>(ptr);
  auto key = String::CanConvertFrom(args[1]) ? args[1].operator String()
                                             : args[1].operator ObjectRef();
  int64_t cnt = n->count(key);
  *rv = cnt;
});

MXNET_REGISTER_GLOBAL("container._String")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  std::string str = args[0].operator std::string();
  *rv = String(std::move(str));
});

MXNET_REGISTER_GLOBAL("container._GetFFIString")
.set_body([] (MXNetArgs args, MXNetRetValue* rv) {
  String str = args[0].operator String();
  *rv = std::string(str);
});

MXNET_REGISTER_OBJECT_TYPE(ADTObj);
MXNET_REGISTER_OBJECT_TYPE(MapObj);
MXNET_REGISTER_OBJECT_TYPE(StringObj);

}  // namespace runtime

}  // namespace mxnet
