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
 * \file _api_internal.cc
 * \brief Internal functions exposed to python for FFI use only
 */
// Acknowledgement: This file originates from incubator-tvm
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <mxnet/expr_operator.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/ir/expr.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/container_ext.h>
#include <mxnet/runtime/ffi_helper.h>
#include <nnvm/c_api.h>

namespace mxnet {

MXNET_REGISTER_GLOBAL("_Integer")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    using namespace runtime;
    if (args[0].type_code() == kDLInt) {
      *ret = Integer(args[0].operator int64_t());
    } else {
      LOG(FATAL) << "only accept int";
    }
});

MXNET_REGISTER_GLOBAL("_Float")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    using namespace runtime;
    if (args[0].type_code() == kDLFloat) {
      *ret = Float(args[0].operator double());
    } else {
      LOG(FATAL) << "only accept float";
    }
});

MXNET_REGISTER_GLOBAL("_ADT")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    using namespace runtime;
    std::vector<ObjectRef> data;
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].type_code() == kNDArrayHandle) {
        mxnet::NDArray *array = args[i].operator mxnet::NDArray*();
        ObjectRef input = NDArrayHandle(array);
        data.push_back(input);
      } else if (args[i].type_code() != kNull) {
        ObjectRef input = String::CanConvertFrom(args[i]) ? args[i].operator String()
                                                          : args[i].operator ObjectRef();
        data.push_back(input);
      } else {
        data.emplace_back(nullptr);
      }
    }
    *ret = ADT(0, data.begin(), data.end());
});

MXNET_REGISTER_GLOBAL("_Map")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    using namespace runtime;
    CHECK_EQ(args.size() % 2, 0);
    std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> data;
    for (int i = 0; i < args.num_args; i += 2) {
      ObjectRef k =
          String::CanConvertFrom(args[i]) ? args[i].operator String()
                                          : args[i].operator ObjectRef();
      ObjectRef v;
      if (args[i + 1].type_code() == kNDArrayHandle) {
        mxnet::NDArray *array = args[i + 1].operator mxnet::NDArray*();
        v = NDArrayHandle(array);
      } else {
        v = args[i + 1];
      }
      data.emplace(std::move(k), std::move(v));
    }
    *ret = Map<ObjectRef, ObjectRef>(data);
});

MXNET_REGISTER_GLOBAL("_String")
.set_body([] (runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  std::string str = args[0].operator std::string();
  *ret = String(std::move(str));
});

MXNET_REGISTER_GLOBAL("_echo")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  *ret = args[0];
});

MXNET_REGISTER_API("_nop")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
});

}  // namespace mxnet
