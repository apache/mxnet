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
 *  Implementation of API functions related to Higher DSL build.
 * \file api_lang.cc
 */

#include <mxnet/runtime/packed_func.h>
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <mxnet/ir/expr.h>
#include <mxnet/node/container.h>
#include <mxnet/expr_operator.h>
#include <nnvm/c_api.h>
#include <iostream>

namespace mxnet {

MXNET_REGISTER_GLOBAL("_const")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    if (args[0].type_code() == kDLInt) {
      *ret = make_const(args[1].operator MXNetDataType(),
                        args[0].operator int64_t());
    } else if (args[0].type_code() == kDLFloat) {
      *ret = make_const(args[1].operator MXNetDataType(),
                        args[0].operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  });

MXNET_REGISTER_GLOBAL("_Array")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
    std::vector<ObjectRef> data;
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].type_code() != kNull) {
        data.push_back(args[i].operator ObjectRef());
      } else {
        data.emplace_back(nullptr);
      }
    }
    auto node = make_object<ArrayNode>();
    node->data = std::move(data);
    *ret = Array<ObjectRef>(node);
  });

MXNET_REGISTER_GLOBAL("_Test")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  TShape shape = args[0].operator TShape();
  std::cout << shape << std::endl;
  // Array<IntImm> arr = Downcast<Array<IntImm>, ObjectRef>(args[0].operator ObjectRef());
  // std::cout << arr.size() << std::endl;
  // for (size_t i = 0; i < arr.size(); ++i) {
  //   std::cout << arr[i]->value << " ";
  // }
  // std::cout << std::endl;
});

}  // namespace mxnet
