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
 * \file unravel.cc
 * \brief Implementation of the API of functions in src/operator/tensor/ravel.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/ravel.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.unravel_index")
    .set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
      using namespace runtime;
      const nnvm::Op* op = Op::Get("_npi_unravel_index");
      nnvm::NodeAttrs attrs;
      op::RavelParam param;
      if (args[1].type_code() == kNull) {
        param.shape = TShape(-1, 0);
      } else if (args[1].type_code() == kDLInt) {
        param.shape = TShape(1, args[1].operator int64_t());
      } else {
        param.shape = TShape(args[1].operator ObjectRef());
      }
      attrs.parsed = param;
      attrs.op     = op;
      SetAttrDict<op::RavelParam>(&attrs);
      NDArray* inputs[] = {args[0].operator mxnet::NDArray *()};
      int num_inputs    = 1;
      int num_outputs   = 0;
      auto ndoutputs    = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
      if (num_outputs == 1) {
        *ret = ndoutputs[0];
      } else {
        std::vector<NDArrayHandle> ndarray_handles;
        ndarray_handles.reserve(num_outputs);
        for (int i = 0; i < num_outputs; ++i) {
          ndarray_handles.emplace_back(ndoutputs[i]);
        }
        *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
      }
    });

}  // namespace mxnet
