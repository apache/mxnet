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
 * \file np_histogram_op.cc
 * \brief Implementation of the API of functions in src/operator/tensor/histogram.cc
 */

#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/histogram-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.histogram")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_histogram");
  op::HistogramParam param;
  // parse bin_cnt
  if (args[2].type_code() == kNull) {
    param.bin_cnt = dmlc::nullopt;
  } else {
    param.bin_cnt = args[2].operator int();
  }

  // parse range
  if (args[3].type_code() == kNull) {
    param.range = dmlc::nullopt;
  } else {
    param.range = Obj2Tuple<double, Float>(args[3].operator ObjectRef());
  }

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::HistogramParam>(&attrs);

  std::vector<NDArray*> inputs_vec;
  int num_inputs = 0;

  if (args[2].type_code() != kNull) {
    CHECK_EQ(args[1].type_code(), kNull)
      << "bins should be None when bin_cnt is provided";
    inputs_vec.push_back((args[0].operator NDArray*()));
    num_inputs = 1;
  } else {
    CHECK_NE(args[1].type_code(), kNull)
      << "bins should not be None when bin_cnt is not provided";
    // inputs
    inputs_vec.push_back((args[0].operator NDArray*()));
    inputs_vec.push_back((args[1].operator NDArray*()));
    num_inputs = 2;
  }

  // outputs
  NDArray** out = nullptr;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs_vec.data(), &num_outputs, out);
  *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                 NDArrayHandle(ndoutputs[1])});
});

}  // namespace mxnet
