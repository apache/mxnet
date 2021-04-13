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
 * \file npx_pooling_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_pooling_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/pooling-inl.h"

namespace mxnet {

inline int String2Layout(const std::string& s) {
  using namespace op;
  if (s == "NCW") {
    return mshadow::kNCW;
  } else if (s == "NCHW") {
    return mshadow::kNCHW;
  } else if (s == "NCDHW") {
    return mshadow::kNCDHW;
  } else if (s == "NWC") {
    return mshadow::kNWC;
  } else if (s == "NHWC") {
    return mshadow::kNHWC;
  } else if (s == "NDHWC") {
    return mshadow::kNDHWC;
  } else {
    LOG(FATAL) << "unknown layout type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

inline int String2PoolType(const std::string& s) {
  using namespace op;
  if (s == "max") {
    return pool_enum::kMaxPooling;
  } else if (s == "avg") {
    return pool_enum::kAvgPooling;
  } else if (s == "sum") {
    return pool_enum::kSumPooling;
  } else if (s == "lp") {
    return pool_enum::kLpPooling;
  } else {
    LOG(FATAL) << "unknown pooling type type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

inline int String2Convention(const std::string& s) {
  using namespace op;
  if (s == "full") {
    return pool_enum::kFull;
  } else if (s == "valid") {
    return pool_enum::kValid;
  } else if (s == "same") {
    return pool_enum::kSame;
  } else {
    LOG(FATAL) << "unknown pooling convention type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.pooling")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_pooling");
  op::PoolingParam param;
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};

  // kernel
  if (args[1].type_code() == kDLInt) {
    param.kernel = TShape(1, args[1].operator int64_t());
  } else {
    param.kernel = TShape(args[1].operator ObjectRef());
  }

  // stride
  if (args[2].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.stride = mshadow::Shape1(1);
    } else if (param.kernel.ndim() == 2) {
      param.stride = mshadow::Shape2(1, 1);
    } else {
      param.stride = mshadow::Shape3(1, 1, 1);
    }
  } else if (args[2].type_code() == kDLInt) {
    param.stride = TShape(1, args[2].operator int64_t());
  } else {
    param.stride = TShape(args[2].operator ObjectRef());
  }
  // pad
  if (args[3].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.pad = mshadow::Shape1(0);
    } else if (param.kernel.ndim() == 2) {
      param.pad = mshadow::Shape2(0, 0);
    } else {
      param.pad = mshadow::Shape3(0, 0, 0);
    }
  } else if (args[3].type_code() == kDLInt) {
    param.pad = TShape(1, args[3].operator int64_t());
  } else {
    param.pad = TShape(args[3].operator ObjectRef());
  }
  // pool type
  param.pool_type = String2PoolType(args[4].operator std::string());
  // pooling convention
  param.pooling_convention = String2Convention(args[5].operator std::string());
  // global pool
  param.global_pool = args[6].operator bool();
  // cudnn_off
  if (args[7].type_code() == kNull) {
    param.cudnn_off = false;
  } else {
    param.cudnn_off = args[7].operator bool();
  }
  // p_value
  if (args[8].type_code() == kNull) {
    param.p_value = dmlc::nullopt;
  } else {
    param.p_value = args[8].operator int();
  }
  // count_include_pad
  if (args[9].type_code() == kNull) {
    param.count_include_pad = dmlc::nullopt;
  } else {
    param.count_include_pad = args[9].operator bool();
  }
  // layout
  if (args[10].type_code() == kNull) {
    param.layout = dmlc::nullopt;
  } else {
    param.layout = String2Layout(args[num_inputs + 10]);
  }

  if (param.global_pool == false) {
    CHECK_EQ(param.kernel.ndim(), 3U) << param.kernel.ndim()
        << "D pooling not supported";
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::PoolingParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
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
