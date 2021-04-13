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
 * \file npx_deconvolution_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_deconvolution_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/deconvolution-inl.h"

namespace mxnet {

inline int String2Layout(const std::string& s) {
  using namespace op;
  if (s == "NCW") {
    return mshadow::kNCW;
  } else if (s == "NCHW") {
    return mshadow::kNCHW;
  } else if (s == "NCDHW") {
    return mshadow::kNCDHW;
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

inline int String2CudnnTune(const std::string& s) {
  using namespace op;
  if (s == "off") {
    return deconv::kOff;
  } else if (s == "limited_workspace") {
    return deconv::kLimited;
  } else if (s == "fastest") {
    return deconv::kFastest;
  } else {
    LOG(FATAL) << "unknown cudnn tune type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.deconvolution")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_deconvolution");
  op::DeconvolutionParam param;
  int args_size = args.size();
  // no_bias
  if (args[args_size - 4].type_code() == kNull) {
    param.no_bias = false;
  } else {
    param.no_bias = args[args_size - 4].operator bool();
  }
  // inputs
  int num_inputs = param.no_bias ? 2 : 3;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // kernel
  if (args[num_inputs].type_code() == kDLInt) {
    param.kernel = TShape(1, args[num_inputs].operator int64_t());
  } else {
    param.kernel = TShape(args[num_inputs].operator ObjectRef());
  }
  // layout
  if (args[num_inputs + 12].type_code() == kNull) {
    param.layout = dmlc::nullopt;
  } else {
    param.layout = String2Layout(args[num_inputs + 12]);
  }
  // Check
  if (param.kernel.ndim() == 1) {
    param.layout = param.layout? param.layout.value() : mshadow::kNCW;
  } else if (param.kernel.ndim() == 2) {
    param.layout = param.layout ? param.layout.value() : mshadow::kNCHW;
  } else {
    CHECK_EQ(param.kernel.ndim(), 3U) << param.kernel.ndim() << "D convolution not supported";
    param.layout = param.layout ? param.layout.value(): mshadow::kNCDHW;
  }
  // stride
  if (args[num_inputs + 1].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.stride = Shape1(1);
    } else if (param.kernel.ndim() == 2) {
      param.stride = Shape2(1, 1);
    } else {
      param.stride = Shape3(1, 1, 1);
    }
  } else if (args[num_inputs + 1].type_code() == kDLInt) {
    param.stride = TShape(1, args[num_inputs + 1].operator int64_t());
  } else {
    param.stride = TShape(args[num_inputs + 1].operator ObjectRef());
  }
  // dilate
  if (args[num_inputs + 2].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.dilate = Shape1(1);
    } else if (param.kernel.ndim() == 2) {
      param.dilate = Shape2(1, 1);
    } else {
      param.dilate = Shape3(1, 1, 1);
    }
  } else if (args[num_inputs + 2].type_code() == kDLInt) {
    param.dilate = TShape(1, args[num_inputs + 2].operator int64_t());
  } else {
    param.dilate = TShape(args[num_inputs + 2].operator ObjectRef());
  }
  // pad
  if (args[num_inputs + 3].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.pad = Shape1(0);
    } else if (param.kernel.ndim() == 2) {
      param.pad = Shape2(0, 0);
    } else {
      param.pad = Shape3(0, 0, 0);
    }
  } else if (args[num_inputs + 3].type_code() == kDLInt) {
    param.pad = TShape(1, args[num_inputs + 3].operator int64_t());
  } else {
    param.pad = TShape(args[num_inputs + 3].operator ObjectRef());
  }
  // adj
  if (args[num_inputs + 4].type_code() == kNull) {
    if (param.kernel.ndim() == 1) {
      param.adj = Shape1(0);
    } else if (param.kernel.ndim() == 2) {
      param.adj = Shape2(0, 0);
    } else {
      param.adj = Shape3(0, 0, 0);
    }
  } else if (args[num_inputs + 4].type_code() == kDLInt) {
    param.adj = TShape(1, args[num_inputs + 4].operator int64_t());
  } else {
    param.adj = TShape(args[num_inputs + 4].operator ObjectRef());
  }
  // target_shape
  if (args[num_inputs + 5].type_code() != kNull) {
    if (args[num_inputs + 5].type_code() == kDLInt) {
      param.target_shape = TShape(1, args[num_inputs + 5].operator int64_t());
    } else {
      param.target_shape = TShape(args[num_inputs + 5].operator ObjectRef());
    }
  }
  // num_filter
  param.num_filter = (uint32_t) (args[num_inputs + 6].operator int());
  // num_group
  param.num_group = (uint32_t) (args[num_inputs + 7].operator int());
  // workspace
  param.workspace = args[num_inputs + 8].operator uint64_t();
  // cudnn_tune
  if (args[num_inputs + 10].type_code() == kNull) {
    param.cudnn_tune = dmlc::nullopt;
  } else {
    param.cudnn_tune = String2CudnnTune(args[num_inputs + 10]);
  }
  // cudnn_off
  if (args[num_inputs + 11].type_code() == kNull) {
    param.cudnn_off = false;
  } else {
    param.cudnn_off = args[num_inputs + 11].operator bool();
  }

  CHECK_EQ(param.kernel.ndim(), param.stride.ndim())
    << "Stride must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param.kernel << " while stride is "
    << param.stride;
  CHECK_EQ(param.kernel.ndim(), param.dilate.ndim())
    << "Dilate must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param.kernel << " while dilate is "
    << param.dilate;
  CHECK_EQ(param.kernel.ndim(), param.pad.ndim())
    << "Padding must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param.kernel << " while padding is "
    << param.pad;
  CHECK_EQ(param.kernel.ndim(), param.adj.ndim())
    << "Adjustment must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param.kernel << " while adjustment is "
    << param.adj;

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::DeconvolutionParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
