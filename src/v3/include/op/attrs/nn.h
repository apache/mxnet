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
 * Copyright (c) 2019 by Contributors
 * \file nn.h
 * \author Junru Shao
 */
#pragma once
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <string>

#include "../../ir.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {

class ConvAttrs : public ir::AttrsNode<ConvAttrs> {
 public:
  ir::Array<ir::Integer> stride = {1};
  ir::Array<ir::Integer> padding = {0};
  ir::Array<ir::Integer> dilation = {1};
  int64_t groups = 1;
  std::string layout = "INVALID";
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(ConvAttrs, "mxnet.v3.attrs.ConvAttrs") {
    MX_V3_ATTR_FIELD(stride);    // {w}, {h, w}, {d, h, w}
    MX_V3_ATTR_FIELD(padding);   // {w}, {h, w}, {d, h, w}
    MX_V3_ATTR_FIELD(dilation);  // {w}, {h, w}, {d, h, w}
    MX_V3_ATTR_FIELD(groups);
    MX_V3_ATTR_FIELD(layout);
  }
};

class BatchNormAttrs : public ir::AttrsNode<BatchNormAttrs> {
 public:
  double eps = 1e-5;
  double momentum = 0.1;
  bool affine = true;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(ConvAttrs, "mxnet.v3.attrs.BatchNormAttrs") {
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(momentum);
    MX_V3_ATTR_FIELD(affine);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif
