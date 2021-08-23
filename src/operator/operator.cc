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
 *  Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief operator module of mxnet
 */
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/operator.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::OperatorPropertyReg);
}  // namespace dmlc

namespace mxnet {
// implementation of all factory functions
OperatorProperty *OperatorProperty::Create(const char* type_name) {
  auto *creator = dmlc::Registry<OperatorPropertyReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Operator " << type_name << " in registry";
  }
  return creator->body();
}
}  // namespace mxnet
