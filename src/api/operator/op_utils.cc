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
 * \file op_utils.cc
 * \brief Utility functions for modification in src/operator
 */

#include "op_utils.h"
#include <mxnet/base.h>

namespace mxnet {

std::string String2MXNetTypeWithBool(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return "float32";
    case mshadow::kFloat64:
      return "float64";
    case mshadow::kFloat16:
      return "float16";
    case mshadow::kUint8:
      return "uint8";
    case mshadow::kInt8:
      return "int8";
    case mshadow::kInt32:
      return "int32";
    case mshadow::kInt64:
      return "int64";
    case mshadow::kBool:
      return "bool";
    default:
      LOG(FATAL) << "Unknown type enum " << dtype;
  }
  LOG(FATAL) << "should not reach here ";
  return "";
}

}  // namespace mxnet
