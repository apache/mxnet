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
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
// Acknowledgement: This file originates from incubator-tvm

#include <mxnet/runtime/c_runtime_api.h>

#include <dmlc/thread_local.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/runtime/registry.h>
#include <sstream>
#include <array>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cctype>

#include "../c_api/c_api_common.h"

using namespace mxnet::runtime;

struct MXNetRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  MXNetByteArray ret_bytes;
};

typedef dmlc::ThreadLocalStore<MXNetRuntimeEntry> MXNetAPIRuntimeStore;

int MXNetFuncFree(MXNetFunctionHandle func) {
  API_BEGIN();
  delete static_cast<PackedFunc*>(func);
  API_END();
}

int MXNetFuncCall(MXNetFunctionHandle func,
                  MXNetValue* args,
                  int* arg_type_codes,
                  int num_args,
                  MXNetValue* ret_val,
                  int* ret_type_code) {
  API_BEGIN();
  MXNetRetValue rv;
  (*static_cast<const PackedFunc*>(func)).CallPacked(
      MXNetArgs(args, arg_type_codes, num_args), &rv);
  // handle return string.
  if (rv.type_code() == kStr ||
      rv.type_code() == kBytes) {
    MXNetRuntimeEntry* e = MXNetAPIRuntimeStore::Get();
    e->ret_str = *rv.ptr<std::string>();
    if (rv.type_code() == kBytes) {
      e->ret_bytes.data = e->ret_str.c_str();
      e->ret_bytes.size = e->ret_str.length();
      *ret_type_code = kBytes;
      ret_val->v_handle = &(e->ret_bytes);
    } else {
      *ret_type_code = kStr;
      ret_val->v_str = e->ret_str.c_str();
    }
  } else {
    rv.MoveToCHost(ret_val, ret_type_code);
  }
  API_END();
}
