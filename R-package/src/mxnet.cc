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
 * \file mxnet.cc
 * \brief The registry of all module functions and objects
 */
#include <fstream>
#include "./base.h"
#include "./ndarray.h"
#include "./symbol.h"
#include "./executor.h"
#include "./io.h"
#include "./kvstore.h"
#include "./export.h"
#include "./im2rec.h"

namespace mxnet {
namespace R {
void SetSeed(int seed) {
  MX_CALL(MXRandomSeed(seed));
}

void NotifyShutdown() {
  MX_CALL(MXNotifyShutdown());
}

void ProfilerSetConfig(SEXP params) {
  Rcpp::List kwargs(params);
  std::vector<std::string> keys = SafeGetListNames(kwargs);
  std::vector<std::string> str_keys(keys.size());
  std::vector<std::string> str_vals(keys.size());
  for (size_t i = 0; i < kwargs.size(); ++i) {
    RCHECK(keys[i].length() != 0)
      << "Profiler::SetConfig only accepts key=value style arguments";
    str_keys[i] = FormatParamKey(keys[i]);
    str_vals[i] = toPyString(keys[i], kwargs[i]);
  }
  std::vector<const char*> c_str_keys = CKeys(str_keys);
  std::vector<const char*> c_str_vals = CKeys(str_vals);

  MX_CALL(MXSetProfilerConfig(static_cast<mx_uint>(str_keys.size()),
                              dmlc::BeginPtr(c_str_keys), dmlc::BeginPtr(c_str_vals)));
}

void ProfilerSetState(int state) {
  MX_CALL(MXSetProfilerState(state));
}

// init rcpp module in base
void InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  function("mx.internal.set.seed", &SetSeed);
  function("mx.internal.notify.shutdown", &NotifyShutdown);
  function("mx.internal.profiler.config", &ProfilerSetConfig);
  function("mx.internal.profiler.state", &ProfilerSetState);
}
}  // namespace R
}  // namespace mxnet


RCPP_MODULE(mxnet) {
  using namespace mxnet::R; // NOLINT(*)
  mxnet::R::InitRcppModule();
  Context::InitRcppModule();
  NDArray::InitRcppModule();
  NDArrayFunction::InitRcppModule();
  Symbol::InitRcppModule();
  SymbolFunction::InitRcppModule();
  Executor::InitRcppModule();
  DataIter::InitRcppModule();
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
  IM2REC::InitRcppModule();
}

