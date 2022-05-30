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
 * \file c_api_test.cc
 * \brief C API of mxnet for the ease of testing backend in Python
 */
#include <mxnet/c_api_test.h>
#include <nnvm/pass.h>
#include "./c_api_common.h"
#include "../operator/subgraph/subgraph_property.h"
#include "../common/cuda/rtc.h"

int MXBuildSubgraphByOpNames(SymbolHandle sym_handle,
                             const char* prop_name,
                             const uint32_t num_ops,
                             const char** op_names,
                             SymbolHandle* ret_sym_handle) {
  nnvm::Symbol* s = new nnvm::Symbol();
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(sym_handle);
  *s                = sym->Copy();
  if (!op_name_set.empty()) {
    auto& backend = mxnet::op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(prop_name);
    LOG(INFO) << "Subgraph backend " << backend->GetName() << " is activated.";
    const auto& subgraph_prop_list = backend->GetSubgraphProperties();
    for (auto property : subgraph_prop_list) {
      nnvm::Graph g;
      g.outputs = s->outputs;
      property->SetAttr("graph", g);
      property->SetAttr("op_names", op_name_set);
      g.attrs["subgraph_property"] = std::make_shared<nnvm::any>(property);
      g                            = nnvm::ApplyPass(std::move(g), "EliminateCommonNodesPass");
      g                            = nnvm::ApplyPass(std::move(g), "BuildSubgraph");
      property->RemoveAttr("graph");
      g.attrs.erase("subgraph_property");
      s->outputs = g.outputs;
    }
  }
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSetSubgraphPropertyOpNames(const char* prop_name,
                                 const uint32_t num_ops,
                                 const char** op_names) {
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  (*mxnet::op::SubgraphPropertyOpNameSet::Get())[prop_name] = op_name_set;
  API_END();
}

int MXSetSubgraphPropertyOpNamesV2(const char* prop_name,
                                   const uint32_t num_ops,
                                   const char** op_names) {
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  auto& backend = mxnet::op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(prop_name);
  const auto& subgraph_prop_list = backend->GetSubgraphProperties();
  for (auto& property : subgraph_prop_list) {
    property->SetAttr("op_names", op_name_set);
  }
  API_END();
}

int MXRemoveSubgraphPropertyOpNames(const char* prop_name) {
  API_BEGIN();
  mxnet::op::SubgraphPropertyOpNameSet::Get()->erase(prop_name);
  API_END();
}

int MXRemoveSubgraphPropertyOpNamesV2(const char* prop_name) {
  API_BEGIN();
  auto& backend = mxnet::op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(prop_name);
  const auto& subgraph_prop_list = backend->GetSubgraphProperties();
  for (auto& property : subgraph_prop_list) {
    property->RemoveAttr("op_names");
  }
  API_END();
}

int MXGetEnv(const char* name, const char** value) {
  API_BEGIN();
  *value = getenv(name);
  API_END();
}

int MXSetEnv(const char* name, const char* value) {
  API_BEGIN();
#ifdef _WIN32
  auto value_arg = (value == nullptr) ? "" : value;
  _putenv_s(name, value_arg);
#else
  if (value == nullptr)
    unsetenv(name);
  else
    setenv(name, value, 1);
#endif
  API_END();
}

int MXGetMaxSupportedArch(uint32_t* max_arch) {
  API_BEGIN();
#if MXNET_USE_CUDA
  *max_arch = static_cast<uint32_t>(mxnet::common::cuda::rtc::GetMaxSupportedArch());
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 to have CUDA runtime compilation.";
#endif
  API_END();
}
