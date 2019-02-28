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
 *  Copyright (c) 2018 by Contributors
 * \file c_api_test.cc
 * \brief C API of mxnet for the ease of testing backend in Python
 */
#include <mxnet/c_api_test.h>
#include <nnvm/pass.h>
#include "./c_api_common.h"
#include "../operator/subgraph/subgraph_property.h"

int MXPartitionGraphByOpNames(SymbolHandle sym_handle,
                              const char* prop_name,
                              const mx_uint num_ops,
                              const char** op_names,
                              SymbolHandle* ret_sym_handle) {
  nnvm::Symbol* s = new nnvm::Symbol();
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(sym_handle);
  *s = sym->Copy();
  nnvm::Graph g;
  g.outputs = s->outputs;
  if (!op_name_set.empty()) {
    mxnet::op::SubgraphPropertyPtr property
        = mxnet::op::SubgraphPropertyRegistry::Get()->CreateSubgraphProperty(prop_name);
    property->SetAttr("op_names", op_name_set);
    g.attrs["subgraph_property"] = std::make_shared<nnvm::any>(std::move(property));
  }
  g = nnvm::ApplyPass(std::move(g), "PartitionGraph");
  s->outputs = g.outputs;
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSetSubgraphPropertyOpNames(const char* prop_name,
                                 const mx_uint num_ops,
                                 const char** op_names) {
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  (*mxnet::op::SubgraphPropertyOpNameSet::Get())[prop_name] = op_name_set;
  API_END();
}

int MXRemoveSubgraphPropertyOpNames(const char* prop_name) {
  API_BEGIN();
  mxnet::op::SubgraphPropertyOpNameSet::Get()->erase(prop_name);
  API_END();
}
