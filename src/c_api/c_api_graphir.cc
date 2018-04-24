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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_graph.cc
 * \brief C API related to Graph IR.
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include "./c_api_common.h"

int MXNNGraphCreate(SymbolHandle symbol, GraphHandle *graph) {
  return NNGraphCreate(symbol, graph);
}

int MXNNGraphFree(GraphHandle handle) {
  return NNGraphFree(handle);
}

int MXNNGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol) {
  return NNGraphGetSymbol(graph, symbol);
}

int MXNNGraphSetJSONAttr(GraphHandle handle,
                         const char *key,
                         const char *json_value) {
  return NNGraphSetJSONAttr(handle, key, json_value);
}

int MXNNGraphGetJSONAttr(GraphHandle handle,
                         const char *key,
                         const char **json_out,
                         int *success) {
  return NNGraphGetJSONAttr(handle, key, json_out, success);
}

int MXNNGraphSetNodeEntryListAttr_(GraphHandle handle,
                                   const char *key,
                                   SymbolHandle list) {
  return NNGraphSetNodeEntryListAttr_(handle, key, list);
}

int MXNNGraphApplyPasses(GraphHandle src,
                         mx_uint num_pass,
                         const char **pass_names,
                         GraphHandle *dst) {
  return NNGraphApplyPasses(src, num_pass, pass_names, dst);
}
