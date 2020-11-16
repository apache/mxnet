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
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation library file
 */

#include <cmath>
#include <iostream>
#include <algorithm>
#include "mxnet/lib_api.h"

using namespace mxnet::ext;

/* \brief a basic pass that prints out the options and the graph */
MXReturnValue myPass(mxnet::ext::Graph *g,
                     const std::unordered_map<std::string, std::string>& options) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }
  g->print();
  return MX_SUCCESS;
}

REGISTER_PASS(myPass)
.setBody(myPass);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
