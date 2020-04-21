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

#include <math.h>
#include <iostream>
#include <algorithm>
#include "lib_api.h"

/* \brief a basic pass that copies the input to the output */
MXReturnValue myPass(const std::string& in_graph, const std::string** out_graph,
                     const std::unordered_map<std::string, std::string>& options,
                     const std::unordered_map<std::string, MXTensor>& args,
                     const std::unordered_map<std::string, MXTensor>& aux,
                     const PassResource& res) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }

  *out_graph = new std::string(in_graph);
  return MX_SUCCESS;
}

REGISTER_PASS(myPass)
.setBody(myPass);

/* \brief a basic pass that parses the input string to JSON and then dumps it back */
MXReturnValue jsonPass(const std::string& in_graph, const std::string** out_graph,
                       const std::unordered_map<std::string, std::string>& options,
                       const std::unordered_map<std::string, MXTensor>& args,
                       const std::unordered_map<std::string, MXTensor>& aux,
                       const PassResource& res) {
  for (auto kv : options)
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;

  // add test arg/aux
  
  MXTensor* arg_ = res.alloc_arg("test_arg",{1},MXContext::CPU(0),kFloat32);
  MXTensor* aux_ = res.alloc_aux("test_aux",{1},MXContext::CPU(0),kFloat32);
  
  // convert json string to json object
  JsonParser parser;
  JsonVal json_val = parser.parse_to_json(in_graph);

  // get nodes list
  JsonVal nodes = json_val.map[JsonVal("nodes")];

  // loop over nodes
  for(int i=0; i<nodes.list.size(); i++) {
    JsonVal node = nodes.list[i];
    // get the op name
    std::string op = node.map[JsonVal("op")].str;
    // get node ID inputs to op
    JsonVal node_inputs = node.map[JsonVal("inputs")];

    //get shape/type if available
    std::string shape;
    int dtype = -1;
    if(node.map.find(JsonVal("attrs")) != node.map.end()) {
      JsonVal attrs = node.map[JsonVal("attrs")];
      if(attrs.map.find(JsonVal("shape")) != attrs.map.end()) 
        shape = attrs.map[JsonVal("shape")].str;
      if(attrs.map.find(JsonVal("dtype")) != attrs.map.end())
        dtype = std::stoi(attrs.map[JsonVal("dtype")].str);
    }
  }
  
  *out_graph = new std::string(parser.dump(json_val));
  return MX_SUCCESS;
}

REGISTER_PASS(jsonPass)
.setBody(jsonPass);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
