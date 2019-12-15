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

#include <iostream>
#include <algorithm>
#include "lib_api.h"

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs,
                         int* num_in, int* num_out) {
  *num_in = 1;
  *num_out = 1;
  if (attrs.count(SUBGRAPH_SYM_JSON)) {
    // example of subgraph json parsing
    JsonParser jp;
    JsonVal val = jp.parse_to_json(attrs[SUBGRAPH_SYM_JSON]);
    int input = 0;
    for (auto &item : val.map[JsonVal("nodes")].list) {
      if (item.map[JsonVal("op")].str == "null")
        input++;
    }
    int output = val.map[JsonVal("heads")].list.size();
    *num_in = input;
    *num_out = output;
  }
  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string, std::string> attrs,
                        std::vector<int> &intypes,
                        std::vector<int> &outtypes) {
  outtypes[0] = intypes[0];
  return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string, std::string> attrs,
                         std::vector<std::vector<unsigned int>> &inshapes,
                         std::vector<std::vector<unsigned int>> &outshapes) {
  outshapes[0] = inshapes[0];
  return MX_SUCCESS;
}

class MyStatefulOp : public CustomStatefulOp {
 public:
  explicit MyStatefulOp(std::string sym) : subgraph_sym(sym) {}

  MXReturnValue Forward(std::vector<MXTensor> inputs,
                        std::vector<MXTensor> outputs,
                        OpResource op_res) {
    std::cout << "Info: subgraph symbol is: " << std::endl;
    std::cout << subgraph_sym << std::endl;
    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();
    std::cout << "Info: output is: " << std::endl;
    for (int i = 0; i < inputs[0].size(); i++) {
      out_data[i] = in_data[i];
    }
    return MX_SUCCESS;
  }

 private:
  std::string subgraph_sym;
};

MXReturnValue createOpState(std::map<std::string, std::string> attrs,
                            CustomStatefulOp** op_inst) {
  std::string serialized_subgraph = "[empty]";
  // MXNet subgraph is stored as Symbol in operator node attrs subgraphs field
  // custom subgraph is stored as json string in custom operator attrs map entry
  if (attrs.count(SUBGRAPH_SYM_JSON)) {
    // user can now parse json and run other custom ops inside subgraph
    serialized_subgraph = attrs[SUBGRAPH_SYM_JSON];
  }
  *op_inst = new MyStatefulOp(serialized_subgraph);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_OP(_custom_subgraph_op)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setCreateOpState(createOpState);

const std::vector<std::string> op_names({"exp","log"});

MXReturnValue mySupportedOps(std::string json,
			     const int num_ids,
			     int *ids) {
  //convert json string to json object
  JsonParser parser;
  JsonVal json_val = parser.parse_to_json(json);
  //get nodes list
  JsonVal nodes = json_val.map[JsonVal("nodes")];

  //loop over nodes
  for(int i=0; i<nodes.list.size(); i++) {
    JsonVal node = nodes.list[i];
    JsonVal op = node.map[JsonVal("op")];

    //get shape/type if available
    std::string shape,dtype;
    if(node.map.find(JsonVal("attrs")) != node.map.end()) {
      JsonVal attrs = node.map[JsonVal("attrs")];
      if(attrs.map.find(JsonVal("shape")) != attrs.map.end()) 
        shape = attrs.map[JsonVal("shape")].str;
      if(attrs.map.find(JsonVal("dtype")) != attrs.map.end())
        dtype = attrs.map[JsonVal("dtype")].str;
    }

    //check if op is in whitelist
    if(std::find(op_names.begin(),op_names.end(),op.str.c_str()) != op_names.end()) {
      std::cout << op.str << " : " << shape << " " << dtype << std::endl;
      // found op in whitelist, set value to 1 to include op in subgraph
      ids[i]=1;
    }
  }  
  return MX_SUCCESS;
}

REGISTER_PARTITIONER(myProp)
.addStrategy("strategy1", mySupportedOps, "_custom_subgraph_op");

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
