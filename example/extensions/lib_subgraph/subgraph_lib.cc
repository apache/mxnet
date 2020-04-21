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

/* function to execute log operator on floats */
void myLog(MXTensor &in, MXTensor &out) {
  float* inp = in.data<float>();
  float* outp = out.data<float>();
  for (int64_t i = 0; i < in.size(); i++) {
    outp[i] = logf(inp[i]);
  }
}
/* function to execute exp operator on floats */
void myExp(MXTensor &in, MXTensor &out) {
  float* inp = in.data<float>();
  float* outp =out.data<float>();
  for (int64_t i = 0; i < in.size(); i++) {
    outp[i] = expf(inp[i]);
  }
}

/* function to execute ops in subgraph
 * In MXNet, subgraphs are sorted in topological order
 * so all we need to do is go through the ops in order
 * and execute each op. 
 */
MXReturnValue myExecutor(std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const std::string& subgraph_sym) {
  std::cout << "Info: subgraph symbol is: " << std::endl;
  std::cout << subgraph_sym << std::endl;

  // convert json string to json object
  JsonParser parser;
  JsonVal json_val = parser.parse_to_json(subgraph_sym);
  // get nodes list
  JsonVal nodes = json_val.map[JsonVal("nodes")];
  //counter for inputs
  int input_cnt = 0;
  // temporary tensor storage
  std::vector<MXTensor> data;
  // track memory allocations to free later
  std::vector<void*> to_free;

  // loop over nodes
  for(int i=0; i<nodes.list.size(); i++) {
    JsonVal node = nodes.list[i];
    // get the op name
    std::string op = node.map[JsonVal("op")].str;
    // get node ID inputs to op
    JsonVal node_inputs = node.map[JsonVal("inputs")];
    
    // handle each op type
    if (op.compare("null") == 0) {
      // null is an input data to the subgraph, add to data storage
      data.push_back(inputs->at(input_cnt++));
    } else if (op.compare("log") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor &input = data[node_inputs.list[0].list[0].num];
      // create temporary storage
      MXTensor tmp(malloc(input.size()*4), input.shape, input.dtype, 0, MXContext::CPU(0), kDefaultStorage);
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute log operator
      myLog(input,tmp);
      // add output tensor to data storage
      data.push_back(tmp);
    } else if (op.compare("exp") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor &input = data[node_inputs.list[0].list[0].num];
      // create temporary storage
      MXTensor tmp(malloc(input.size()*4), input.shape, input.dtype, 0, MXContext::CPU(0), kDefaultStorage);
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute exp operator 
      myExp(input,tmp);
      // add output tensor to data storage
      data.push_back(tmp);
    } else {
      std::cout << "Error! Unsupported op '" << op << "' found in myExecutor";
      // free allocated temporary storage
      for (void* ptr : to_free)
        free(ptr);
      return MX_FAIL;
    }
  }
  
  // get list of outputs from subgraph
  JsonVal heads = json_val.map[JsonVal("heads")];
  // copy all operator results to outputs of subgraph
  for (int j = 0; j < heads.list.size(); j++) {
    // get computed result
    MXTensor &result = data[heads.list[0].list[0].num];
    // get output tensor to pass to MX
    MXTensor &out = outputs->at(j);
    float *out_data = out.data<float>();
    float *res_data = result.data<float>();
    // loop and copy data
    for (int64_t i = 0; i < result.size(); i++) {
      out_data[i] = res_data[i];
    }
  }

  // free allocated temporary storage
  for (void* ptr : to_free) {
    free(ptr);
  }
  
  return MX_SUCCESS;
}

class MyStatefulOp : public CustomStatefulOp {
 public:
  explicit MyStatefulOp(const std::string& sym,
                        const std::unordered_map<std::string, std::string>& attrs)
    : subgraph_sym(sym), attrs_(attrs) {
    for (auto kv : attrs) {
      std::cout << "subgraphOp attributes: " << kv.first << " ==> " << kv.second << std::endl;
    }
  }

  MXReturnValue Forward(std::vector<MXTensor>* inputs,
                        std::vector<MXTensor>* outputs,
                        const OpResource& op_res) {
    return myExecutor(inputs, outputs, subgraph_sym);
  }

 private:
  const std::string subgraph_sym;
  const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue createOpState(const std::unordered_map<std::string, std::string>& attrs,
                            CustomStatefulOp** op_inst) {
  std::string serialized_subgraph = "[empty]";
  // MXNet subgraph is stored as Symbol in operator node attrs subgraphs field
  // custom subgraph is stored as json string in custom operator attrs map entry
  if (attrs.count(MX_STR_SUBGRAPH_SYM_JSON)) {
    // user can now parse json and run other custom ops inside subgraph
    serialized_subgraph = attrs.at(MX_STR_SUBGRAPH_SYM_JSON);
  }
  *op_inst = new MyStatefulOp(serialized_subgraph, attrs);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_OP(_custom_subgraph_op)
.setIsSubgraphOp()
.setCreateOpState(createOpState, "cpu");

const std::vector<std::string> op_names({"exp","log"});

MXReturnValue mySupportedOps(const std::string& json,
                             std::vector<int>* ids,
                             const std::unordered_map<std::string, std::string>& options) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }
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
    std::string shape;
    int dtype = -1;
    if(node.map.find(JsonVal("attrs")) != node.map.end()) {
      JsonVal attrs = node.map[JsonVal("attrs")];
      if(attrs.map.find(JsonVal("shape")) != attrs.map.end()) 
        shape = attrs.map[JsonVal("shape")].str;
      if(attrs.map.find(JsonVal("dtype")) != attrs.map.end())
        dtype = std::stoi(attrs.map[JsonVal("dtype")].str);
    }

    //check if op dtype is float, and if option was specified to require float types
    if((dtype == kFloat32 && options.count("reqFloat") > 0) || options.count("reqFloat") == 0) {
      //check if op is in whitelist
      if(std::find(op_names.begin(),op_names.end(),op.str.c_str()) != op_names.end()) {
        // found op in whitelist, set value to -1 to include op in any subgraph
        ids->at(i) = -1;
      }
    }
  }
  return MX_SUCCESS;
}

MXReturnValue myReviewSubgraph(const std::string& json, int subgraph_id, bool* accept,
                               const std::unordered_map<std::string, std::string>& options,
                               std::unordered_map<std::string, std::string>* attrs,
                               const std::unordered_map<std::string, MXTensor>& args,
                               const std::unordered_map<std::string, MXTensor>& aux) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }
  for (auto kv : args) {
    std::cout << "arg: " << kv.first << " ==> (";
    for (auto s : kv.second.shape)
      std::cout << s << ",";
    std::cout << ") [";
    for (int i=0; i<kv.second.size(); i++)
      std::cout << kv.second.data<float>()[i] << ", ";
    std::cout << "]" << std::endl;
  }

  // check if option `reqArgs` was specified, and if so check if args were provided
  if(options.count("reqArgs") > 0 && args.size() == 0) {
    *accept = false;
    std::cout << "rejecting subgraph since args were not provided" << std::endl;
    return MX_SUCCESS;
  }

  // check if option `reject` was specified, and if so check if value is 'True'
  if(options.count("reject") > 0 && options.at("reject").compare("True") == 0) {
    // if specified, reject the subgraph. this is only used for testing
    *accept = false;
    std::cout << "rejecting subgraph" << std::endl;
  } else {
    *accept = true;
    std::cout << "accepting subgraph" << std::endl;
    attrs->insert(std::pair<std::string,std::string>("myKey","myVal"));
  }
  return MX_SUCCESS;
}

REGISTER_PARTITIONER(myProp)
.addStrategy("strategy1", "_custom_subgraph_op")
.setSupportedOps("strategy1", mySupportedOps)
.setReviewSubgraph("strategy1", myReviewSubgraph);

class MySelector : public CustomOpSelector {
 public:
  MySelector(const std::string& json,
             const std::unordered_map<std::string, std::string>& options) :
    graph_json(json), options_(options) {
    for (auto kv : options) {
      std::cout << "selector options: " << kv.first
                << " ==> " << kv.second << std::endl;
    }
    //convert json string to json object
    JsonParser parser;
    JsonVal json_val = parser.parse_to_json(json);
    //get nodes list
    nodes = json_val.map[JsonVal("nodes")];
  }
  bool chooseNode(int nodeID) {
    JsonVal node = nodes.list[nodeID];
    JsonVal op = node.map[JsonVal("op")];

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

    //check if op dtype is float, and if option was specified to require float types
    if((dtype == kFloat32 && options_.count("reqFloat") > 0) || options_.count("reqFloat") == 0) {
      //check if op is in whitelist
      if(std::find(op_names.begin(),op_names.end(),op.str.c_str()) != op_names.end()) {
        // found op in whitelist, return true to include op subgraph
	return true;
      }
    }
    return false;
  }
  virtual bool Select(int nodeID) {
    return chooseNode(nodeID);
  }
  virtual bool SelectInput(int nodeID, int input_nodeID) {
    return chooseNode(input_nodeID);
  }
  virtual bool SelectOutput(int nodeID, int output_nodeID) {
    return chooseNode(output_nodeID);
  }
  virtual void Filter(std::vector<int>& candidates,
                      std::vector<int>& keep) {
    keep.insert(keep.end(), candidates.begin(), candidates.end());
  }
  virtual void Reset() {}
 private:
  std::string graph_json;
  JsonVal nodes;
  const std::unordered_map<std::string, std::string> options_;
};

MXReturnValue createSelector(const std::string& json, CustomOpSelector** sel_inst,
                             const std::unordered_map<std::string, std::string>& options) {
  *sel_inst = new MySelector(json, options);
  std::cout << "Info: selector created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_PARTITIONER(mySelect)
.addStrategy("strategy1", "_custom_subgraph_op")
.setCreateSelector("strategy1", createSelector)
.setReviewSubgraph("strategy1", myReviewSubgraph);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
