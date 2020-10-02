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
#include "mxnet/lib_api.h"

using namespace mxnet::ext;

/* function to execute log operator on floats */
void myLog(MXTensor *in, MXTensor *out) {
  float* inp = in->data<float>();
  float* outp = out->data<float>();
  for (int64_t i = 0; i < in->size(); i++) {
    outp[i] = logf(inp[i]);
  }
}
/* function to execute exp operator on floats */
void myExp(MXTensor *in, MXTensor *out) {
  float* inp = in->data<float>();
  float* outp =out->data<float>();
  for (int64_t i = 0; i < in->size(); i++) {
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
                         mxnet::ext::Graph *subgraph) {
  std::cout << "Info: subgraph is: " << std::endl;
  subgraph->print();

  //counter for inputs
  int input_cnt = 0;
  // temporary tensor storage
  std::vector<MXTensor> data;
  // track memory allocations to free later
  std::vector<void*> to_free;

  // loop over nodes
  for(int i=0; i<subgraph->size(); i++) {
    mxnet::ext::Node* node = subgraph->getNode(i);
    // handle each op type
    if (node->op.compare("null") == 0) {
      // set tensor for this input to the subgraph
      node->tensor = &inputs->at(input_cnt++);
    } else if (node->op.compare("log") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor *input = node->inputs.at(0).node->tensor;
      // create temporary storage
      MXTensor tmp(malloc(input->size()*4), input->shape, input->dtype, 0, MXContext::CPU(0), kDefaultStorage);  // NOLINT
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute log operator
      myLog(input,&tmp);
      // add output tensor to data storage
      data.push_back(tmp);
      // set tensor for this node so we can read it later
      node->tensor = &data.back();
    } else if (node->op.compare("exp") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor *input = node->inputs.at(0).node->tensor;
      // create temporary storage
      MXTensor tmp(malloc(input->size()*4), input->shape, input->dtype, 0, MXContext::CPU(0), kDefaultStorage);  // NOLINT
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute exp operator 
      myExp(input,&tmp);
      // add output tensor to data storage
      data.push_back(tmp);
      // set tensor for this node so we can read it later
      node->tensor = &data.back();
    } else {
      MX_ERROR_MSG << "Error! Unsupported op '" << node->op << "' found in myExecutor";
      // free allocated temporary storage
      for (void* ptr : to_free)
        free(ptr);
      return MX_FAIL;
    }
  }
  
  // copy all operator results to outputs of subgraph
  for (int j = 0; j < subgraph->outputs.size(); j++) {
    // get computed result
    MXTensor *result = subgraph->outputs[j].node->tensor;
    // get output tensor to pass to MX
    MXTensor &out = outputs->at(j);
    float *out_data = out.data<float>();
    float *res_data = result->data<float>();
    // loop and copy data
    for (int64_t i = 0; i < result->size(); i++) {
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
  explicit MyStatefulOp(std::string json,
                        const std::unordered_map<std::string, std::string>& attrs)
    : attrs_(attrs) {
    for (const auto &kv : attrs) {
      std::cout << "subgraphOp attributes: " << kv.first << " ==> " << kv.second << std::endl;
    }
    subgraph_ = mxnet::ext::Graph::fromString(json);
  }

  MXReturnValue Forward(std::vector<MXTensor>* inputs,
                        std::vector<MXTensor>* outputs,
                        const OpResource& op_res) override {
    if(attrs_.count(MX_STR_EXTRA_INPUTS) > 0 && std::stoi(attrs_.at(MX_STR_EXTRA_INPUTS)) > 0)
      std::cout << "forward::extra_inputs(" << attrs_.at(MX_STR_EXTRA_INPUTS) << ")::inputs ["
		<< inputs->size() << "]" << std::endl;
    return myExecutor(inputs, outputs, subgraph_);
  }

 private:
  mxnet::ext::Graph *subgraph_;
  const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue createOpState(const std::unordered_map<std::string, std::string>& attrs,
                            const MXContext& ctx,
                            const std::vector<std::vector<unsigned int> >& in_shapes,
                            const std::vector<int> in_types,
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

MXReturnValue mySupportedOps(const mxnet::ext::Graph* graph,
                             std::vector<int>* ids,
                             const std::unordered_map<std::string, std::string>& options) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }

  //loop over nodes
  for(int i=0; i<graph->size(); i++) {
    const mxnet::ext::Node *node = graph->getNode(i);

    //get shape/type if available
    std::string shape;
    int dtype = -1;
    if(node->attrs.count("shape") > 0)
      shape = node->attrs.at("shape");
    if(node->attrs.count("dtype") > 0)
      dtype = std::stoi(node->attrs.at("dtype"));

    //check if op dtype is float, and if option was specified to require float types
    if((dtype == kFloat32 && options.count("reqFloat") > 0) || options.count("reqFloat") == 0) {
      //check if op is in allowlist
      if(std::find(op_names.begin(),op_names.end(),node->op.c_str()) != op_names.end()) {
        // found op in allowlist, set value to -1 to include op in any subgraph
        ids->at(i) = -1;
      }
    }
  }
  return MX_SUCCESS;
}

MXReturnValue myReviewSubgraph(const mxnet::ext::Graph *subgraph, int subgraph_id, bool* accept,
                               const std::unordered_map<std::string, std::string>& options,
                               std::unordered_map<std::string, std::string>* attrs) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }

  std::string sg = subgraph->toString();
  std::cout << "subgraph " << subgraph_id << ": " << std::endl;
  std::cout << sg << std::endl;

  // check if option `reject` was specified, and if so check if value is 'True'
  if(options.count("reject") > 0 && options.at("reject").compare("True") == 0) {
    // if specified, reject the subgraph. this is only used for testing
    *accept = false;
    std::cout << "rejecting subgraph" << std::endl;
  } else {
    *accept = true;
    std::cout << "accepting subgraph" << std::endl;
  }

  attrs->emplace("myKey","myVal");

  return MX_SUCCESS;
}

REGISTER_PARTITIONER(myProp)
.addStrategy("strategy1", "_custom_subgraph_op")
.setSupportedOps("strategy1", mySupportedOps)
.setReviewSubgraph("strategy1", myReviewSubgraph);

class MySelector : public CustomOpSelector {
 public:
  MySelector(const mxnet::ext::Graph *graph,
             const std::unordered_map<std::string, std::string>& options) :
    graph_(graph), options_(options) {
    for (auto kv : options) {
      std::cout << "selector options: " << kv.first
                << " ==> " << kv.second << std::endl;
    }
  }
  bool chooseNode(int nodeID) {
    const mxnet::ext::Node *node = graph_->getNode(nodeID);

    //get shape/type if available
    std::string shape;
    int dtype = -1;
    if(node->attrs.count("shape") > 0)
      shape = node->attrs.at("shape");
    if(node->attrs.count("dtype") > 0)
      dtype = std::stoi(node->attrs.at("dtype"));

    //check if op dtype is float, and if option was specified to require float types
    if((dtype == kFloat32 && options_.count("reqFloat") > 0) || options_.count("reqFloat") == 0) {
      //check if op is in allowlist
      if(std::find(op_names.begin(),op_names.end(),node->op.c_str()) != op_names.end()) {
        // found op in allowlist, return true to include op subgraph
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
  const mxnet::ext::Graph *graph_;
  const std::unordered_map<std::string, std::string> options_;
};

MXReturnValue createSelector(const mxnet::ext::Graph *graph, CustomOpSelector** sel_inst,
                             const std::unordered_map<std::string, std::string>& options) {
  *sel_inst = new MySelector(graph, options);
  std::cout << "Info: selector created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_PARTITIONER(mySelect)
.addStrategy("strategy1", "_custom_subgraph_op")
.setCreateSelector("strategy1", createSelector)
.setReviewSubgraph("strategy1", myReviewSubgraph);

/* \brief a basic pass that adds a new input for subgraph ops */
MXReturnValue addInputPass(mxnet::ext::Graph *graph,
			   const std::unordered_map<std::string, std::string>& options) {
  //find node with '_custom_subgraph_op' op type
  for(int i=0; i<graph->size(); i++) {
    mxnet::ext::Node* n = graph->getNode(i);
    if(n->op.compare("_custom_subgraph_op") == 0) {
      //set extra input
      n->attrs[MX_STR_EXTRA_INPUTS] = std::to_string(1);
      
      //create a new input Node
      Node* input = graph->addNode(n->name + "_input", "null");
      //set this node as an input in the graph
      graph->inputs.push_back(input);
      //connect new input to node
      input->outputs.push_back({n,(int)(n->inputs.size())});
      //connect node to new input
      n->inputs.push_back({input,0});
      // add a corresponding tensor for this input
      input->alloc_arg({1},MXContext::CPU(0),kFloat32);
    }
  }

  return MX_SUCCESS;
}

REGISTER_PASS(addInputPass)
.setBody(addInputPass);

MXReturnValue initialize(int version) {
  if (version >= 10800) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported by custom library" << std::endl;
    return MX_FAIL;
  }
}
