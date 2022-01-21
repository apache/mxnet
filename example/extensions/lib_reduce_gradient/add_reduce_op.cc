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
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation library file
 */

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include "mxnet/lib_api.h"

using namespace mxnet::ext;



MXReturnValue add_reduce_op(mxnet::ext::Graph* g,
                     const std::unordered_map<std::string, std::string>& options) {
  std::string cur_rank = "";

  std::string num_gpus = "";
  std::string nccl_unique_id = "";

  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
    if (kv.first == "rank")
    {
        cur_rank = kv.second.c_str();
    }
    if (kv.first == "nccl_unique_id")
        nccl_unique_id = kv.second.c_str();
    if (kv.first == "num_gpus")
        num_gpus = kv.second.c_str();
    }
  size_t length = g->size();
  mxnet::ext::Node *tmp;
  std::string root_rank;
  mxnet::ext::Node *target_node;
  int index = 0;
  for (int i = 0;i < length; i += 1)
  {
    target_node = g->getNode(i);
    //std::cout<<"deal with:" << target_node->name<<std::endl;
    auto it = options.find(target_node->name);
    if (it == options.end()) {continue;} // req_grad == null
    root_rank = it->second;
    mxnet::ext::Node *new_reduce = g->addNode("ncclreduce_" + target_node->name,"_contrib_NCCLReduce");
    index += 1;
    auto new_attrs = &new_reduce->attrs;
    auto old_attrs = target_node->attrs;
    for (auto it = old_attrs.begin(); it!=old_attrs.end(); it++)
    {
        if (it->first == "__ext_dtype__" || it->first == "__ext_shape__" || it->first == "__profiler_scope__")
        {
            new_attrs ->insert({{it->first, it->second}});
        }
    }
    new_attrs->insert({{"nccl_unique_id", nccl_unique_id}});
    new_attrs->insert({{"num_gpus", num_gpus}});
    new_attrs->insert({{"rank", cur_rank}});
    new_attrs->insert({{"root_rank", root_rank}});

  for (int i=0;i<target_node->outputs.size(); i++)
  {
     new_reduce->outputs.push_back(target_node->outputs[i]);
     mxnet::ext::Node *output_node = target_node->outputs[i].node;
     int index = target_node->outputs[i].entry;
     //std::cout<<"try change:"<<output_node->name<<":"<<output_node->inputs.size()<<std::endl;
     output_node->inputs[index].node = new_reduce;
  }
  for (int i=0;i<target_node->outputs.size(); i++)
  {
     target_node->outputs.pop_back();
  }
  target_node->outputs.push_back({new_reduce, 0});
  new_reduce->inputs.push_back({target_node, 0});

  }
  g->print();


  return MX_SUCCESS;
}



REGISTER_PASS(add_reduce_op).setBody(add_reduce_op);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
