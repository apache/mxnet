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

/* \brief a basic pass that prints out the options and the graph */
MXReturnValue myPass(mxnet::ext::Graph* g,
                     const std::unordered_map<std::string, std::string>& options) {
  for (auto kv : options) {
    std::cout << "option: " << kv.first << " ==> " << kv.second << std::endl;
  }
  //g->addNode("myConv","Convolution");
  mxnet::ext::Node *copy_old_layer = g->getNode(g->size()-2);
  mxnet::ext::Node *old_layer = g->getNode(g->size()-1);
  std::cout<<old_layer->name<<std::endl;

  mxnet::ext::Node *new_weight = g->addNode("myweight3","null");
  new_weight->alloc_arg({3,2}, MXContext::CPU(0),kFloat32);

  mxnet::ext::Node *new_bias = g->addNode("mybias3","null");
  new_bias->alloc_arg({3,}, MXContext::CPU(0),kFloat32);




  //new_bias->alloc_aux({2}, MXContext::CPU(0), kInt32);


  auto attrs = copy_old_layer->attrs;
  mxnet::ext::Node *new_layer = g->addNode("mylinaer","FullyConnected");
  auto new_attrs = &new_layer->attrs;
  for (auto it = attrs.begin(); it!=attrs.end(); it ++ )
  {

    std::cout<<it->first<<" : "<<it->second<<std::endl;
    if (it->first!="num_hidden"){
    new_attrs->insert({{it->first,it->second}});}
  }
  new_attrs->insert({{"num_hidden","3"}});

  std::cout<<"current:"<<std::endl;



  for (auto it = new_attrs->begin(); it!=new_attrs->end(); it ++ )
  {
    std::cout<<it->first<<" : "<<it->second<<std::endl;
  }

  new_layer->inputs.push_back({old_layer, 0});
  new_layer->inputs.push_back({new_weight, 0});
  new_layer->inputs.push_back({new_bias, 0});
  old_layer->outputs.push_back({new_layer, 0});
  new_weight->outputs.push_back({new_layer, 1});
  new_bias->outputs.push_back({new_layer, 2});
  auto it = g->outputs.begin();
  g->outputs.erase(it);
  mxnet::ext::NodeEntry new_entry;
  new_entry.node = new_layer;
  new_entry.entry = 0;
  g->outputs.push_back(new_entry);

  g->print();


  return MX_SUCCESS;
}

REGISTER_PASS(myPass).setBody(myPass);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
