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
 * \file custom_subgraph_property.h
 *
 * This file contains an implementation of a subgraph property
 * that interfaces between MXNet and custom subgraph properties
 * created by users in external libraries. It does not implement
 * any custom subgraphing logic itself, rather it calls APIs
 * in the user's custom library to enable control of partitioning
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_

#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "../common.h"
#include "../subgraph_property.h"
#include "../../include/mxnet/lib_api.h"

namespace mxnet {
namespace op {

/*!
 * This selects nodes for a subgraph based on node name as supplied
 * by the supportedOps from an external library. It visits nodes via
 * both input and output links.
 */
class CustomContainOpSelector : public SubgraphSelector {
 public:
  explicit CustomContainOpSelector(std::unordered_map<std::string, int> supported_nodes,
                                   void* sel_inst,
                                   mxnet::ext::partCallSelect_t callSelect,
                                   mxnet::ext::partCallSelectInput_t callSelectInput,
                                   mxnet::ext::partCallSelectOutput_t callSelectOutput,
                                   mxnet::ext::partCallFilter_t callFilter,
                                   mxnet::ext::partCallReset_t callReset,
                                   mxnet::ext::opCallFree_t callFree,
                                   std::unordered_map<const nnvm::Node*, unsigned> node2id)
      : supported_nodes_(supported_nodes),
        sel_inst_(sel_inst),
        callSelect_(callSelect),
        callSelectInput_(callSelectInput),
        callSelectOutput_(callSelectOutput),
        callFilter_(callFilter),
        callReset_(callReset),
        callFree_(callFree),
        node2id_(node2id) {}
  virtual bool Select(const nnvm::Node& n) {
    if (!sel_inst_) {
      return supported_nodes_.count(n.attrs.name) > 0;
    } else {
      int selected = 0;
      callSelect_(sel_inst_, node2id_[&n], &selected);
      return selected;
    }
  }
  virtual bool SelectInput(const nnvm::Node& n, const nnvm::Node& new_node) {
    if (!sel_inst_) {
      // check that op type is supported and that both nodes have the same ID
      // or the new node 's subgraph ID is any (-1)
      return supported_nodes_.count(new_node.attrs.name) > 0 &&
             (supported_nodes_[n.attrs.name] == supported_nodes_[new_node.attrs.name] ||
              supported_nodes_[new_node.attrs.name] == -1);
    } else {
      int selected = 0;
      callSelectInput_(sel_inst_, node2id_[&n], node2id_[&new_node], &selected);
      return selected;
    }
  }
  virtual bool SelectOutput(const nnvm::Node& n, const nnvm::Node& new_node) {
    if (!sel_inst_) {
      // check that op type is supported and that both nodes have the same ID
      // or the new node 's subgraph ID is any (-1)
      return supported_nodes_.count(new_node.attrs.name) > 0 &&
             (supported_nodes_[n.attrs.name] == supported_nodes_[new_node.attrs.name] ||
              supported_nodes_[new_node.attrs.name] == -1);
    } else {
      int selected = 0;
      callSelectOutput_(sel_inst_, node2id_[&n], node2id_[&new_node], &selected);
      return selected;
    }
  }
  virtual std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) {
    if (!sel_inst_) {
      return candidates;
    } else {
      std::unordered_map<int, nnvm::Node*> rev_map;
      std::vector<int> cand;
      for (nnvm::Node* node : candidates) {
        cand.push_back(node2id_[node]);
        rev_map[node2id_[node]] = node;
      }
      int* keep_   = nullptr;
      int num_keep = 0;
      callFilter_(sel_inst_, cand.data(), cand.size(), &keep_, &num_keep);
      std::vector<nnvm::Node*> keep;
      for (int i = 0; i < num_keep; i++) {
        keep.push_back(rev_map[keep_[i]]);
      }
      callFree_(keep_);
      return keep;
    }
  }
  virtual void Reset() {
    if (sel_inst_)
      return callReset_(sel_inst_);
  }

  std::unordered_map<std::string, int> supported_nodes_;
  void* sel_inst_;
  mxnet::ext::partCallSelect_t callSelect_;
  mxnet::ext::partCallSelectInput_t callSelectInput_;
  mxnet::ext::partCallSelectOutput_t callSelectOutput_;
  mxnet::ext::partCallFilter_t callFilter_;
  mxnet::ext::partCallReset_t callReset_;
  mxnet::ext::opCallFree_t callFree_;
  std::unordered_map<const nnvm::Node*, unsigned> node2id_;
};

/*!
 * This subgraph property finds a subgraph that only contains
 * nodes as specified by the supportedOps from an external library.
 * The operators in the subgraph will be executed by the operator
 * specified by the external library too.
 */
class CustomSubgraphProperty : public SubgraphProperty {
 public:
  CustomSubgraphProperty()
      : subgraph_prop("error"),
        call_supported_ops_(nullptr),
        supported_ops_(nullptr),
        call_create_selector_(nullptr),
        create_selector_(nullptr),
        callSelect_(nullptr),
        callSelectInput_(nullptr),
        callSelectOutput_(nullptr),
        callFilter_(nullptr),
        callReset_(nullptr),
        call_review_subgraph_(nullptr),
        review_subgraph_(nullptr),
        subgraph_op_name("error") {}
  CustomSubgraphProperty(std::string subgraph_prop_name,
                         mxnet::ext::partCallSupportedOps_t call_supported_ops,
                         mxnet::ext::supportedOps_t supported_ops,
                         mxnet::ext::partCallCreateSelector_t call_create_selector,
                         mxnet::ext::createSelector_t create_selector,
                         mxnet::ext::partCallSelect_t callSelect,
                         mxnet::ext::partCallSelectInput_t callSelectInput,
                         mxnet::ext::partCallSelectOutput_t callSelectOutput,
                         mxnet::ext::partCallFilter_t callFilter,
                         mxnet::ext::partCallReset_t callReset,
                         mxnet::ext::partCallReviewSubgraph_t call_review_subgraph,
                         mxnet::ext::reviewSubgraph_t review_subgraph,
                         mxnet::ext::opCallFree_t call_free,
                         std::string op_name)
      : subgraph_prop(subgraph_prop_name),
        call_supported_ops_(call_supported_ops),
        supported_ops_(supported_ops),
        call_create_selector_(call_create_selector),
        create_selector_(create_selector),
        callSelect_(callSelect),
        callSelectInput_(callSelectInput),
        callSelectOutput_(callSelectOutput),
        callFilter_(callFilter),
        callReset_(callReset),
        call_review_subgraph_(call_review_subgraph),
        review_subgraph_(review_subgraph),
        call_free_(call_free),
        subgraph_op_name(op_name) {}

  // create custom subgraph property
  static SubgraphPropertyPtr Create() {
    return std::make_shared<CustomSubgraphProperty>();
  }

  void PrePartition(const nnvm::Graph& g,
                    const std::unordered_map<std::string, std::string>& options_map) {
    // clear supported_nodes to remove state from previous calls
    supported_nodes.clear();
    // get input args and arg names
    in_arg_names = g.GetAttr<std::vector<std::string>>("in_arg_names");
    in_args_ptr  = g.GetAttr<NDArray**>("in_args");
    in_aux_names = g.GetAttr<std::vector<std::string>>("in_aux_names");
    in_aux_ptr   = g.GetAttr<NDArray**>("in_aux");

    // convert input args
    arg_names.clear();
    arg_data.clear();
    arg_shapes.clear();
    arg_dims.clear();
    arg_types.clear();
    arg_verIDs.clear();
    arg_dev_type.clear();
    arg_dev_id.clear();
    for (size_t i = 0; i < in_arg_names.size(); i++) {
      if (in_args_ptr[i] != nullptr) {
        arg_names.push_back(in_arg_names[i].c_str());
        const NDArray& in_arg = *(in_args_ptr[i]);

#if MXNET_USE_ONEDNN == 1
        // reorder data if in DNNL format
        if (in_arg.IsDNNLData()) {
          in_arg.Reorder2DefaultAsync();
          in_arg.WaitToRead();
        }
#endif

        // pull out parts of NDArray to send to backend
        arg_data.push_back(in_arg.data().dptr_);
        arg_shapes.push_back(in_arg.shape().data());
        arg_dims.push_back(in_arg.shape().ndim());
        arg_types.push_back(in_arg.dtype());
        arg_verIDs.push_back(in_arg.version());
        const char* arg_ctx_str = in_arg.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
        arg_dev_type.push_back(arg_ctx_str);
        arg_dev_id.push_back(in_arg.ctx().real_dev_id());
      }
    }

    // convert input aux
    aux_names.clear();
    aux_data.clear();
    aux_shapes.clear();
    aux_dims.clear();
    aux_types.clear();
    aux_verIDs.clear();
    aux_dev_type.clear();
    aux_dev_id.clear();
    for (size_t i = 0; i < in_aux_names.size(); i++) {
      if (in_aux_ptr[i] != nullptr) {
        aux_names.push_back(in_aux_names[i].c_str());
        const auto& in_aux = *(in_aux_ptr[i]);

#if MXNET_USE_ONEDNN == 1
        // reorder data if in DNNL format
        if (in_aux.IsDNNLData()) {
          in_aux.Reorder2DefaultAsync();
          in_aux.WaitToRead();
        }
#endif

        // pull out parts of NDArray to send to backend
        aux_data.push_back(in_aux.data().dptr_);
        aux_shapes.push_back(in_aux.shape().data());
        aux_dims.push_back(in_aux.shape().ndim());
        aux_types.push_back(in_aux.dtype());
        aux_verIDs.push_back(in_aux.version());
        const char* aux_ctx_str = in_aux.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
        aux_dev_type.push_back(aux_ctx_str);
        aux_dev_id.push_back(in_aux.ctx().real_dev_id());
      }
    }

    // remove all graph attrs, some cannot be saved to json
    nnvm::Graph graph = std::move(g);
    graph.attrs.clear();
    const nnvm::IndexedGraph& indexed_graph = graph.indexed_graph();

    // create map from nnvm::Node to nid
    node2id.clear();
    for (unsigned nid = 0; nid < indexed_graph.num_nodes(); nid++) {
      nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
      node2id[node]    = nid;
    }

    // set shape attrs for each node in the graph
    if (g.HasAttr("shape")) {
      mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
      for (unsigned nid = 0; nid < indexed_graph.num_nodes(); nid++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
        std::stringstream ss;
        ss << "[";
        // set the output shapes for this node
        for (unsigned oid = 0; oid < node->num_outputs(); oid++) {
          const uint32_t out_entry_id = indexed_graph.entry_id(nid, oid);
          mxnet::TShape& shape        = shapes[out_entry_id];
          if (shape.ndim() == -1)
            ss << "[None]";
          else
            ss << shape;
          if (oid < node->num_outputs() - 1)
            ss << ",";
        }
        ss << "]";
        node->attrs.dict[MX_STR_SHAPE] = ss.str();
      }
    }

    // set dtype attrs for each node in the graph
    if (g.HasAttr("dtype")) {
      std::vector<int> dtypes = g.GetAttr<std::vector<int>>("dtype");
      for (unsigned nid = 0; nid < indexed_graph.num_nodes(); nid++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
        std::stringstream ss;
        ss << "[";
        // set the output dtypes for this node
        for (unsigned oid = 0; oid < node->num_outputs(); oid++) {
          const uint32_t out_entry_id = indexed_graph.entry_id(nid, oid);
          int dtype                   = dtypes[out_entry_id];
          ss << dtype;
          if (oid < node->num_outputs() - 1)
            ss << ",";
        }
        ss << "]";
        node->attrs.dict[MX_STR_DTYPE] = ss.str();
      }
    }

    std::string graph_json = nnvm::pass::SaveJSON(graph);
    const char* json       = graph_json.c_str();

    // clear options from previous call
    opt_keys_.clear();
    opt_vals_.clear();
    options_map_.clear();
    // store options in map in subgraph property to re-use later for reviewSubgraph
    options_map_.insert(options_map.begin(), options_map.end());

    // convert options_map_ to char* to pass to backend library
    for (auto& kv : options_map_) {
      opt_keys_.push_back(kv.first.c_str());
      opt_vals_.push_back(kv.second.c_str());
    }

    // check if supportedOps was registered
    if (supported_ops_ && call_supported_ops_) {
      // setup array of subgraph IDs for each node
      std::vector<int> supported_node_IDs(indexed_graph.num_nodes(), -2);
      int* ids = supported_node_IDs.data();
      // call supportedOps
      CHECK(call_supported_ops_(supported_ops_,
                                json,
                                supported_node_IDs.size(),
                                ids,
                                opt_keys_.data(),
                                opt_vals_.data(),
                                opt_keys_.size()))
          << "Error calling supported_ops for '" << subgraph_prop << "'";

      const auto& idx = g.indexed_graph();
      // loop and add node names for each supported node ID
      for (unsigned i = 0; i < supported_node_IDs.size(); i++) {
        if (supported_node_IDs[i] != -2) {
          supported_nodes[idx[i].source->attrs.name] = supported_node_IDs[i];
        }
      }
    } else if (call_create_selector_ && callSelect_ && callSelectInput_ && callSelectOutput_ &&
               callFilter_ && callReset_ && create_selector_) {
      sel_inst = nullptr;
      CHECK(call_create_selector_(
          create_selector_, json, &sel_inst, opt_keys_.data(), opt_vals_.data(), opt_keys_.size()))
          << "Error calling supported_ops for '" << subgraph_prop << "'";
    } else {
      CHECK(supported_ops_ != nullptr)
          << "supported_ops_ is null for " << subgraph_prop << std::endl;
      CHECK(call_supported_ops_ != nullptr)
          << "call_supported_ops_ is null for " << subgraph_prop << std::endl;
    }
  }
  // override CreateSubgraphNode
  virtual nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                             const int subgraph_id = 0) const {
    int accept   = 1;
    int num_attr = 0;
    std::map<std::string, std::string> user_attrs;
    char** attr_keys = nullptr;
    char** attr_vals = nullptr;
    if (review_subgraph_) {
      nnvm::Graph g;
      g.outputs       = sym.outputs;
      const auto& idx = g.indexed_graph();

      // set isArg/isAux for each null op/param in the graph
      const std::vector<std::string> aux_state_names =
          sym.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      std::unordered_set<std::string> aux_set(aux_state_names.begin(), aux_state_names.end());
      for (unsigned i = 0; i < idx.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(idx[i].source);
        // check if this node is input to subgraph
        if (node->is_variable()) {
          // check if this node is an aux param
          if (aux_set.count(node->attrs.name))
            node->attrs.dict["isAux"] = "True";
          else
            node->attrs.dict["isAux"] = "False";
        }
      }

      std::string subgraph_json = nnvm::pass::SaveJSON(g);
      CHECK(call_review_subgraph_(review_subgraph_,
                                  subgraph_json.c_str(),
                                  subgraph_id,
                                  &accept,
                                  opt_keys_.data(),
                                  opt_vals_.data(),
                                  opt_keys_.size(),
                                  &attr_keys,
                                  &attr_vals,
                                  &num_attr,
                                  arg_names.data(),
                                  arg_names.size(),
                                  arg_data.data(),
                                  arg_shapes.data(),
                                  arg_dims.data(),
                                  arg_types.data(),
                                  arg_verIDs.data(),
                                  arg_dev_type.data(),
                                  arg_dev_id.data(),
                                  aux_names.data(),
                                  aux_names.size(),
                                  aux_data.data(),
                                  aux_shapes.data(),
                                  aux_dims.data(),
                                  aux_types.data(),
                                  aux_verIDs.data(),
                                  aux_dev_type.data(),
                                  aux_dev_id.data()))
          << "Error calling review_subgraph for '" << subgraph_prop << "'";

      if (num_attr > 0) {
        // set user specified attributes
        for (int i = 0; i < num_attr; i++) {
          user_attrs[attr_keys[i]] = attr_vals[i];
          call_free_(attr_vals[i]);
          call_free_(attr_keys[i]);
        }
        // free memory used by custom op to allocate attributes
        call_free_(attr_vals);
        call_free_(attr_keys);
      }
    }

    if (accept) {
      nnvm::ObjectPtr n = nnvm::Node::Create();
      n->attrs.op       = Op::Get(subgraph_op_name);
      n->attrs.name     = "_op" + std::to_string(subgraph_id);
      n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));

      // set shapes
      {
        std::stringstream ss;
        ss << "[";
        for (unsigned i = 0; i < sym.outputs.size(); i++) {
          const nnvm::NodeEntry& e = sym.outputs[i];
          if (e.node->attrs.dict.count(MX_STR_SHAPE) > 0) {
            std::string& shape = e.node->attrs.dict[MX_STR_SHAPE];
            // add this shape to the list
            ss << mxnet::ext::getShapeAt(shape, e.index);
          }
          if (i < sym.outputs.size() - 1)
            ss << ",";
        }
        ss << "]";
        n->attrs.dict[MX_STR_SHAPE] = ss.str();
      }
      // set dtypes
      {
        std::stringstream ss;
        ss << "[";
        for (unsigned i = 0; i < sym.outputs.size(); i++) {
          const nnvm::NodeEntry& e = sym.outputs[i];
          if (e.node->attrs.dict.count(MX_STR_DTYPE) > 0) {
            std::string& dtype = e.node->attrs.dict[MX_STR_DTYPE];
            // add this dtype to the list
            ss << mxnet::ext::getDtypeAt(dtype, e.index);
          }
          if (i < sym.outputs.size() - 1)
            ss << ",";
        }
        ss << "]";
        n->attrs.dict[MX_STR_DTYPE] = ss.str();
      }
      // set user specified attributes
      for (auto attr : user_attrs)
        n->attrs.dict[attr.first] = attr.second;
      return n;
    } else {
      return nullptr;
    }
  }

  virtual void InitSubgraphInputs(std::vector<nnvm::NodeEntry*>* input_entries,
                                  std::vector<nnvm::NodeEntry>* orig_input_entries) const {
    for (size_t i = 0; i < input_entries->size(); ++i) {
      nnvm::NodeEntry* e    = input_entries->at(i);
      nnvm::NodeEntry& orig = orig_input_entries->at(i);

      // set attribute for subgraph input to indicate if it is from an arg/param to model
      if (orig.node->is_variable()) {
        // get name of original output entry
        nnvm::Symbol sym;
        sym.outputs.push_back(orig);
        const auto output_names = sym.ListOutputNames();
        CHECK_EQ(output_names.size(), 1U);
        const std::string& var_name = output_names[0];

        e->node->attrs.dict["isArg"]   = "True";
        e->node->attrs.dict["argName"] = var_name;
      } else {
        e->node->attrs.dict["isArg"] = "False";
      }

      // pass down other attributes if available
      if (orig.node->attrs.dict.count(MX_STR_DTYPE) > 0) {
        // get dtype string from other node
        std::string& dtype = orig.node->attrs.dict[MX_STR_DTYPE];
        std::stringstream ss;
        ss << "[" << mxnet::ext::getDtypeAt(dtype, orig.index) << "]";
        e->node->attrs.dict[MX_STR_DTYPE] = ss.str();
      }

      if (orig.node->attrs.dict.count(MX_STR_SHAPE) > 0) {
        // get shape string from other node
        std::string& shape = orig.node->attrs.dict[MX_STR_SHAPE];
        // create new shape string for this node
        std::stringstream ss;
        ss << "[" << mxnet::ext::getShapeAt(shape, orig.index) << "]";
        e->node->attrs.dict[MX_STR_SHAPE] = ss.str();
      }
    }
  }

  // override CreateSubgraphSelector
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<CustomContainOpSelector>(supported_nodes,
                                                     sel_inst,
                                                     callSelect_,
                                                     callSelectInput_,
                                                     callSelectOutput_,
                                                     callFilter_,
                                                     callReset_,
                                                     call_free_,
                                                     node2id);
  }

  std::string subgraph_prop;
  mxnet::ext::partCallSupportedOps_t call_supported_ops_;
  mxnet::ext::supportedOps_t supported_ops_;
  mxnet::ext::partCallCreateSelector_t call_create_selector_;
  mxnet::ext::createSelector_t create_selector_;
  mxnet::ext::partCallSelect_t callSelect_;
  mxnet::ext::partCallSelectInput_t callSelectInput_;
  mxnet::ext::partCallSelectOutput_t callSelectOutput_;
  mxnet::ext::partCallFilter_t callFilter_;
  mxnet::ext::partCallReset_t callReset_;
  mxnet::ext::partCallReviewSubgraph_t call_review_subgraph_;
  mxnet::ext::reviewSubgraph_t review_subgraph_;
  mxnet::ext::opCallFree_t call_free_;
  std::unordered_map<std::string, int> supported_nodes;
  std::string subgraph_op_name;
  std::unordered_map<std::string, std::string> options_map_;
  std::vector<const char*> opt_keys_, opt_vals_;
  std::vector<std::string> in_arg_names, in_aux_names;
  NDArray** in_args_ptr;
  NDArray** in_aux_ptr;
  std::vector<const char*> arg_names, aux_names;
  std::vector<void*> arg_data, aux_data;
  std::vector<const int64_t*> arg_shapes, aux_shapes;
  std::vector<int> arg_dims, aux_dims;
  std::vector<int> arg_types, aux_types;
  std::vector<size_t> arg_verIDs, aux_verIDs;
  std::vector<const char*> arg_dev_type, aux_dev_type;
  std::vector<int> arg_dev_id, aux_dev_id;
  void* sel_inst = nullptr;
  std::unordered_map<const nnvm::Node*, unsigned> node2id;
};
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
