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

#ifndef MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_PROPERTY_H_

#include <nnvm/node.h>
#include <dmlc/base.h>
#include <dmlc/thread_local.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace mxnet {
namespace op {

/*
 * This provides criteria for the graph partitioning algorithm to select
 * nodes to subgraphs.
 * The algorithm first sorts all the nodes in topological order, and then
 * loops through the sorted nodes and tries to find a subgraph starting
 * from each node (we call it a seed node) that satisfies the following two conditions:
 * 1. The node has not been selected before.
 * 2. The function Select is called on the node and returns true.
 *
 * Expanding from this seed node, we do BFS to traverse the graph.
 * During the traversal, we call SelectInput and SelectOutput to determine
 * if a neighboring node of the current node should be selected as a candidate for the subgraph.
 * The search continues when a new node is selected as a candidate, and terminates when no more
 * qualified nodes are found. When the search ends, all of the candidate nodes will
 * be passed to the function Filter to finalize the subgraph. The filtering gives
 * developers the last opportunity to drop off some of the candidate nodes.
 * By default, Filter returns all nodes as the subgraph nodes.
 * If the pre-selected subgraph becomes disconnected because some
 * nodes are filtered out in the Filter function, the algorithm will automatically convert
 * the rest of the nodes to multiple valid subgraphs based upon their connectivity.
 */
class SubgraphSelector {
 public:
  virtual ~SubgraphSelector() {}
  /*!
   * \brief Determines if to search for other nodes to form a subgraph from the seed_node.
   */
  virtual bool Select(const nnvm::Node &seed_node) = 0;
  /*!
   * \brief Determines if to select input_node when traverse to the cur_node.
   * \param cur_node the node for determining whether its input_node should be selected
   * \param input_node the input node of the cur_node
   * \return true if input_node is selected
   */
  virtual bool SelectInput(const nnvm::Node &cur_node, const nnvm::Node &input_node) = 0;
  /*!
   * \brief Determines if to select output_node when traverse to the cur_node.
   * \param cur_node the node for determining whether its output_node should be selected
   * \param output_node the output node of the cur_node
   * \return true if output_node is selected
   */
  virtual bool SelectOutput(const nnvm::Node &cur_node, const nnvm::Node &output_node) = 0;
  /*!
   * \brief Post processes pre-selected subgraph nodes. Return a list of nodes that
   *        users want to keep in subgraph(s).
   * \param candidates re-selected subgraph nodes to filt
   * \return a list of nodes to keep
   */
  virtual std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) {
    return candidates;
  }
};

using SubgraphSelectorPtr = std::shared_ptr<SubgraphSelector>;

/*!
 * \brief This provides a set of properties for partitioning a graph into subgraphs,
 *        reconstructing a new graph from the subgraphs and creating a subgraph
 *        operator to execute the subgraph.
 */
class SubgraphProperty {
 public:
  /*!
   * \brief The criteria of selecting the subgraph nodes.
   */
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const = 0;
  /*!
   * \brief Create an nnvm node for a given subgraph. Here users can customize how to
   *        execute the operators in the subgraph.
   * \param sym the symbol to create subgraph node
   * \param subgraph_id subgraph id
   */
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                           const int subgraph_id = 0) const = 0;
  /*!
   * \brief Connect subgraph internal output with external output entries.
   *        By default, each output entry will connect to an unique internal output.
   * \param subgraph_node the subgraph node to connect output
   * \param output_entries external output entries depending on this subgraph node
   */
  virtual void ConnectSubgraphOutputs(const nnvm::NodePtr subgraph_node,
                                      std::vector<nnvm::NodeEntry*>* output_entries) const {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      *output_entries->at(i) = nnvm::NodeEntry{subgraph_node, static_cast<uint32_t>(i), 0};
    }
  }
  /*!
   * \brief Connect subgraph internal input with external input entries.
   * By default, each input entry will connect in top sorted order.
   * \param subgraph_node the subgraph node to connect input
   * \param input_entries input entries inside subgraph
   * \param orig_input_entries input entries outside subgraph
   */
  virtual void ConnectSubgraphInputs(const nnvm::NodePtr subgraph_node,
                                     std::vector<nnvm::NodeEntry*>* input_entries,
                                     std::vector<nnvm::NodeEntry>* orig_input_entries) const {
    subgraph_node->inputs = *orig_input_entries;
  }
  /*!
   * \brief Set an attr with name in the attr map.
   */
  template<typename T>
  SubgraphProperty& SetAttr(const std::string& name, const T& value) {
    attrs_[name] = std::make_shared<dmlc::any>(value);
    return *this;
  }
  /*!
   * \brief Get the attr with the name.
   */
  template<typename T>
  const T& GetAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    CHECK(it != attrs_.end()) << "Cannot find attribute " << name << " in SubgraphProperty";
    return nnvm::get<T>(*it->second);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<nnvm::any>> attrs_;
};

using SubgraphPropertyPtr = std::shared_ptr<SubgraphProperty>;

class SubgraphPropertyRegistry {
 public:
  typedef SubgraphPropertyPtr (*SubgraphPropertyCreateFn)(void);
  static SubgraphPropertyRegistry* Get() {
    static SubgraphPropertyRegistry inst;
    return &inst;
  }

  SubgraphPropertyPtr CreateSubgraphProperty(const std::string& name) {
    auto it = prop_fn_map_.find(name);
    CHECK(it != prop_fn_map_.end()) << "SubgraphProperty " << name
                                    << " is not found in SubgraphPropertyRegistry";
    return it->second();
  }

  SubgraphPropertyCreateFn __REGISTER_OR_GET__(const std::string& name,
                                               SubgraphPropertyCreateFn fn) {
    if (prop_fn_map_.count(name) == 0U) {
      return __REGISTER__(name, fn);
    } else {
      return prop_fn_map_.at(name);
    }
  }

 private:
  SubgraphPropertyCreateFn __REGISTER__(const std::string& name, SubgraphPropertyCreateFn fn) {
    CHECK_EQ(prop_fn_map_.count(name), 0U) << "Subgraph property " << name
                                           << " has been registered";
    prop_fn_map_[name] = fn;
    return prop_fn_map_[name];
  }

  SubgraphPropertyRegistry() = default;
  SubgraphPropertyRegistry(const SubgraphPropertyRegistry&) = delete;
  SubgraphPropertyRegistry(SubgraphPropertyRegistry&&) = delete;
  SubgraphPropertyRegistry& operator=(const SubgraphPropertyRegistry&) = delete;
  std::unordered_map<std::string, SubgraphPropertyCreateFn> prop_fn_map_;
};

// This op name set is for setting the names of operators that should be grouped into
// subgraphs. In practice, every backend accelerator should have a predefined name set.
// This set is only used for the testing purpose.
// key: property name, value: op name set
typedef dmlc::ThreadLocalStore<std::unordered_map<std::string, std::unordered_set<std::string>>>
  SubgraphPropertyOpNameSet;

#define MXNET_REGISTER_SUBGRAPH_PROPERTY(Name, SubgraphPropertyType) \
  static DMLC_ATTRIBUTE_UNUSED auto __make_ ## SubgraphPropertyType ## _ ## Name ## __ = \
    SubgraphPropertyRegistry::Get()->__REGISTER_OR_GET__(#Name, &SubgraphPropertyType::Create)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_PROPERTY_H_
