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

struct BiDirectedNode;
using BiDirectedNodePtr = std::shared_ptr<BiDirectedNode>;

/*!
 * \brief Node of the undirected graph which replicates the network structures
 * of the computational graph. It is used to ease the graph traversal for finding
 * subgraphs.
 */
struct BiDirectedNode {
  static BiDirectedNodePtr Create() { return std::make_shared<BiDirectedNode>(); }
  BiDirectedNode() : label(-1), node(nullptr) {}
  /*! subgraph label */
  int label;
  /*! the original node in the computational graph it references*/
  nnvm::Node* node;
  /*!
   * \brief output nodes of the current node
   * key is node ptr and value is an array of indices standing for the entry indices
   * in key->inputs whose source is the current node.
   */
  std::unordered_map<nnvm::Node*, std::vector<size_t>> outputs;
};  // struct BiDirectedNode

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
  /*!
   * \brief Reset the state of selector for SelectInput.
   *        Note: the state should reset to Select() is successful.
   */
  virtual void Reset() {}
};

using SubgraphSelectorPtr = std::shared_ptr<SubgraphSelector>;

class SubgraphSelectorV2 {
 public:
  virtual ~SubgraphSelectorV2() {}
  /*!
   * \brief Determines if to search for other nodes to form a subgraph from the seed_node.
   */
  virtual bool Select(const BiDirectedNode& seed_node) = 0;
  /*!
   * \brief Determines if to select input_node when traverse to the cur_node.
   * \param cur_node the node for determining whether its input_node should be selected
   * \param input_node the input node of the cur_node
   * \return true if input_node is selected
   */
  virtual bool SelectInput(const BiDirectedNode& cur_node,
                           const BiDirectedNode& input_node) = 0;
  /*!
   * \brief Determines if to select output_node when traverse to the cur_node.
   * \param cur_node the node for determining whether its output_node should be selected
   * \param output_node the output node of the cur_node
   * \return true if output_node is selected
   */
  virtual bool SelectOutput(const BiDirectedNode& cur_node,
                            const BiDirectedNode& output_node) = 0;
  /*!
   * \brief Post processes pre-selected subgraph nodes. Return a list of nodes that
   *        users want to keep in subgraph(s).
   * \param candidates re-selected subgraph nodes to filt
   * \return a list of nodes to keep
   */
  virtual std::vector<BiDirectedNode*> Filter(
      const std::vector<BiDirectedNode*>& candidates) {
    return candidates;
  }

  /*!
   * \brief Reset the state of selector for SelectInput.
   *        Note: the state should reset to Select() is successful.
   */
  virtual void Reset() {}
};

using SubgraphSelectorV2Ptr = std::shared_ptr<SubgraphSelectorV2>;

class SubgraphSelectorV2Bridge : public SubgraphSelectorV2 {
 public:
  explicit SubgraphSelectorV2Bridge(SubgraphSelectorPtr ptr) : ss_ptr_(ptr) {}

  virtual ~SubgraphSelectorV2Bridge() {}

  bool Select(const BiDirectedNode& seed_node) override {
    return ss_ptr_->Select(*seed_node.node);
  }

  bool SelectInput(const BiDirectedNode& cur_node,
                   const BiDirectedNode& input_node) override {
    return ss_ptr_->SelectInput(*cur_node.node, *input_node.node);
  }

  bool SelectOutput(const BiDirectedNode& cur_node,
                    const BiDirectedNode& output_node) override {
    return ss_ptr_->SelectOutput(*cur_node.node, *output_node.node);
  }

  std::vector<BiDirectedNode*> Filter(
      const std::vector<BiDirectedNode*>& candidates) override {
    std::unordered_map<nnvm::Node*, BiDirectedNode*> node_2_snode_map;
    std::vector<nnvm::Node*> n_candidates;
    for (auto i : candidates) {
      node_2_snode_map[i->node] = i;
      n_candidates.push_back(i->node);
    }
    auto n_ret = ss_ptr_->Filter(n_candidates);
    std::vector<BiDirectedNode*> ret;
    for (auto i : n_ret) ret.push_back(node_2_snode_map[i]);
    return ret;
  }

  void Reset() override { ss_ptr_->Reset(); }

  const SubgraphSelectorPtr& GetV1ptr() const { return ss_ptr_; }

 private:
  SubgraphSelectorPtr ss_ptr_;
};

/*!
 * \brief This provides a set of properties for partitioning a graph into subgraphs,
 *        reconstructing a new graph from the subgraphs and creating a subgraph
 *        operator to execute the subgraph.
 */
class SubgraphProperty {
 public:
  /*! \brief Property type */
  enum SgPropertyType {
    kCreate,
    kAdjust,
  };

  explicit SubgraphProperty(SgPropertyType type = kCreate) : type_(type) {}

  /*!
   * \brief The criteria of selecting the subgraph nodes.
   */
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    LOG(FATAL) << "No CreateSubgraphSelector is implemented for this SubgraphProperty.";
    return nullptr;
  }

  virtual SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const {
    auto v1_ptr = CreateSubgraphSelector();
    return std::make_shared<SubgraphSelectorV2Bridge>(v1_ptr);
  }

  /*!
   * \brief Create an nnvm node for a given subgraph. Here users can customize how to
   *        execute the operators in the subgraph.
   * \param sym the symbol to create subgraph node
   * \param subgraph_id subgraph id
   */
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                           const int subgraph_id = 0) const {
    CHECK_EQ(GetPropertyType(), kCreate);
    LOG(FATAL) << "Not implement CreateSubgraphNode() for this subgraph property.";
    return nullptr;
  }

  /*!
   * \brief Create an nnvm node for a given subgraph. Here users can customize how to
   *        execute the operators in the subgraph.
   * \param sym the symbol to create subgraph node
   * \param subgraph_selector the selector used for creating this subgraph
   * \param subgraph_id subgraph id
   */
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                           const SubgraphSelectorPtr& subgraph_selector,
                                           const int subgraph_id = 0) const {
    return CreateSubgraphNode(sym, subgraph_id);
  }

  /*!
   * \brief Create an nnvm node for a given subgraph. Here users can customize how to
   *        execute the operators in the subgraph.
   * \param sym the symbol to create subgraph node
   * \param subgraph_selector The selector used for selecting this node set
   * \param subgraph_id subgraph id
   */
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                           const SubgraphSelectorV2Ptr& subgraph_selector,
                                           const int subgraph_id = 0) const {
    CHECK_EQ(GetPropertyType(), kCreate);
    const auto bridge = static_cast<SubgraphSelectorV2Bridge*>(subgraph_selector.get());
    return CreateSubgraphNode(sym, bridge->GetV1ptr(), subgraph_id);
  }

  /*!
   * \brief Adjust nnvm nodes from a given subgraph. No new node is created, but adjust
   *        selected nodes' attributes. This can be used to implement peephole optimization.
   *        Here users can customize how to adjust the operators in the subgraph.
   * \param subgraph_nodes the subgraph nodes to adjust
   * \param subgraph_selector The selector used for selecting this node set.
   * \param subgraph_id subgraph id
   */
  virtual void AdjustSubgraphNode(const std::vector<nnvm::Node*>& subgraph_nodes,
                                  const SubgraphSelectorV2Ptr &subgraph_selector,
                                  const int subgraph_id = 0) const {
    CHECK_EQ(GetPropertyType(), kAdjust);
    LOG(FATAL) << "Not implement AdjustSubgraphNode() for this subgraph property.";
  }

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
  /*!
   * \brief Check if the attr exists.
   */
  bool HasAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    return it != attrs_.end();
  }
  /*!
   * \brief Get the property type.
   */
  SgPropertyType GetPropertyType() const { return type_; }

 protected:
  SgPropertyType type_;
  std::unordered_map<std::string, std::shared_ptr<nnvm::any>> attrs_;
};

using SubgraphPropertyPtr = std::shared_ptr<SubgraphProperty>;

class SubgraphPropertyEntry {
 public:
  explicit SubgraphPropertyEntry(std::shared_ptr<SubgraphProperty> entry) : entry_(entry) {}
  SubgraphPropertyEntry set_attr(const std::string& name, const int value) const {
    entry_->SetAttr<int>(name, value);
    return *this;
  }

 private:
  std::shared_ptr<SubgraphProperty> entry_;
};

class SubgraphPropertyRegistry {
 public:
  typedef SubgraphPropertyPtr (*SubgraphPropertyCreateFn)(void);
  static SubgraphPropertyRegistry* Get() {
    static SubgraphPropertyRegistry inst;
    return &inst;
  }

  std::vector<SubgraphPropertyPtr> CreateSubgraphProperty(const std::string& name) {
    auto it = prop_ptr_map_.find(name);
    CHECK(it != prop_ptr_map_.end()) << "SubgraphProperty " << name
                                    << " is not found in SubgraphPropertyRegistry";
    return it->second;
  }

  SubgraphPropertyEntry __REGISTER__(const std::string& name, SubgraphPropertyCreateFn fn) {
    prop_ptr_map_[name].emplace_back(fn());
    return SubgraphPropertyEntry(prop_ptr_map_[name].back());
  }

  SubgraphPropertyRegistry() = default;
  SubgraphPropertyRegistry(const SubgraphPropertyRegistry&) = delete;
  SubgraphPropertyRegistry(SubgraphPropertyRegistry&&) = delete;
  SubgraphPropertyRegistry& operator=(const SubgraphPropertyRegistry&) = delete;
  std::unordered_map<std::string, std::vector<SubgraphPropertyPtr>> prop_ptr_map_;
};

// This op name set is for setting the names of operators that should be grouped into
// subgraphs. In practice, every backend accelerator should have a predefined name set.
// This set is only used for the testing purpose.
// key: property name, value: op name set
typedef dmlc::ThreadLocalStore<std::unordered_map<std::string, std::unordered_set<std::string>>>
  SubgraphPropertyOpNameSet;

#define DECLARE_PROPERTY_EX(NAME, SubgraphPropertyType, X) \
  static const DMLC_ATTRIBUTE_UNUSED auto __make_##SubgraphPropertyType##_##Name##_##X##__
#define DECLARE_PROPERTY(NAME, SubgraphPropertyType, X) \
  DECLARE_PROPERTY_EX(NAME, SubgraphPropertyType, X)

#define MXNET_REGISTER_SUBGRAPH_PROPERTY(Name, SubgraphPropertyType) \
  DECLARE_PROPERTY(Name, SubgraphPropertyType, __LINE__) =           \
      SubgraphPropertyRegistry::Get()->__REGISTER__(#Name, &SubgraphPropertyType::Create)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_PROPERTY_H_
