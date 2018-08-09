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
 * This provides criteria for selecting nodes in a subgraph.
 * When a node is passed to this object, the selection criteria may be changed.
 * We can also specify what links we should use when traversing the neighbor
 * nodes.
 */
class SubgraphSelector {
 public:
  virtual ~SubgraphSelector() {}
  // Determine if the node should be selected for a subgraph.
  virtual bool Select(const nnvm::Node &n) = 0;
  // Determine if the input node should be selected for a subgraph.
  virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) = 0;
  // Determine if the output node should be selected for a subgraph.
  virtual bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) = 0;
  // Post processes pre-selected subgraph nodes. Return a list of nodes that
  // users want to keep in subgraph(s).
  virtual std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) {
    return candidates;
  }
};

using SubgraphSelectorPtr = std::shared_ptr<SubgraphSelector>;

/*!
 * \brief This provides a set of properties for partitioning a graph into subgraphs,
 * reconstructing a new graph from the subgraphs and creating a subgraph
 * operator to execute the subgraph.
 */
class SubgraphProperty {
 public:
  // the criteria of selecting the subgraph nodes.
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const = 0;
  // create an nnvm node for a given subgraph. Here users can customize how to
  // execute the operators in the subgraph.
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &s,
                                           const int subgraph_id = 0) const = 0;
  // set an attr with name in the attr map
  template<typename T>
  SubgraphProperty& SetAttr(const std::string& name, const T& value) {
    attrs_[name] = std::make_shared<dmlc::any>(value);
    return *this;
  }
  // get the attr with the name
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

  SubgraphPropertyCreateFn __REGISTER__(const std::string& name, SubgraphPropertyCreateFn fn) {
    CHECK_EQ(prop_fn_map_.count(name), 0U) << "Subgraph property " << name
                                           << " has been registered";
    prop_fn_map_[name] = fn;
    return prop_fn_map_[name];
  }

 private:
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
    SubgraphPropertyRegistry::Get()->__REGISTER__(#Name, &SubgraphPropertyType::Create);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_PROPERTY_H_
