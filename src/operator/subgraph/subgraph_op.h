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

#ifndef MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_OP_H_

#include <nnvm/graph.h>
#include <nnvm/pass.h>

namespace mxnet {

namespace op {

namespace sg {  // sg stands for subgraph

struct SimpleNode;
using SimpleNodePtr = std::shared_ptr<SimpleNode>;

struct SimpleNode {
  static SimpleNodePtr Create() {
    return std::make_shared<SimpleNode>();
  }
  SimpleNode() : label(-1), node(nullptr) {}
  int label;
  nnvm::Node* node;
  // key is node ptr
  // value is the index array standing for the entry indices
  // in key->inputs that use this->node as input node
  std::unordered_map<nnvm::Node*, std::vector<int>> outputs;
};

}

/*
 * This provides criteria for selecting nodes in a subgraph.
 * When a node is passed to this object, the selection criteria may be changed.
 * We can also specify what links we should use when traversing the neighbor
 * nodes.
 */
class SubgraphSelector {
 public:
  virtual ~SubgraphSelector() {
  }
  /*
   * Given a set of nodes that have been selected so far for a subgraph, determine
   * if the input node should be selected for a subgraph.
   */
  virtual bool Select(const nnvm::Node &n) = 0;
  virtual bool UseIncomingEdges() const = 0;
  virtual bool UseOutgoingEdges() const = 0;
};

using SubgraphSelectorPtr = std::shared_ptr<SubgraphSelector>;

/*
 * This is the interface of the subgraph operator that executes the computation
 * in the subgraph.
 */
class SubgraphOperator {
public:
  SubgraphOperator(const nnvm::Symbol &sym) {
    this->subgraph_sym_ = sym;
  }

  virtual ~SubgraphOperator() {
  }

  const nnvm::Symbol &GetSubgraph() const {
    return subgraph_sym_;
  }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) = 0;
  virtual void Backward(const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) = 0;
private:
  nnvm::Symbol subgraph_sym_;
};

using SubgraphOperatorPtr = std::shared_ptr<SubgraphOperator>;

/*
 * This provides a set of properties for partitioning a graph into subgraphs,
 * reconstructing a new graph from the subgraphs and creating a subgraph
 * operator to execute the subgraph.
 */
class SubgraphProperty {
 public:
  // the criteria of selecting the subgraph nodes.
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const = 0;
  // create an nnvm node for a given subgraph. Here users can customize how to
  // execute the operators in the subgraph.
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &s) const = 0;
  // Create a subgraph operator for execution.
  virtual SubgraphOperatorPtr CreateSubgraphOperator(const nnvm::Symbol &sym) const = 0;
  // The type of the subgraph.
  virtual std::string GetType() const = 0;
};

using SubgraphPropertyPtr = std::shared_ptr<SubgraphProperty>;

void RegisterSubgraphProperty(SubgraphPropertyPtr property);

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class ContainOpSelector: public SubgraphSelector {
  std::shared_ptr<const std::unordered_set<std::string>> op_names;

 public:
  ContainOpSelector(std::shared_ptr<const std::unordered_set<std::string>> op_names) {
    this->op_names = op_names;
  }

  virtual bool UseIncomingEdges() const {
    return true;
  }

  virtual bool UseOutgoingEdges() const {
    return true;
  }

  virtual bool Select(const nnvm::Node &n) {
    return !n.is_variable() && op_names->count(n.op()->name);
  }
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _subgraph_op.
 */
class SimpleSubgraphProperty: public SubgraphProperty {
 public:
  SimpleSubgraphProperty(const std::unordered_set<std::string> &op_names) {
    this->op_names = std::make_shared<std::unordered_set<std::string>>(op_names);
  }
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_subgraph_op");
    n->attrs.name = "_subgraph_op";
    n->attrs.dict.insert(std::pair<std::string, std::string>("exec_type", GetType()));
    n->attrs.parsed = sym;
    return n;
  }
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<ContainOpSelector>(op_names);
  }

  virtual SubgraphOperatorPtr CreateSubgraphOperator(const nnvm::Symbol &sym) const;
  virtual std::string GetType() const {
    return "default";
  }

 private:
  std::shared_ptr<const std::unordered_set<std::string>> op_names;
};

}
}

#endif  // MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_OP_H_ 
