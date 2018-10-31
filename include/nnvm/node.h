/*!
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/node.h
 * \brief Graph node data structure.
 */
#ifndef NNVM_NODE_H_
#define NNVM_NODE_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "base.h"
#include "op.h"
#include "c_api.h"

namespace nnvm {

// Forward declare node.
class Node;
class Symbol;

/*!
 * \brief we always used NodePtr for a reference pointer
 *  to the node, so this alias can be changed in case.
 *
 *  By default, NodePtr is a std::shared_ptr of node
 */
using NodePtr = std::shared_ptr<Node>;

/*! \brief an entry that represents output data from a node */
struct NodeEntry {
  /*! \brief the source node of this data */
  NodePtr node;
  /*! \brief index of output from the source. */
  uint32_t index;
  /*!
   * \brief version of input Variable.
   *  This field can only be nonzero when this->node is a Variable node.
   *  version is increased by one each time a Variable get composed to a mutation Op.
   *  This information can be helpful to decide order of operations when sequence of mutation happens.
   */
  uint32_t version;
};

/*!
 * \brief This lets you use a NodeEntry as a key in a unordered_map of the form
 * unordered_map<NodeEntry, ValueType, NodeEntryHash, NodeEntryEqual>
 */
struct NodeEntryHash {
  size_t operator()(const NodeEntry& e) const {
    return std::hash<Node*>()(e.node.get()) ^
          (std::hash<size_t>()(e.index) << 1 >> 1) ^
          (std::hash<size_t>()(e.version) << 1);
  }
};

/*!
 * \brief This lets you use a NodeEntry as a key in a unordered_map of the form
 * unordered_map<NodeEntry, ValueType, NodeEntryHash, NodeEntryEqual>
 */
struct NodeEntryEqual {
  size_t operator()(const NodeEntry& a, const NodeEntry& b) const {
    return (a.node.get() == b.node.get()) &&
           (a.index == b.index) &&
           (a.version == b.version);
  }
};

/*! use NodeEntry as key in unordered_map */
template<typename ValueType>
using NodeEntryMap = std::unordered_map<NodeEntry, ValueType, NodeEntryHash, NodeEntryEqual>;

/*!
 * \brief The attributes of the current operation node.
 *  Usually are additional parameters like axis,
 */
struct NodeAttrs {
  /*!
   * \brief The operator this node uses.
   *  For place holder variable, op == nullptr.
   */
  const Op *op{nullptr};
  /*! \brief name of the node */
  std::string name;
  /*! \brief The dictionary representation of attributes */
  std::unordered_map<std::string, std::string> dict;
  /*!
   * \brief A parsed version of attributes,
   * This is generated if OpProperty.attr_parser is registered.
   * The object can be used to quickly access attributes.
   */
  any parsed;
  /*!
   * \brief Some operators take graphs as input. These operators include
   * control flow operators and high-order functions.
   * These graphs don't change when the operators are invoked for different
   * mini-batches. In this sense, the subgraphs are kind of similar to
   * the parameters and show be kept as node attributes.
   *
   * Users need to make sure the subgraphs are disjoint with the main graph.
   * If a graph shares nodes with subgraphs, loading the graph from LoadJSON
   * may generate a graph that has a different structure from the original graph
   * (some of the nodes are duplicated). If nodes are shared between two graphs,
   * shared nodes might be executed multiple times, which can be a problem for
   * stateful operators.
   */
  std::vector<std::shared_ptr<Symbol> > subgraphs;
};

/*!
 * \brief Node represents an operation in a computation graph.
 */
class NNVM_DLL Node {
 public:
  /*! \brief The attributes in the node. */
  NodeAttrs attrs;
  /*! \brief inputs to this node */
  std::vector<NodeEntry> inputs;
  /*!
   * \brief Optional control flow dependencies
   *  Gives operation must be performed before this operation.
   */
  std::vector<NodePtr> control_deps;
  /*! \brief additional fields for this node */
  any info;
  /*! \brief destructor of node */
  ~Node();
  /*! \return operator in this node */
  inline const Op* op() const;
  /*!
   * \brief return whether node is placeholder variable.
   *  This is equivalent to op == nullptr
   * \return whether node is placeholder input variable
   */
  inline bool is_variable() const;
  /*! \return number of outputs from this node */
  inline uint32_t num_outputs() const;
  /*! \return number of inputs from this node */
  inline uint32_t num_inputs() const;
  /*!
   * \brief create a new empty shared_ptr of Node.
   * \return a created empty node.
   */
  static NodePtr Create();
};

/*!
 * \brief Quick utilities make node.
 * \param op_name The name of operator
 * \param node_name The name of the node
 * \param inputs The input entries
 * \param attrs The attributes
 * \return The created node entry.
 */
inline NodeEntry MakeNode(
    const char* op_name,
    std::string node_name,
    std::vector<NodeEntry> inputs,
    std::unordered_map<std::string, std::string> attrs =
    std::unordered_map<std::string, std::string>()) {
  NodePtr p = Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = std::move(node_name);
  p->attrs.dict = attrs;
  if (p->attrs.op->attr_parser) {
    p->attrs.op->attr_parser(&(p->attrs));
  }
  p->inputs = std::move(inputs);
  return NodeEntry{p, 0, 0};
}

// implementation of functions.
inline const Op* Node::op() const {
  return this->attrs.op;
}
inline bool Node::is_variable() const {
  return this->op() == nullptr;
}

inline uint32_t Node::num_outputs() const {
  if (is_variable()) return 1;
  if (this->op()->get_num_outputs == nullptr) {
    return this->op()->num_outputs;
  } else {
    return this->op()->get_num_outputs(this->attrs);
  }
}

inline uint32_t Node::num_inputs() const {
  if (is_variable()) return 1;
  if (this->op()->get_num_inputs == nullptr) {
    return this->op()->num_inputs;
  } else {
    return this->op()->get_num_inputs(this->attrs);
  }
}

}  // namespace nnvm

#endif  // NNVM_NODE_H_
