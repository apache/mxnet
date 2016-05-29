/*!
 * Copyright (c) 2015 by Contributors
 * \file static_graph.h
 * \brief A memory compact representation of symbolic graph
 *   Used for serialization, and helper data structure.
 * \author Naiyan Wang
 */
#ifndef MXNET_SYMBOL_STATIC_GRAPH_H_
#define MXNET_SYMBOL_STATIC_GRAPH_H_

#include <dmlc/base.h>
#include <dmlc/json.h>
#include <dmlc/type_traits.h>
#include <dmlc/parameter.h>
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>

namespace mxnet {
/*!
 * \brief StaticGraph is the configuration of computation graphs.
 *  This is the "configuration file" of mxnet.
 *  It can be converted to/from Symbol, and can be used to bind to operators.
 *   The symbol can be converted from/to StaticGraph, the actual configuration used by mxnet.
 *   Symbol offers more flexible way to composite nodes than StaticGraph, which makes it good
 *   tool to generate configurations from language bindings such as python.
 * \sa Symbol
 */
class StaticGraph {
 public:
  /*! \brief represents a data in the graph */
  struct DataEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t source_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief default constructor */
    DataEntry() {}
    /*!
     * \brief constructor with source and index
     * \param source_id source id
     * \param index node index
     */
    DataEntry(uint32_t source_id, uint32_t index)
        : source_id(source_id), index(index) {}
    /*!
     * \brief compare equality
     * \param other the other entry to compare
     * \return whether two entries equals to each other
     */
    inline bool operator==(const DataEntry &other) const {
      return source_id == other.source_id && index == other.index;
    }
    /*!
     * \brief comparator, allows to use map
     * \param other the other entry to compare
     * \return whether two entries is smaller than the other
     */
    inline bool operator<(const DataEntry &other) const {
      if (source_id == other.source_id) return index < other.index;
      return source_id < other.source_id;
    }
    /*!
     * \brief interface for json serialization.
     * \param writer the JSON writer to write json into.
     */
    inline void Save(dmlc::JSONWriter *writer) const {
      writer->BeginArray(false);
      writer->WriteArrayItem(source_id);
      writer->WriteArrayItem(index);
      writer->EndArray();
    }
    /*!
     * \brief interface for json serialization.
     * \param reader the JSON reader to read json from.
     */
    inline void Load(dmlc::JSONReader *reader) {
      std::pair<uint32_t, uint32_t> p;
      reader->Read(&p);
      *this = DataEntry(p.first, p.second);
    }
  };
  /*!
   * \brief Operation Node in static graphs.
   *  There are two types of node, Forward and Backward Node.
   *
   *  - Forward node corresponds to the op.Forward
   *  - Backward node corresponds to the Backward pass,
   *    where the corresponding forward node is indicated by backward_source_id.
   *    The op field in Backward node is nullptr
   *
   *  The reason we explicit support Backward node is to allow special treatment
   *  such as shape inference and state sharing with Forward pass.
   */
  struct Node {
    /*! \brief wrapped operator property */
    std::unique_ptr<OperatorProperty> op;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs (node_id, index) for of the nodes*/
    std::vector<DataEntry> inputs;
    /*!
     * \brief If this field is nonnegative, this indicates this
     *  Node is corresponds to a Backward Operation of Operator.
     *  backward_source_id will points to the corresponding Forward Node.
     *
     *  For normal node, this field is -1.
     *  When the node is a Backward node, the op field will be nullptr
     */
    int32_t backward_source_id;
    /*! \brief additional attributes about the node */
    std::map<std::string, std::string> attr;
    /*!
     * \brief Data structure to enable add-to operations in the node.
     *  Use to enable memory efficient gradient sum aggregation.
     *  Normally this array is empty.
     *
     *  Let n = inputs.size() - addto_index_.size();
     *    the output of the node is defined as:
     *  - out[j] = op(input[0:n]) for j not in addto_index_
     *  - out[addto_index_[i]] = op(input[0:n]) + inputs[n + i]
     */
    std::vector<uint32_t> addto_index;
    /*! \brief default constructor */
    Node() : backward_source_id(-1) {}
    /*! \brief copy constructor in favor of serialization. */
    Node(const Node& another)
        : op(another.op.get() ? another.op.get()->Copy() : nullptr),
          name(another.name),
          inputs(another.inputs),
          backward_source_id(another.backward_source_id),
          attr(another.attr),
          addto_index(another.addto_index) {}

    inline Node& operator=(Node another) {
      op = std::move(another.op);
      name = std::move(another.name);
      inputs = std::move(another.inputs);
      backward_source_id = std::move(another.backward_source_id);
      attr = std::move(another.attr);
      addto_index = std::move(another.addto_index);
      return *this;
    }

    template<typename ValueType>
    inline ValueType get_attr(const std::string& key, ValueType default_value) const {
      auto it = attr.find(key);
      if (it == attr.end()) {
        return default_value;
      } else {
        ValueType ret;
        dmlc::parameter::FieldEntry<ValueType> e;
        e.Init(key, &ret, ret);
        e.Set(&ret, it->second);
        return ret;
      }
    }

    /*! \return whether the node is forward op node */
    inline bool is_forward() const {
      return op != nullptr;
    }
    /*! \return whether the node is backward op node */
    inline bool is_backward() const {
      return backward_source_id != -1;
    }
    /*! \return whether the node is variable node */
    inline bool is_variable() const {
      return op == nullptr && !is_backward();
    }
    /*!
     * \brief interface for json serialization.
     * \param writer the JSON writer write json.
     */
    void Save(dmlc::JSONWriter *writer) const;
    /*!
     * \brief interface for json serialization.
     * \param reader the JSON read to read json.
     */
    void Load(dmlc::JSONReader *reader);
  };
  /*! \brief all nodes in the graph */
  std::vector<Node> nodes;
  /*! \brief index of nodes that correspods to arguments */
  std::vector<uint32_t> arg_nodes;
  /*! \brief heads outputs of the graph */
  std::vector<DataEntry> heads;
  /*!
   * \brief interface for json serialization.
   * \param writer the JSON writer write json.
   */
  void Save(dmlc::JSONWriter *writer) const;
  /*!
   * \brief interface for json serialization.
   * \param reader the JSON read to read json.
   */
  void Load(dmlc::JSONReader *reader);
  // funtions to help inference in static graph
  /*!
   * \brief Perform a topological sort on the graph
   * \return a topological order of node indices.
   */
  std::vector<uint32_t> TopoSort() const;
  /*!
   * \brief Get a post DFS order traversal order from the head nodes.
   *  Post DFS order is a special case of Topological order.
   * \param heads The head of the node.
   * \param banned The banned map, used to ban some nodes from the graph.
   * \return a post DFS visit order of nodes that can reach heads.
   */
  std::vector<uint32_t> PostDFSOrder(const std::vector<uint32_t>& head_nodes) const;
  /*!
   * \brief infer the node shapes in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_shapes The shapes of the each outputs of nodes in the graph.
   * \param node_aux_shapes The shapes of the each auxiliary states of nodes in the graph.
   * \param partial_infer Whether return partially inferred results.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeShapes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<TShape> > *node_out_shapes,
                       std::vector<std::vector<TShape> > *node_aux_shapes,
                       bool partial_infer = false) const;
  /*!
   * \brief infer the node types in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_types The types of the each outputs of nodes in the graph.
   * \param node_aux_types The types of the each auxiliary states of nodes in the graph.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeTypes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<int> > *node_out_types,
                       std::vector<std::vector<int> > *node_aux_types) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be inferred
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param aux_shape the shape of auxiliary states of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param partial_infer Whether return partially inferred results.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferShape(std::vector<TShape>* in_shape,
                  std::vector<TShape>* out_shape,
                  std::vector<TShape>* aux_shape,
                  bool partial_infer = false) const;

  /*!
   * \brief infer the types of outputs and unknown input arguments
   * \param in_type the type of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_type allows unknown elements, which are checked by type.ndim() == 0.
   *     For unknown types, Infertype will try to fill in the correct type in in_type
   *     For known types, Infertype will check type consistency
   *
   *     common practice: set the type of data input, and usually weight's type can be inferred
   *
   * \param out_type the type of outputs of the operator
   *     Infertype will modify the vector to fill output int
   * \param aux_type the type of auxiliary states of the operator
   *     Infertype will modify the vector to fill output int
   * \return if the type inference is successful, return true, else return false.
   */
  bool InferType(std::vector<int>* in_type,
                  std::vector<int>* out_type,
                  std::vector<int>* aux_type) const;
  /*!
   * \brief Add a full backward pass in the static graph.
   *  This function will add gradient nodes for each heads,
   *  and add the backward pass to backprop the gradients all
   *  the way to the arguments.
   *
   *  This will change the nodes field in the StaticGraph, but will not change other fields.
   *  The head and input of Backward pass will be returned by head_grad_nodes and arg_grads.
   *
   * \param head_grad_nodes used to store the created head gradient inputs for backward pass.
   * \param arg_grads used to store gradients to args, can be multiple one if an argument is used by operator
   * \param out_mirror_map The mirror map of the backward plan.
   */
  void MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                        std::vector<DataEntry> *arg_grads,
                        std::map<uint32_t, uint32_t>* out_mirror_map);
  /*!
   * \brief Convert symbol into static graph.
   * \param symbol the symbol to convert from.
   */
  inline void FromSymbol(const Symbol &symbol) {
    symbol.ToStaticGraph(this);
  }
  /*!
   * \brief create a sum node that aggregates gradient together
   * \param grad_source the source of the inputs.
   * \return a created ElementWiseSum node
   */
  Node CreateGradSumNode(const std::vector<DataEntry> &grad_source);
  /*!
   * \brief create a copy node.
   * \param source the Source data
   * \return a created _CrossDeviceCopy node
   */
  static Node CreateCopyNode(const DataEntry& source);
};
}  // namespace mxnet

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, ::mxnet::StaticGraph::DataEntry, true);
}
#endif  //  MXNET_SYMBOL_STATIC_GRAPH_H_
