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
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include <vector>

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
    /*! \brief default constructor */
    Node() : backward_source_id(-1) {}

    friend void swap(Node& lhs, Node& rhs) {
      std::swap(lhs.op, rhs.op);
      std::swap(lhs.name, rhs.name);
      std::swap(lhs.inputs, rhs.inputs);
      std::swap(lhs.backward_source_id, rhs.backward_source_id);
    }
    /*! \brief copy constructor in favor of serialization. */
    Node(const Node& another) : op(another.op.get() ? another.op.get()->Copy() : nullptr),
                                name(another.name),
                                inputs(another.inputs),
                                backward_source_id(another.backward_source_id) {}

    inline Node& operator=(Node another) {
      swap(*this, another);
      return *this;
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
   * \brief infer the node shapes in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_shapes The shapes of the each outputs of nodes in the graph.
   * \param node_aux_shapes The shapes of the each auxiliary states of nodes in the graph.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeShapes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<TShape> > *node_out_shapes,
                       std::vector<std::vector<TShape> > *node_aux_shapes) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param aux_shape the shape of auxiliary states of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferShape(std::vector<TShape>* in_shape,
                  std::vector<TShape>* out_shape,
                  std::vector<TShape>* aux_shape) const;
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
   */
  void MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                        std::vector<DataEntry> *arg_grads);
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
  static Node CreateSumNode(const std::vector<DataEntry> &grad_source);
};
}  // namespace mxnet

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, ::mxnet::StaticGraph::DataEntry, true);
}
#endif  //  MXNET_SYMBOL_STATIC_GRAPH_H_
