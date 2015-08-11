/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.h
 * \brief The static graph of symbols
 */
#ifndef MXNET_STATIC_GRAPH_H_
#define MXNET_STATIC_GRAPH_H_

#include <vector>
#include <string>
#include <memory>
#include "./base.h"
#include "./atomic_symbol.h"

namespace mxnet {
/*!
 * \brief StaticGraph is the configuration of computation graphs.
 *  This is the "configuration file" of mxnet.
 *  It can be converted to/from Symbol, and can be used to bind to operators.
 */
class StaticGraph {
 public:
  /*! \brief represents a data in the graph */
  struct DataEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t source_id;
    /*!
     * \brief index of output from the source.
     * If index == -1, it represents all the outputs.
     */
    int32_t index;
  };
  /*! \brief Operation Node in static graph */
  struct Node {
    /*! \brief wrapped atomic symbol */
    std::unique_ptr<AtomicSymbol> sym;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs (node_id, index) for of the nodes*/
    std::vector<DataEntry> inputs;
  };
  /*! \brief all nodes in the graph */
  std::vector<Node> nodes;
  /*! \brief index is nodes that correspods to arguments */
  std::vector<uint32_t> arg_nodes;
  /*! \brief outputs(heads) of the graph */
  std::vector<DataEntry> outputs;
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
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeShapes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<TShape> > *node_out_shapes) const;
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
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape) const;
};
}  // namespace mxnet
#endif  // MXNET_STATIC_GRAPH_H_
