/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.h
 * \brief the static graph of symbols
 */
#ifndef MXNET_STATIC_GRAPH_H_
#define MXNET_STATIC_GRAPH_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include "./atomic_symbol.h"
namespace mxnet {
/*! \brief static graph interface
 *  static graph is an internal representation of symbol graph.
 *
 *  The main purpose for static graph for binding a composite operator
 */
struct StaticGraph {
  /*! \brief Node in static graph */
  struct StaticNode {
    /*! \brief wrapped atomic symbol */
    std::unique_ptr<AtomicSymbol> sym;
    /*! \brief name of the node */
    std::string name;
    /*! \brief index of output from the source. */
    int index;
    /*! \brief output shape for node */
    std::vector<TShape> in_shape;
    /*! \brief output shape for node */
    std::vector<TShape> out_shape;
    /*! \brief input id for each node */
    std::vector<int> inputs_index;
    /*! \brief output id for each node */
    std::vector<int> outputs_index;
  };
  /*! \brief head node (need input from outside) */
  std::vector<int> in_args_node_id;
  /*! \brief tail node (generate data to outside) */
  std::vector<int> return_node_id;
  /*! \brief node name to id dictionary */
  std::unordered_map<std::string, int> name_id_map;
  /*! \brief all nodes in the graph */
  std::vector<StaticNode> nodes;
};
}  // namespace mxnet
#endif  // MXNET_STATIC_GRAPH_H_
