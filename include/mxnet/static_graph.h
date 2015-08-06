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
    AtomicSymbol* sym_;
    /*! \brief name of the node */
    std::string name_;
  };
  /*! \brief node name to id dictionary */
  std::unordered_map<std::string, int> name_id_map;
  /*! \brief all nodes in the graph */
  std::vector<StaticNode> nodes;
  /*! \brief output id for each node */
  std::vector<std::vector<int> > output_index;
  /*! \brief connected graph for each node */
  std::vector<std::vector<int> > connected_graph;
  /*! \brief find node by using name
   *  \param name node name
   *  \param sym symbol need to be copied into node
   *  \return node id
   */
  int FindNodeByName(const std::string& name, const AtomicSymbol* sym) {
    int id = 0;
    if (name_id_map.find(name) == name_id_map.end()) {
      name_id_map[name] = name_id_map.size();
      StaticNode static_node;
      static_node.sym_ = sym->Copy();
      static_node.name_ = name;
      nodes.push_back(static_node);
      output_index.push_back(std::vector<int>());
      connected_graph.push_back(std::vector<int>());
      id = name_id_map.size();
    } else {
      id = name_id_map[name];
    }
    return id;
  }
};
}  // namespace mxnet
#endif  // MXNET_STATIC_GRAPH_H_
