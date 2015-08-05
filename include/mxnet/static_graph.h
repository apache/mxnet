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
  struct NodeMetaInfo{
    /*! \brief wrapped atomic symbol */
    AtomicSymbol* sym_;
    /*! \brief name of the node */
    std::string name_;
  };

  /*!
   * \brief Node is the container of AtomicSymbol, it also stores the connection of the AtomicSymbol
   *  with input symbols.
   */
  struct Node {

    NodeMetaInfo info_;
    /*! \brief inputs to this node */
    std::vector<std::shared_ptr<Node> > in_symbol_;
    /*! \brief index of the inputs if the inputs are tuple */
    std::vector<int> in_index_;
    /*! \brief the output shape of the wrapped symbol */
    std::vector<TShape> out_shape_;
    /*!
     * \brief constructor
     */
    explicit Node(AtomicSymbol* sym = nullptr, const std::string& name = "") {
        info_.sym_ = sym;
        info_.name_ = name;
    }
    /*!
     * \brief destructor
     */
    ~Node() {
      if (info_.sym_) {
        delete info_.sym_;
      }
    }
  };

  struct StaticGraph {
    std::unordered_map<std::string, int> name_id_map;
    std::vector<NodeMetaInfo> nodes;
    std::vector<std::vector<int> > output_index;
    std::vector<std::vector<int> > connected_graph;

    int FindNodeByName(const std::string& name, const std::shared_ptr<Node> node) {
      int id = 0;
      if (name_id_map.find(name) == name_id_map.end()) {
        name_id_map[name] = name_id_map.size();
        nodes.push_back(node->info_);
        output_index.push_back(std::vector<int>());
        connected_graph.push_back(std::vector<int>());
        id = name_id_map.size();
      } else {
        id = name_id_map[name]; 
      }
      return id;
    }
  };
}
#endif
