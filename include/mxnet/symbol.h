/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.h
 * \brief symbol interface of mxnet
 */
#ifndef MXNET_SYMBOL_H_
#define MXNET_SYMBOL_H_

#include <mxnet/atomic_symbol.h>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {
class NArrayOperator;
/*!
 * \brief Symbol is the wrapper of AtomicSymbol, the reason for this indirection is that Symbol
 *  should support expressions and often passed by value. While AtomicSymbol have many subclasses,
 *  passing by value would result in object slicing.
 *
 *  Symbol is always composite, the head Node is the output node of the symbol.
 *  A atomic symbol can be seen as a special case of the composite symbol with only the head node.
 */
class Symbol {
 protected:
  /*!
   * \brief Node is the container of AtomicSymbol, it also stores the connection of the AtomicSymbol
   *  with input symbols.
   */
  struct Node {
    /*! wrapped atomic symbol */
    AtomicSymbol* sym_;
    /*! name of the node */
    std::string name_;
    /*! inputs to this node */
    std::vector<std::shared_ptr<Node> > in_symbol_;
    /*! index of the inputs if the inputs are tuple */
    std::vector<int> in_index_;
    /*! the output shape of the wrapped symbol */
    std::vector<TShape> out_shape_;
    /*!
     * \brief constructor
     */
    explicit Node(AtomicSymbol* sym = nullptr, const std::string& name = "");
    /*!
     * \brief destructor
     */
    ~Node();
  };
  /*! \brief the head node of the Symbol, it could be shared in many graphs */
  std::shared_ptr<Node> head_;
  /*! \brief if the head has multiple return values, index is used to specify */
  int index_;
  /*! \brief find the nodes that use placeholder arguments */
  std::unique_ptr<std::vector<std::pair<Node*, int> > > arg_users_;
  /*! \brief find arg users */
  void FindArgUsers();

 public:
  /*!
   *  \brief bind to device and returns an NArrayOperator.
   *  \param ctx context of the operator
   *  \return returns the pointer to a created NArrayOperator. It is on the user to delete.
   */
  virtual NArrayOperator* Bind(Context ctx) const { return nullptr; }
  /*!
   * \brief copy the symbol
   * \return a deep copy of the graph
   */
  virtual Symbol Copy() const;
  /*!
   * \brief compose with arguments
   * \param args positional arguments for the symbol
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  virtual Symbol operator () (const std::vector<Symbol>& args) const;
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  virtual Symbol operator () (const std::unordered_map<std::string, Symbol>& kwargs) const;
  /*!
   * \brief get the index th element from the returned tuple.
   */
  virtual Symbol operator[] (int index) const;
  /*!
   * \brief arguments information
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  virtual std::vector<std::string> ListArgs();
};

}  // namespace mxnet
#endif  // MXNET_SYMBOL_H_
