/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.h
 * \brief symbol interface of mxnet
 */
#ifndef MXNET_SYMBOL_H_
#define MXNET_SYMBOL_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "./base.h"
#include "./tensor_blob.h"

using std::shared_ptr;
using std::vector;
using std::map;

namespace mxnet {
/*!
 * \brief Symbol is the wrapper of AtomicSymbol, the reason for this indirection is that Symbol should
 *  support expressions and often passed by value. While AtomicSymbol have many subclasses, passing by
 *  value would result in object slicing.
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
  class Node {
   protected:
    /*! wrapped atomic symbol */
    AtomicSymbol* sym_;
    /*! inputs to this node */
    std::vector<std::shared_ptr<Node> > in_symbol_;
    /*! the output shape of the wrapped symbol */
    std::vector<TShape> out_shape_;
    /*!
     * \brief hide the constructor
     */
    explicit Node(AtomicSymbol* sym) : sym_(sym) {}

   public:
    /*!
     * \brief wrap the atomic symbol with a new Node and return this Node as shared_ptr
     * \param sym the atomic symbol to be wrapped
     * \return the shared_ptr to the Node that wraps the sym
     */
    static std::shared_ptr<Node> Wrap(AtomicSymbol* sym) {
      return std::make_shared<Node>(sym);
    }
    /*!
     * \brief destructor
     */
    virtual ~Node() { delete sym_; }
    /*!
     * \brief getter for the output shape of the wrapped atomic symbol
     * \return const reference to the internal out_shape_
     */
    inline const std::vector<TShape>& OutShape() const { return out_shape_; }
    /*!
     * \brief set the in_symbol_
     * \param in_symbol the input symbol to set for this Node.
     * \tparam V vector<shared_ptr<Node> > or its lvalue/rvalue references
     */
    template <typename V>
    inline void SetInSymbol(V in_symbol) {
      in_symbol_ = std::forward<V>(in_symbol);
    }
    /*!
     * \brief getter for the in_symbol_
     * \return the input symbols for this Node
     */
    inline const std::vector<std::shared_ptr<Node> >& InSymbol() { return in_symbol_; }
    /*!
     * \brief getter for the symbol wrapped in this Node
     * \return get the pointer to the atomic symbol wrappe in this Node.
     */
    inline const AtomicSymbol* Sym() const { return sym_; }
  };
  /*! \brief the head node of the Symbol, it could be shared in many graphs */
  std::shared_ptr<Node> head_;

 public:
  /*!
   *  \brief bind to device and returns an NArrayOperator.
   *  \param ctx context of the operator
   *  \return returns the pointer to a created NArrayOperator. It is on the user to delete.
   */
  virtual NArrayOperator* Bind(Context ctx) const;
  /*!
   * \brief elementwise add to current symbol
   * \param src the data to add
   * \return reference of self
   */
  Symbol &operator += (const Symbol &src);
  /*!
   * \brief elementwise subtract from current symbol
   * \param src the data to substract
   * \return reference of self
   */
  Symbol &operator -= (const Symbol &src);
  /*!
   * \brief elementwise multiplication to current symbol
   * \param src the data to multiply
   * \return reference of self
   */
  Symbol &operator *= (const Symbol &src);
  /*!
   * \brief elementwise division from current symbol
   * \param src the data to divide
   * \return reference of self
   */
  Symbol &operator /= (const Symbol &src);
  /*!
   * \brief copy the symbol
   * \return a deep copy of the graph
   */
  virtual Symbol Copy() const {
    // use Node* to avoid copying shared_ptr
    std::map<Node*, std::shared_ptr<Node> > old_new;
    std::vector<Node*> stk;
    stk.push_back(head_.get());
    // copy nodes
    while (!stk.empty()) {
      Node* top = stk.back();
      stk.pop_back();
      if (old_new.count(top) == 0) {
        old_new[top] = Node::Wrap(top->Sym()->Copy());
      }
      for (const std::shared_ptr<Node>& n : top->InSymbol()) {
        if (old_new->count(n.get()) == 0) {
          stk.push_back(n.get());
        }
      }
    }
    // connect nodes
    for (auto kv : old_new) {
      std::vector<std::shared_ptr<Node> > in_symbol;
      for (const std::shared_ptr<Node>& n : kv.first->InSymbol()) {
        in_symbol.push_back(old_new[n.get()]);
      }
      kv.first->SetInSymbol(std::move(in_symbol));
    }
    Symbol s;
    s.head_ = old_new[this->head_.get()];
    return s;
  }
  /*!
   * \brief compose with arguments
   * \param args positional arguments for the symbol
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  virtual Symbol operator() (const vector<Symbol>& args);
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  virtual Symbol operator() (const map<string, Symbol>& kwargs) {
    Symbol s = this->Copy();
  }
  /*!
   * \brief get the index th element from the returned tuple.
   */
  virtual Symbol& operator[] (int index) {
  }
};

/*!
 * \brief AtomicSymbol is the base class of all atomic symbols.
 *  This is not meant to be used by user, it should be wrapped in Symbol, so that the same instance
 *  of AtomicSymbol can be shared in the graphs of different Symbols
 */
class AtomicSymbol {
  /*! Only accessible from its wrapper Symbol */
 protected:
  /*!
   * \brief Constructor with param as the argument.
   * \param param name value pairs of the param, the constructor call SetParam to set each of them.
   */
  explicit AtomicSymbol(const std::map<std::string, std::string> &param) {
    for (std::map<std::string, std::string>::iterator it = param.begin(); it != param.end(); ++it) {
      this->SetParam(it->first.c_str(), it->second.c_str());
    }
  }
  /*! \brief get the number of inputs for this symbol */
  virtual int InCount() const { return 1; }
  /*! \brief get the number of outputs for this symbol */
  virtual int OutCount() const { return 1; }
  /*!
   *  \brief set param for the symbol from string
   *  \param name parameter name
   *  \param val string for the configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   */
  virtual void InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape) = 0;
  /*!
   * \brief Copy this AtomicSymbol and returns a shared_ptr to the copied object.
   *  this is a virtual function because different subclass of AtomicSymbol would copy differently.
   * \return a const reference of the shared_ptr to the copied object.
   *  with return value optimization may be returning const reference is not necessary.
   */
  virtual AtomicSymbol* Copy() const = 0;
  /*!
   * \brief Bind this AtomicSymbol to a context and get back a static operator
   *  Bind function of AtomicSymbol does not return NArrayOperator, but static operator.
   *  Calling bind from the Symbol wrapper would generate a NArrayOperator.
   */
  virtual Operator* Bind(Context ctx) const = 0;
  friend class Symbol;
};

}  // namespace mxnet
#endif  // MXNET_SYMBOL_H_
