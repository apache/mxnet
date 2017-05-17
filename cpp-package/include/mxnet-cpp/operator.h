/*!
*  Copyright (c) 2016 by Contributors
* \file operator.h
* \brief definition of operator
* \author Chuntao Hong, Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_OPERATOR_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_OPERATOR_H_

#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/op_map.h"
#include "mxnet-cpp/symbol.h"

namespace mxnet {
namespace cpp {
class Mxnet;
/*!
* \brief Operator interface
*/
class Operator {
 public:
  /*!
  * \brief Operator constructor
  * \param operator_name type of the operator
  */
  explicit Operator(const std::string &operator_name);
  Operator &operator=(const Operator &rhs);
  /*!
  * \brief set config parameters
  * \param name name of the config parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Operator &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  /*!
  * \brief set config parameters from positional inputs
  * \param pos the position of parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Operator &SetParam(int pos, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[arg_names_[pos]] = value_str;
    return *this;
  }
  /*!
  * \brief add an input symbol
  * \param name name of the input symbol
  * \param symbol the input symbol
  * \return reference of self
  */
  Operator &SetInput(const std::string &name, Symbol symbol);
  /*!
  * \brief add an input symbol
  * \param symbol the input symbol
  */
  template<int N = 0>
  void PushInput(const Symbol &symbol) {
    input_symbols.push_back(symbol.GetHandle());
  }
  /*!
  * \brief add input symbols
  * \return reference of self
  */
  Operator &operator()() { return *this; }
  /*!
  * \brief add input symbols
  * \param symbol the input symbol
  * \return reference of self
  */
  Operator &operator()(const Symbol &symbol) {
    input_symbols.push_back(symbol.GetHandle());
    return *this;
  }
  /*!
  * \brief add a list of input symbols
  * \param symbols the vector of the input symbols
  * \return reference of self
  */
  Operator &operator()(const std::vector<Symbol> &symbols) {
    for (auto &s : symbols) {
      input_symbols.push_back(s.GetHandle());
    }
    return *this;
  }
  /*!
  * \brief create a Symbol from the current operator
  * \param name the name of the operator
  * \return the operator Symbol
  */
  Symbol CreateSymbol(const std::string &name = "");

  /*!
  * \brief add an input ndarray
  * \param name name of the input ndarray
  * \param ndarray the input ndarray
  * \return reference of self
  */
  Operator &SetInput(const std::string &name, NDArray ndarray);
  /*!
  * \brief add an input ndarray
  * \param ndarray the input ndarray
  */
  template<int N = 0>
  Operator &PushInput(const NDArray &ndarray) {
    input_ndarrays.push_back(ndarray.GetHandle());
    return *this;
  }
  /*!
  * \brief add positional inputs
  */
  template <class T, class... Args, int N = 0>
  Operator &PushInput(const T &t, Args... args) {
    SetParam(N, t);
    PushInput<Args..., N+1>(args...);
    return *this;
  }
  /*!
  * \brief add the last positional input
  */
  template <class T, int N = 0>
  Operator &PushInput(const T &t) {
    SetParam(N, t);
    return *this;
  }
  /*!
  * \brief add input ndarrays
  * \param ndarray the input ndarray
  * \return reference of self
  */
  Operator &operator()(const NDArray &ndarray) {
    input_ndarrays.push_back(ndarray.GetHandle());
    return *this;
  }
  /*!
  * \brief add a list of input ndarrays
  * \param ndarrays the vector of the input ndarrays
  * \return reference of self
  */
  Operator &operator()(const std::vector<NDArray> &ndarrays) {
    for (auto &s : ndarrays) {
      input_ndarrays.push_back(s.GetHandle());
    }
    return *this;
  }
  /*!
  * \brief add input ndarrays
  * \return reference of self
  */
  template <typename... Args>
  Operator &operator()(Args... args) {
    PushInput(args...);
    return *this;
  }
  std::vector<NDArray> Invoke();
  void Invoke(NDArray &output);
  void Invoke(std::vector<NDArray> &outputs);

 private:
  std::map<std::string, std::string> params_desc_;
  bool variable_params_ = false;
  std::map<std::string, std::string> params_;
  std::vector<SymbolHandle> input_symbols;
  std::vector<NDArrayHandle> input_ndarrays;
  std::vector<std::string> input_keys;
  std::vector<std::string> arg_names_;
  AtomicSymbolCreator handle_;
  static OpMap*& op_map();
};
}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_OPERATOR_H_
