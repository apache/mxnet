/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2016 by Contributors
* \file symbol.h
* \brief definition of symbol
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_LITE_SYMBOL_H_
#define MXNET_LITE_SYMBOL_H_

#include <map>
#include <string>
#include <vector>

#include <mxnet/c_api.h>
#include <nnvm/c_api.h>

#include <mxnet-lite/ndarray.h>

namespace mxnet {
namespace lite {
/*!
* \brief Symbol interface
*/
class Symbol {
 public:
  /*!
  * \brief construct a variable Symbol
  * \param name the name of the variable
  */
  static Symbol Variable(const std::string &name) {
    SymbolHandle handle;
    CHECK_EQ(MXSymbolCreateVariable(name.c_str(), &(handle)), 0);
    return Symbol(handle);
  }
  /*!
  * \brief Create a symbol that groups symbols together
  * \param symbols List of symbols to be groupe
  */
  static Symbol Group(const std::vector<Symbol> &symbols) {
    SymbolHandle out;
    std::vector<SymbolHandle> handle_list;
    for (const auto &t : symbols) {
      handle_list.push_back(t.GetHandle());
    }
    CHECK_EQ(MXSymbolCreateGroup(handle_list.size(), handle_list.data(), &out), 0);
    return Symbol(out);
  }
  /*!
  * \brief load Symbol from a JSON file
  * \param file_name the name of the file
  */
  static Symbol Load(const std::string &file_name) {
    SymbolHandle handle;
    CHECK_EQ(MXSymbolCreateFromFile(file_name.c_str(), &(handle)), 0);
    return Symbol(handle);
  }
  /*!
  * \brief load Symbol from a JSON string
  * \param json_str the JSON string
  */
  static Symbol FromJSON(const std::string &json_str) {
    SymbolHandle handle;
    CHECK_EQ(MXSymbolCreateFromJSON(json_str.c_str(), &(handle)), 0);
    return Symbol(handle);
  }
  /*!
  * \brief construct a Symbol with SymbolHandle
  * \param handle the given SymbolHandle
  */
  explicit Symbol(SymbolHandle handle) {
    blob_ptr_ = std::make_shared<SymBlob>(handle);
  }

  /*!
  * \return the SymbolHandle
  */
  SymbolHandle GetHandle() const {
    return (blob_ptr_) ? blob_ptr_->handle: NULL;
  }

  Symbol Copy() const {
    SymbolHandle handle;
    CHECK_EQ(MXSymbolCopy(GetHandle(), &handle), 0);
    return Symbol(handle);
  }
  Symbol operator[](int index) const {
    SymbolHandle out;
    MXSymbolGetOutput(GetHandle(), index, &out);
    return Symbol(out);
  }
  /*!
  * \brief save Symbol to a file
  * \param file_name the name of the file
  */
  void Save(const std::string &file_name) const {
    CHECK_EQ(MXSymbolSaveToFile(GetHandle(), file_name.c_str()), 0);
  }
  /*!
  * \brief save Symbol into a JSON string
  */
  std::string ToJSON() const {
    const char *out_json;
    CHECK_EQ(MXSymbolSaveToJSON(GetHandle(), &out_json), 0);
    return std::string(out_json);
  }
  /*!
  * \brief save Symbol into a JSON string
  * \retutrn the symbol whose outputs are all the internals.
  */
  Symbol GetInternals() const {
    SymbolHandle handle;
    CHECK_EQ(MXSymbolGetInternals(GetHandle(), &handle), 0);
    return Symbol(handle);
  }
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListOutputs() const {
    std::vector<std::string> ret;
    mx_uint size;
    const char **sarr;
    CHECK_EQ(MXSymbolListOutputs(GetHandle(), &size, &sarr), 0);
    for (mx_uint i = 0; i < size; ++i) {
      ret.push_back(std::string(sarr[i]));
    }
    return ret;
  }
  /*!
  * \brief List the arguments names.
  *
  * The position of the returned list also corresponds to calling position in
  *operator()
  * \return the arguments list of this symbol, they can be either named or
  *unnamed (empty string).
  */
  std::vector<std::string> ListArguments() const {
    std::vector<std::string> ret;
    mx_uint size;
    const char **sarr;
    CHECK_EQ(MXSymbolListArguments(GetHandle(), &size, &sarr), 0);
    for (mx_uint i = 0; i < size; ++i) {
      ret.push_back(std::string(sarr[i]));
    }
    return ret;
  }
  /*! \return get the descriptions of auxiliary data for this symbol */
  std::vector<std::string> ListAuxiliaryStates() const {
    std::vector<std::string> ret;
    mx_uint size;
    const char **sarr;
    CHECK_EQ(MXSymbolListAuxiliaryStates(GetHandle(), &size, &sarr), 0);
    for (mx_uint i = 0; i < size; ++i) {
      ret.push_back(std::string(sarr[i]));
    }
    return ret;
  }
  /*!
  * \brief List the arguments names.
  *
  * The position of the returned list also corresponds to calling position in
  *operator()
  * \return the arguments list of this symbol, they can be either named or
  *unnamed (empty string).
  */
  std::vector<std::string> ListInputs() const {
    std::vector<std::string> ret;
    mx_uint size;
    const char **sarr;
    CHECK_EQ(NNSymbolListInputNames(GetHandle(), 0, &size, &sarr), 0);
    for (mx_uint i = 0; i < size; ++i) {
      ret.push_back(std::string(sarr[i]));
    }
    return ret;
  }

 private:
  /*!
  * \brief struct to store SymbolHandle
  */
  struct SymBlob {
    /*!
    * \brief default constructor
    */
    SymBlob() : handle(nullptr) {}
    /*!
    * \brief construct with SymbolHandle to store
    */
    explicit SymBlob(SymbolHandle handle_) : handle(handle_) {}
    /*!
    * \brief destructor, free the SymbolHandle
    */
    ~SymBlob() { MXSymbolFree(handle); }
    /*!
    * \brief the SymbolHandle to store
    */
    SymbolHandle handle;
  };

  std::shared_ptr<SymBlob> blob_ptr_;
};
}  // namespace lite
}  // namespace mxnet
#endif  // MXNET_LITE_SYMBOL_H_
