/*!
 *  Copyright (c) 2015 by Contributors
 * \file atomic_symbol_registry.h
 * \brief atomic symbol registry interface of mxnet
 */
#ifndef MXNET_ATOMIC_SYMBOL_REGISTRY_H_
#define MXNET_ATOMIC_SYMBOL_REGISTRY_H_
#include <dmlc/base.h>
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for symbol registry module"
#endif
#include <mxnet/atomic_symbol.h>
#include <unordered_map>
#include <functional>
#include <string>
#include <vector>
#include "./base.h"

namespace mxnet {

/*!
 * \brief Register AtomicSymbol
 */
class AtomicSymbolRegistry {
 public:
  /*! \return get a singleton */
  static AtomicSymbolRegistry *Get();
  /*! \brief registered entry */
  struct Entry {
    /*! \brief constructor */
    explicit Entry(const std::string& type_str) : type_str(type_str),
                                                  use_param(true),
                                                  body(nullptr) {}
    /*! \brief type string of the entry */
    std::string type_str;
    /*! \brief whether param is required */
    bool use_param;
    /*! \brief function body to create AtomicSymbol */
    std::function<AtomicSymbol*(void)> body;
    /*!
     * \brief set if param is needed by this AtomicSymbol
     */
    Entry &set_use_param(bool use_param) {
      this->use_param = use_param;
      return *this;
    }
    /*!
     * \brief set the function body
     */
    Entry &set_body(const std::function<AtomicSymbol*()>& body) {
      this->body = body;
      return *this;
    }
  };
  /*!
   * \brief register the maker function with name
   * \return the type string of the AtomicSymbol
   */
  template <typename AtomicType>
  Entry &Register() {
    AtomicType instance;
    std::string type_str = instance.TypeString();
    Entry *e = new Entry(type_str);
    fmap_[type_str] = e;
    fun_list_.push_back(e);
    e->set_body([]()->AtomicSymbol* {
        return new AtomicType;
      });
    return *e;
  }
  /*!
   * \brief find the entry by type string
   * \param type_str the type string of the AtomicSymbol
   * \return the corresponding entry
   */
  inline static const Entry* Find(const std::string& type_str) {
    auto &fmap = Get()->fmap_;
    auto p = fmap.find(type_str);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return nullptr;
    }
  }
  /*! \brief list all the AtomicSymbols */
  inline static const std::vector<const Entry*> &List() {
    return Get()->fun_list_;
  }
  /*! \brief make a atomicsymbol according to the typename */
  inline static AtomicSymbol* Make(const std::string& name) {
    return Get()->fmap_[name]->body();
  }

 protected:
  /*! \brief list of functions */
  std::vector<const Entry*> fun_list_;
  /*! \brief map of name->function */
  std::unordered_map<std::string, Entry*> fmap_;
};

/*! \brief macro to register AtomicSymbol to AtomicSymbolFactory */
#define REGISTER_ATOMIC_SYMBOL(AtomicType)                        \
  static auto __## AtomicType ## _entry__ =                       \
      AtomicSymbolRegistry::Get()->Register<AtomicType>()

}  // namespace mxnet
#endif  // MXNET_ATOMIC_SYMBOL_REGISTRY_H_
