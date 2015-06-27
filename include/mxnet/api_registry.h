/*!
 *  Copyright (c) 2015 by Contributors
 * \file api_registry.h
 * \brief api registry that registers functions
 *   for C API module and possiblity other modules
 */
#ifndef MXNET_API_REGISTRY_H_
#define MXNET_API_REGISTRY_H_
#include <dmlc/base.h>
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for api registry module"
#endif
#include <map>
#include <functional>
#include "./base.h"
#include "./narray.h"

namespace mxnet {
/*! \brief registry of NArray functions */
class FunctionRegistry {
 public:
  /*! \brief definition of NArray function */
  typedef std::function<void (NArray **used_vars,
                              real_t *scalars,
                              NArray **mutate_vars)> Function;
  /*! \brief registry entry */
  struct Entry {
    /*! \brief function name */
    std::string name;
    /*! \brief number of variable used by this function */
    unsigned num_use_vars;
    /*! \brief number of variable mutated by this function */
    unsigned num_mutate_vars;
    /*! \brief number of scalars used by this function */
    unsigned num_scalars;
    /*! \brief the real function */
    Function body;    
    /*!
     * \brief constructor 
     * \param name name of the function
     */
    Entry(const std::string &name)
        : name(name),
          num_use_vars(0),
          num_mutate_vars(0),
          num_scalars(0),
          body(nullptr) {}
    /*!
     * \brief set the number of mutate variables
     * \param n number of mutate variablesx
     * \return ref to the registered entry, used to set properties
     */    
    inline Entry &set_num_use_vars(unsigned n) {
      num_use_vars = n; return *this;
    }
    /*!
     * \brief set the number of mutate variables
     * \param n number of mutate variablesx
     * \return ref to the registered entry, used to set properties
     */
    inline Entry &set_num_mutate_vars(unsigned n) {
      num_mutate_vars = n; return *this;
    }
    /*!
     * \brief set the number of scalar arguments
     * \param n number of scalar arguments
     * \return ref to the registered entry, used to set properties
     */
    inline Entry &set_num_scalars(unsigned n) {
      num_scalars = n; return *this;
    }
    /*!
     * \brief set the function body
     * \param f function body to set
     * \return ref to the registered entry, used to set properties
     */
    inline Entry &set_body(Function f) {
      body = f; return *this;
    }
    /*!
     * \brief set the function body to a binary NArray function
     *  this will also auto set the parameters correctly
     * \param fbinary function body to set
     * \return ref to the registered entry, used to set properties
     */
    inline Entry &set_function(void fbinary(const NArray &lhs,
                                            const NArray &rhs,
                                            NArray *out)) {
      body = [fbinary] (NArray **used_vars, real_t *s, NArray **mutate_vars) {
        fbinary(*used_vars[0], *used_vars[1], mutate_vars[0]);
      };
      num_use_vars = 2; num_mutate_vars = 1;
      return *this;
    }                                           
    /*!
     * \brief invoke the function
     * \param use_vars variables used by the function
     * \param scalars the scalar arguments passed to function
     * \param mutate_vars the variables mutated by the function
     */
    inline void operator()(NArray **use_vars,
                           real_t *scalars,
                           NArray **mutate_vars) const {
      body(use_vars, scalars, mutate_vars);
    }
  };  // Entry    
  /*! \return get a singleton */
  static FunctionRegistry *Get();
  /*!
   * \brief register a name function under name
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  Entry &Register(const std::string name);
  /*! \return list of functions in the registry */
  inline static const std::vector<const Entry*> &List() {
    return Get()->fun_list_;
  }
  /*!
   * \brief find an function entry with corresponding name 
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const Entry *Find(const std::string &name) {
    auto &fmap = Get()->fmap_;
    auto p = fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return nullptr;
    }
  }
  
 private:
  /*! \brief list of functions */
  std::vector<const Entry*> fun_list_;
  /*! \brief map of name->function */
  std::map<std::string, Entry*> fmap_;
  /*! \brief constructor */
  FunctionRegistry() {}
  /*! \brief destructor */
  ~FunctionRegistry();
};

/*!
 * \brief macro to register NArray function
 *
 * Example: the following code is example to register aplus
 * \code
 *
 * REGISTER_NARRAY_FUN(Plus)
 * .set_body([] (NArray **used_vars, real_t *scalars, NArray **mutate_vars) {
 *    BinaryPlus(*used_vars[0], *used_vars[1], mutate_vars[0]);
 *  })
 * .set_num_use_vars(2)
 * .set_num_mutate_vars(1);
 *
 * \endcode
 */
#define REGISTER_NARRAY_FUN(name)                                \
  static auto __ ## name ## _narray_fun__ =                      \
      ::mxnet::FunctionRegistry::Get()->Register("" # name)

}  // namespace mxnet
#endif  // MXNET_API_REGISTRY_H_
