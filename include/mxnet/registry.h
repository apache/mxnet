/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.h
 * \brief registry that registers all sorts of functions
 */
#ifndef MXNET_REGISTRY_H_
#define MXNET_REGISTRY_H_

#include <dmlc/base.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include "./base.h"
#include "./narray.h"
#include "./operator.h"

namespace mxnet {

/*! \brief registry template */
template <typename Entry>
class Registry {
 public:
  /*! \return get a singleton */
  static Registry *Get();
  /*!
   * \brief register a name function under name
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  Entry &Register(const std::string& name);
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
    const std::map<std::string, Entry*> &fmap = Get()->fmap_;
    typename std::map<std::string, Entry*>::const_iterator p = fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return NULL;  //  c++11 is not required
    }
  }

 private:
  /*! \brief list of functions */
  std::vector<const Entry*> fun_list_;
  /*! \brief map of name->function */
  std::map<std::string, Entry*> fmap_;
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {
    for (typename std::map<std::string, Entry*>::iterator p = fmap_.begin();
         p != fmap_.end(); ++p) {
      delete p->second;
    }
  }
};

/*! \brief mask information on how functions can be exposed */
enum FunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};

/*! \brief registry entry */
struct NArrayFunctionEntry {
  /*! \brief definition of NArray function */
  typedef std::function<void (NArray **used_vars,
                              real_t *scalars,
                              NArray **mutate_vars)> Function;
  /*! \brief function name */
  std::string name;
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*! \brief the real function */
  Function body;
  /*!
   * \brief constructor
   * \param name name of the function
   */
  explicit NArrayFunctionEntry(const std::string &name)
      : name(name),
        num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0),
        body(nullptr) {}
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set the function body
   * \param f function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_body(Function f) {
    body = f; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
  /*!
   * \brief set the function body to a binary NArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_function(void fbinary(const NArray &lhs,
                                                        const NArray &rhs,
                                                        NArray *out)) {
    body = [fbinary] (NArray **used_vars,
                      real_t *s, NArray **mutate_vars) {
      fbinary(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    return *this;
  }
  /*!
   * \brief set the function body to a unary NArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionEntry &set_function(void funary(const NArray &src,
                                                       NArray *out)) {
    body = [funary] (NArray **used_vars,
                     real_t *s, NArray **mutate_vars) {
      funary(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
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
};  // NArrayFunctionEntry

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
      ::mxnet::Registry<NArrayFunctionEntry>::Get()->Register("" # name)


/*! \brief OperatorPropertyEntry to register */
struct OperatorPropertyEntry {
  /*! \brief typedef Creator function */
  typedef OperatorProperty*(*Creator)();
  /*! \brief if OperatorProperty use param */
  bool use_param;
  /*! \brief name of the entry */
  std::string name;
  /*! \brief description of operator */
  std::string description;
  /*! \brief pair of name description the arguments */
  std::vector<std::pair<std::string, std::string> > arguments;
  /*! \brief documentation to parameters */
  std::string param_doc;
  /*! \brief function body to create OperatorProperty */
  Creator body;
  /*! \brief constructor */
  explicit OperatorPropertyEntry(const std::string& name)
      : use_param(true), name(name), body(NULL) {}
  /*!
   * \brief set the function body
   * \param body body to set
   * \return reference to self.
   */
  inline OperatorPropertyEntry &set_body(Creator body) {
    this->body = body;
    return *this;
  }
  /*!
   * \brief describe the operator.
   * \param description the description.
   * \return reference to self.
   */
  inline OperatorPropertyEntry &describe(const std::string &description) {
    this->description = description;
    return *this;
  }
  /*!
   * \brief set the parameter documents.
   * \param description the description.
   * \return reference to self.
   */
  inline OperatorPropertyEntry &set_param_doc(const std::string &param_doc) {
    this->param_doc = param_doc;
    return *this;
  }
  /*!
   * \brief add argument to the entry.
   * \param name name of the argument.
   * \param desc description of the argument.
   * \return reference to self.
   */
  inline OperatorPropertyEntry &add_argument(const std::string &name,
                                             const std::string &desc) {
    arguments.push_back(std::make_pair(name, desc));
    return *this;
  }
  /*!
   * \brief invoke the function
   * \return the created OperatorProperty
   */
  inline OperatorProperty* operator () () const {
    return body();
  }

 private:
  /*! \brief disable copy constructor */
  OperatorPropertyEntry(const OperatorPropertyEntry& other) {}
  /*! \brief disable assignment operator */
  const OperatorPropertyEntry& operator = (const OperatorPropertyEntry& other) { return *this; }
};

/*!
 * \brief macro to register OperatorProperty to OperatorPropertyFactory
 *
 * Example: the following code is example to register aplus
 * \code
 *
 * REGISTER_ATOMIC_SYMBOL(fullc)
 * .set_use_param(false)
 *
 * \endcode
 */
#define REGISTER_OP_PROPERTY(name, OperatorPropertyType)                \
  ::mxnet::OperatorProperty* __make_ ## OperatorPropertyType ## __() {  \
    return new OperatorPropertyType;                                    \
  }                                                                     \
  static ::mxnet::OperatorPropertyEntry& __ ## name ## _atomic_symbol__ = \
      ::mxnet::Registry< ::mxnet::OperatorPropertyEntry >::Get()->Register("" # name) \
      .set_body(__make_ ## OperatorPropertyType ## __)

}  // namespace mxnet
#endif  // MXNET_REGISTRY_H_
