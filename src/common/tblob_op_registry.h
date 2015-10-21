/*!
 *  Copyright (c) 2015 by Contributors
 * \file tblob_op_registry.h
 * \brief Helper registry to make registration of simple unary binary math function easy.
 * Register to this registry will enable both symbolic operator and NDArray operator in client.
 *
 * More complicated operators can be registered in normal way in ndarray and operator modules.
 */
#ifndef MXNET_COMMON_TBLOB_OP_REGISTRY_H_
#define MXNET_COMMON_TBLOB_OP_REGISTRY_H_

#include <dmlc/registry.h>
#include <mxnet/base.h>
#include <map>
#include <string>
#include <vector>

namespace mxnet {
namespace common {

/*! \brief pre-declare generic TBlob function*/
struct GenericTBlobOp;

/*! \brief registry for function entry */
class TBlobOpRegEntry {
 public:
  /*! \brief unary tblob function */
  typedef void (*UnaryFunction)(const TBlob &src,
                                TBlob *ret,
                                RunContext ctx);
  /*! \brief declare self type */
  typedef TBlobOpRegEntry TSelf;
  /*! \brief name of the entry */
  std::string name;
  /*!
   * \brief set function of the function to be funary
   * \param dev_mask The device mask of the function can act on.
   * \param funary The unary function that peforms the operation.
   */
  virtual TSelf& set_function(int dev_mask, UnaryFunction funary) = 0;
  /*!
   * \brief Describe the function.
   * \param description The description of the function.
   * \return reference to self.
   */
  virtual TSelf& describe(const std::string &description) = 0;
  /*!
   * \brief get the internal function representation
   * \return the internal function representation.
   */
  virtual GenericTBlobOp *GetOp() const = 0;
  /*! \brief destructor */
  virtual ~TBlobOpRegEntry() {}
};

/*! \brief registry for TBlob functions */
class TBlobOpRegistry {
 public:
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  TBlobOpRegEntry &__REGISTER_OR_FIND__(const std::string& name);
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const TBlobOpRegEntry *Find(const std::string &name) {
    return Get()->fmap_.at(name);
  }
  /*! \return global singleton of the registry */
  static TBlobOpRegistry* Get();

 private:
  // destructor
  ~TBlobOpRegistry();
  /*! \brief internal registry map */
  std::map<std::string, TBlobOpRegEntry*> fmap_;
};

#if DMLC_USE_CXX11
struct GenericTBlobOp {
  /*! \brief function type of the function */
  typedef std::function<void (const std::vector<TBlob> &in,
                              TBlob *out,
                              RunContext ctx)> OpType;
  /*! \brief the real operator */
  OpType op;
};
#endif

#define MXNET_REGISTER_TBLOB_FUN(Name, DEV)                             \
  static ::mxnet::common::TBlobOpRegEntry &                             \
  __make_ ## TBlobOpRegEntry ## _ ## Name ## __ ## DEV ##__ =           \
      ::mxnet::common::TBlobOpRegistry::Get()->__REGISTER_OR_FIND__(#Name)

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_TBLOB_OP_REGISTRY_H_
