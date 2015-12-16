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
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <functional>

#if DMLC_USE_CXX11
#include <functional>
#endif

namespace mxnet {
namespace common {
/*! \brief namespace of arguments */
namespace arg {
/*! \brief super class of all gradient function argument */
struct GradFunctionArgument {
  /*! \brief The real data */
  TBlob data;
};
/*! \brief First input to the function */
struct Input0 : GradFunctionArgument {};
/*! \brief Second input to the function */
struct Input1 : GradFunctionArgument {};

/*! \brief Ouput value of the function to the function */
struct OutValue : GradFunctionArgument {};
/*! \brief Gradient of output value */
struct OutGrad : GradFunctionArgument {};
}  // namespace arg

/*! \brief registry for function entry */
class TBlobOpRegEntry {
 public:
  typedef void (*UnaryFunction)(const TBlob &src,
                                TBlob* ret,
                                OpReqType req,
                                RunContext ctx);
  typedef TShape (*UnaryShapeInfer)(const TShape &src);
  typedef void (*UnaryGradType1)(const arg::OutGrad& out_grad,
                                 const arg::OutValue& out_value,
                                 TBlob* in_grad,
                                 OpReqType req,
                                 RunContext ctx);
  typedef void (*UnaryGradType2)(const arg::OutGrad& out_grad,
                                 const arg::Input0& in_data0,
                                 TBlob* in_grad,
                                 OpReqType req,
                                 RunContext ctx);
  /*! \brief declare self type */
  typedef TBlobOpRegEntry TSelf;
  /*! \brief name of the entry */
  std::string name;
  /*!
   * \brief set shape inference function, by default use same shape.
   * \param fshapeinfer The unary function that peforms the operation.
   */
  virtual TSelf& set_shape_infer(UnaryShapeInfer fshapeinfer) = 0;
  /*!
   * \brief set function of the function to be funary
   * \param dev_mask The device mask of the function can act on.
   * \param funary The unary function that peforms the operation.
   * \param inplace_in_out Whether do inplace optimization on in and out.
   * \param register_symbolic Whether register a symbolic operator as well.
   */
  virtual TSelf& set_function(int dev_mask,
                              UnaryFunction funary,
                              bool inplace_in_out,
                              bool register_symbolic = true) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_in_grad whether out_grad and in_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradType1 fgrad,
                              bool inplace_out_in_grad) = 0;
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradType2 fgrad,
                              bool inplace_out_in_grad) = 0;
  /*!
   * \brief Describe the function.
   * \param description The description of the function.
   * \return reference to self.
   */
  virtual TSelf& describe(const std::string &description) = 0;
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

#define MXNET_REGISTER_TBLOB_FUN(Name, DEV)                             \
  static ::mxnet::common::TBlobOpRegEntry &                             \
  __make_ ## TBlobOpRegEntry ## _ ## Name ## __ ## DEV ##__ =           \
      ::mxnet::common::TBlobOpRegistry::Get()->__REGISTER_OR_FIND__(#Name)
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_TBLOB_OP_REGISTRY_H_
