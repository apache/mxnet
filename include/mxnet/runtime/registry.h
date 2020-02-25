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
 * \file registry.h
 * \brief This file defines the TVM global function registry.
 *
 *  The registered functions will be made available to front-end
 *  as well as backend users.
 *
 *  The registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the TVM back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   TVM_REGISTER_GLOBAL(MyAPIFuncName)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_REGISTRY_H_
#define MXNET_RUNTIME_REGISTRY_H_

#include <string>
#include <vector>
#include "packed_func.h"

namespace mxnet {
namespace runtime {

/*! \brief Registry for global function */
class Registry {
 public:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  MXNET_DLL Registry& set_body(PackedFunc f);  // NOLINT(*)
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
  /*!
   * \brief set the body of the function to be TypedPackedFunc.
   *
   * \code
   *
   * TVM_REGISTER_API("addone")
   * .set_body_typed<int(int)>([](int x) { return x + 1; });
   *
   * \endcode
   *
   * \param f The body of the function.
   * \tparam FType the signature of the function.
   * \tparam FLambda The type of f.
   */
  template<typename FType, typename FLambda>
  Registry& set_body_typed(FLambda f) {
    return set_body(TypedPackedFunc<FType>(f).packed());
  }

  /*!
   * \brief set the body of the function to the given function pointer.
   *        Note that this doesn't work with lambdas, you need to
   *        explicitly give a type for those.
   *        Note that this will ignore default arg values and always require all arguments to be provided.
   *
   * \code
   *
   * int multiply(int x, int y) {
   *   return x * y;
   * }
   *
   * TVM_REGISTER_API("multiply")
   * .set_body_typed(multiply); // will have type int(int, int)
   *
   * \endcode
   *
   * \param f The function to forward to.
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template<typename R, typename ...Args>
  Registry& set_body_typed(R (*f)(Args...)) {
    return set_body(TypedPackedFunc<R(Args...)>(f));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_API("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template<typename T, typename R, typename ...Args>
  Registry& set_body_method(R (T::*f)(Args...)) {
    return set_body_typed<R(T, Args...)>([f](T target, Args... params) -> R {
      // call method pointer
      return (target.*f)(params...);
    });
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_API("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template<typename T, typename R, typename ...Args>
  Registry& set_body_method(R (T::*f)(Args...) const) {
    return set_body_typed<R(T, Args...)>([f](const T target, Args... params) -> R {
      // call method pointer
      return (target.*f)(params...);
    });
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_API("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template<typename TObjectRef, typename TNode, typename R, typename ...Args,
    typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...)) {
    return set_body_typed<R(TObjectRef, Args...)>([f](TObjectRef ref, Args... params) {
      TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    });
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_API("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template<typename TObjectRef, typename TNode, typename R, typename ...Args,
    typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...) const) {
    return set_body_typed<R(TObjectRef, Args...)>([f](TObjectRef ref, Args... params) {
      const TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    });
  }

  /*!
   * \brief Register a function with given name
   * \param name The name of the function.
   * \param override Whether allow oveeride existing function.
   * \return Reference to theregistry.
   */
  MXNET_DLL static Registry& Register(const std::string& name, bool override = false);  // NOLINT(*)
  /*!
   * \brief Erase global function from registry, if exist.
   * \param name The name of the function.
   * \return Whether function exist.
   */
  MXNET_DLL static bool Remove(const std::string& name);
  /*!
   * \brief Get the global function by name.
   * \param name The name of the function.
   * \return pointer to the registered function,
   *   nullptr if it does not exist.
   */
  MXNET_DLL static const PackedFunc* Get(const std::string& name);  // NOLINT(*)
  /*!
   * \brief Get the names of currently registered global function.
   * \return The names
   */
  MXNET_DLL static std::vector<std::string> ListNames();

  // Internal class.
  struct Manager;

 protected:
  /*! \brief name of the function */
  std::string name_;
  /*! \brief internal packed function */
  PackedFunc func_;
  friend struct Manager;
};

/*! \brief helper macro to supress unused warning */
#if defined(__GNUC__)
#define MXNET_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define MXNET_ATTRIBUTE_UNUSED
#endif

#define MXNET_STR_CONCAT_(__x, __y) __x##__y
#define MXNET_STR_CONCAT(__x, __y) MXNET_STR_CONCAT_(__x, __y)

#define MXNET_FUNC_REG_VAR_DEF                                            \
  static MXNET_ATTRIBUTE_UNUSED ::mxnet::runtime::Registry& __mk_ ## MXNET

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define MXNET_REGISTER_GLOBAL(OpName)                              \
  MXNET_STR_CONCAT(MXNET_FUNC_REG_VAR_DEF, __COUNTER__) =            \
      ::mxnet::runtime::Registry::Register(OpName)

}  // namespace runtime
}  // namespace mxnet
#endif  // MXNET_RUNTIME_REGISTRY_H_
