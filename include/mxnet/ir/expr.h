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
 * \file expr.h
 * \brief Base expr nodes in MXNet.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_IR_EXPR_H_
#define MXNET_IR_EXPR_H_

#include <mxnet/runtime/object.h>
#include <mxnet/node/node.h>
#include <mxnet/node/container.h>
#include <mxnet/runtime/data_type.h>
#include <string>

namespace mxnet {

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public Object {
 public:
  static constexpr const char* _type_key = "Expr";
  MXNET_DECLARE_BASE_OBJECT_INFO(BaseExprNode, Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ObjectRef {
 public:
  /*! \brief Cosntructor */
  BaseExpr() {}
  /*!
   * \brief Cosntructor from object ptr.
   * \param ptr The object pointer.
   */
  explicit BaseExpr(runtime::ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief The container type. */
  using ContainerType = BaseExprNode;
};

/*!
 * \brief Base node of all primitive expressions.
 *
 *  A primitive expression deals with low-level
 *  POD data types and handles without
 *  doing life-cycle management for objects.
 *
 *  PrimExpr is used in the low-level code
 *  optimizations and integer analysis.
 *
 * \sa PrimExpr
 */
class PrimExprNode : public BaseExprNode {
 public:
  /*!
   * \brief The runtime data type of the primitive expression.
   *
   * MXNetDataType(dtype) provides coarse grained type information
   * during compile time and runtime. It is eagerly built in
   * PrimExpr expression construction and can be used for
   * quick type checking.
   *
   * dtype is sufficient to decide the Type of the PrimExpr
   * when it corresponds to POD value types such as i32.
   *
   * When dtype is MXNetDataType::Handle(), the expression could corresponds to
   * a more fine-grained Type, and we can get the type by running lazy type inference.
   */
  MXNetDataType dtype;

  static constexpr const char* _type_key = "PrimExpr";
  MXNET_DECLARE_BASE_OBJECT_INFO(PrimExprNode, BaseExprNode);
};

/*!
 * \brief Reference to PrimExprNode.
 * \sa PrimExprNode
 */
class PrimExpr : public BaseExpr {
 public:
    /*! \brief Cosntructor */
  PrimExpr() {}
  /*!
   * \brief Cosntructor from object ptr.
   * \param ptr The object pointer.
   */
  explicit PrimExpr(runtime::ObjectPtr<Object> ptr) : BaseExpr(ptr) {}
  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  MXNET_DLL PrimExpr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  MXNET_DLL PrimExpr(float value);  // NOLINT(*)
  /*!
   * \brief construct from string.
   * \param str The value to be constructed.
   */
  MXNET_DLL PrimExpr(std::string str);  // NOLINT(*)

  /*! \return the data type of this expression. */
  MXNetDataType dtype() const {
    return static_cast<const PrimExprNode*>(get())->dtype;
  }
  /*! \brief The container type. */
  using ContainerType = PrimExprNode;
};

/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmNode : public PrimExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  static constexpr const char* _type_key = "IntImm";
  MXNET_DECLARE_FINAL_OBJECT_INFO(IntImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to IntImmNode.
 *
 * \sa IntImmNode
 */
class IntImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor
   */
  IntImm() {}
  /*!
   * \brief constructor from node.
   */
  explicit IntImm(runtime::ObjectPtr<Object> node) : PrimExpr(node) {}
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  MXNET_DLL IntImm(MXNetDataType dtype, int64_t value);
  /*!
   * \brief Get pointer to the internal value.
   * \return the content of the integer.
   */
  const IntImmNode* operator->() const {
    return static_cast<const IntImmNode*>(get());
  }
  /*! \brief type indicate the container type */
  using ContainerType = IntImmNode;
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  static constexpr const char* _type_key = "FloatImm";
  MXNET_DECLARE_FINAL_OBJECT_INFO(FloatImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to FloatImmNode.
 *
 * \sa FloatImmNode
 */
class FloatImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor
   */
  FloatImm() {}
  /*!
   * \brief constructor from node.
   */
  explicit FloatImm(runtime::ObjectPtr<Object> node) : PrimExpr(node) {}
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  MXNET_DLL FloatImm(MXNetDataType dtype, double value);
  /*!
   * \brief Get pointer to the container.
   * \return The pointer.
   */
  const FloatImmNode* operator->() const {
    return static_cast<const FloatImmNode*>(get());
  }
  /*! \brief type indicate the container type */
  using ContainerType = FloatImmNode;
};

}  // namespace mxnet
#endif  // MXNET_IR_EXPR_H_
