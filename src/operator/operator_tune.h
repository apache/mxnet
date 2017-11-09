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
#ifndef MXNET_OPERATOR_OPERATOR_TUNE_H_
#define MXNET_OPERATOR_OPERATOR_TUNE_H_

#include <mshadow/base.h>

namespace mxnet {
namespace op {
/*!
 * \brief Declare a template specialization for the Kernel::Launch call for the given OP
 *        wrapped with mxnet_op::op_with_req, using the given OpReqType as the 'req'
 *        template parameter for 'op_with_req'.  This is useful for the standard mshadow_op
 *        operators which need to be wrapped with op_with_req in order to be used with the
 *        Kernel::Launch command.
 *
 * \note Expects to be used within the mxnet::op namespace
 *
 * For example:
 *
 * namespace mxnet_op {
 * template <>
 * template <typename... Args>
 * inline void Kernel<typename mxnet_op::op_with_req<mshadow::op::identity, kNullOp>, cpu>
 *   ::Launch(mshadow::Stream<cpu>* s, const int N, Args... args) {
 *   ::mxnet::op::mxnet_op::Kernel<typename mxnet_op::op_with_req<mshadow::op::identity, kNullOp>,
 *     cpu>::LaunchMShadowOpEx(s, N, args...);
 *   }
 * }
 *
 */
#define MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, __req$) \
  namespace mxnet_op { \
  template<> template<typename ...Args> \
  inline void Kernel<typename mxnet_op::op_with_req<__op$, __req$>, ::mshadow::cpu>:: \
    Launch(mshadow::Stream<::mshadow::cpu> *s, const int N, Args... args) { \
      /* Launch via LaunchMShadowOpEx() */ \
      Kernel<typename mxnet_op::op_with_req<__op$, __req$>, ::mshadow::cpu>:: \
        LaunchMShadowOpEx(s, N, args...); \
  } \
  }  /* namespace mxnet_op */

/*!
 * \brief Declare template specializations for the Kernel::Launch call for the given OP
 *        wrapped with mxnet_op::op_with_req, using the all supported OpReqType as the 'req'
 *        template parameter for 'op_with_req'.  This is useful for the standard mshadow_op
 *        operators which need to be wrapped with op_with_req in order to be used with the
 *        Kernel::Launch command.
 * \note Expects to be used within the mxnet::op namespace
 */
#define MXNET_TUNABLE_MSHADOW_OP(__op$) \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kNullOp); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kWriteTo); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kWriteInplace); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kAddTo);

#define MXNET_TUNABLE_MSHADOW_OP_BACKWARD(__op$) \
  MXNET_TUNABLE_MSHADOW_OP(mxnet::op::mxnet_op::backward_grad<__op$>)

#define MXNET_TUNABLE_MSHADOW_OP_FWD_AND_BWD(__op$) \
  MXNET_TUNABLE_MSHADOW_OP(__op$) \
  MXNET_TUNABLE_MSHADOW_OP_BACKWARD(__op$)

/*!
 * \brief mxnet::op::mxnet_op format ops (work directly with Kernel<>::Launch()
 *        Used from within mxnet::op::mxnet_op namespace
 */
#define _MXNET_TUNABLE_MXNET_OP_FWD(__op$) \
  template<> template<typename ...Args> inline void Kernel<__op$, ::mshadow::cpu>::Launch( \
    mshadow::Stream<::mshadow::cpu> *s, const int N, Args... args) { \
      /* Launch via LaunchMXNetOpEx() */ \
      Kernel<__op$, ::mshadow::cpu>::LaunchMXNetOpEx(s, N, args...); \
  }

/*!
 * \brief mxnet::op::mxnet_op format ops (work directly with Kernel<>::Launch()
 *        Used from within mxnet::op
 */
#define MXNET_TUNABLE_MXNET_OP_FWD(__op$) \
  namespace mxnet_op { _MXNET_TUNABLE_MXNET_OP_FWD(__op$) }  /* namespace mxnet_op */

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_OPERATOR_TUNE_H_
