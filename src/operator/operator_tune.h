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


/*!
 * \brief Init variable used to facilitate registering a tunable operator during
 *        static initialization
 * \tparam OP Operator type
 * \tparam DType Data type
 */
template<typename OP, typename DType>
struct static_init_var {
  static bool init_;
};

/*!
 * \brief Repeat the given macro and associated arguments for each data type,
 *        appending the data type to the end of the arguments
 */
#define MSHADOW_MACRO_FOREACH_TYPE(__macro$, ...) \
  __macro$(__VA_ARGS__, float); \
  __macro$(__VA_ARGS__, double); \
  __macro$(__VA_ARGS__, mshadow::half::half_t); \
  __macro$(__VA_ARGS__, uint8_t); \
  __macro$(__VA_ARGS__, int8_t); \
  __macro$(__VA_ARGS__, int32_t); \
  __macro$(__VA_ARGS__, int64_t);


#define IMPLEMENT_BASIC_WORKLOAD(__op$, __v1$, __typ$) \
  namespace mxnet_op { \
  template<> size_t mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::workload_ = (__v1$) == 0 ? \
     (INT_MAX / 4) : (__v1$); }  /* namespace mxnet_op */

/*!
 * \brief Implement tuning objects for a forward blank (no arguments) kernel operator
 */
#define _IMPLEMENT_BLANK_WORKLOAD_FWD(__op$, __v1$, __typ$) \
  IMPLEMENT_BASIC_WORKLOAD(__op$, __v1$, __typ$); \
  namespace mxnet_op { \
  template<> int mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      mxnet::op::UnaryOpTune<__typ$>::TuneBlankOperatorEx<__op$>)

/*!
 * \brief Implement tuning objects for a forward unary kernel operator
 */
#define _IMPLEMENT_UNARY_WORKLOAD_FWD(__op$, __v1$, __typ$) \
  IMPLEMENT_BASIC_WORKLOAD(__op$, __v1$, __typ$); \
  namespace mxnet_op { \
  template<> int mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      mxnet::op::UnaryOpTune<__typ$>::TuneUnaryOperator<__op$>)

/*!
 * \brief Implement tuning objects for a backward unary kernel operator
 */
#define _IMPLEMENT_UNARY_WORKLOAD_BWD(__op$, __v1$, __typ$) \
  IMPLEMENT_BASIC_WORKLOAD(mxnet::op::mxnet_op::backward_grad<__op$>, __v1$, __typ$); \
  namespace mxnet_op { \
  template<> \
  int mxnet::op::mxnet_op::tuned_op<mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op< \
      mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>>(N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>::init_ = \
    mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      mxnet::op::UnaryOpTune<__typ$>::TuneUnaryBackwardOperator<__op$>)

/*!
 * \brief Implement tuning objects for a forward binary kernel operator
 */
#define _IMPLEMENT_BINARY_WORKLOAD_FWD(__op$, __v1$, __typ$) \
  IMPLEMENT_BASIC_WORKLOAD(__op$, __v1$, __typ$); \
  namespace mxnet_op { \
  template<> int mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return mxnet::op::BinaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      mxnet::op::BinaryOpTune<__typ$>::TuneBinaryOperator<__op$>)

/*!
 * \brief Implement tuning objects for a backward binary kernel operator
 */
#define _IMPLEMENT_BINARY_WORKLOAD_BWD(__op$, __v1$, __typ$) \
  IMPLEMENT_BASIC_WORKLOAD(mxnet::op::mxnet_op::backward_grad<__op$>, __v1$, __typ$); \
  namespace mxnet_op { \
  template<> \
    int mxnet::op::mxnet_op::tuned_op<mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return mxnet::op::BinaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op< \
      mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>>(N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<mxnet::op::mxnet_op::backward_grad<__op$>, __typ$>::init_ = \
    mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>(  \
      mxnet::op::BinaryOpTune<__typ$>::TuneBinaryBackwardOperator<__op$>)

/*!
 * \brief Macros for manually adding new blank, unary and binary operators to the tuning set
 */
#define IMPLEMENT_UNARY_WORKLOAD_FWD(__op$, __v1$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_UNARY_WORKLOAD_FWD, __op$, __v1$)

#define IMPLEMENT_BLANK_WORKLOAD_FWD(__op$, __v1$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BLANK_WORKLOAD_FWD, __op$, __v1$)

#define IMPLEMENT_UNARY_WORKLOAD_BWD(__op$, __v1$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_UNARY_WORKLOAD_BWD, __op$, __v1$)

#define IMPLEMENT_BINARY_WORKLOAD_FWD(__op$, __v1$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BINARY_WORKLOAD_FWD, __op$, __v1$)

#define IMPLEMENT_BINARY_WORKLOAD_BWD(__op$, __v1$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BINARY_WORKLOAD_BWD, __op$, __v1$)

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_OPERATOR_TUNE_H_
