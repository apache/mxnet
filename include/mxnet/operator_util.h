/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator_util.h
 * \brief Utility functions and registries to help quickly build new operators.
 *
 *  Use the register functions in this file when possible to simplify operator creations.
 *  Operators registered in this file will be exposed to both NDArray API and symbolic API.
 *
 * \author Tianqi Chen
 */
#ifndef MXNET_OPERATOR_UTIL_H_
#define MXNET_OPERATOR_UTIL_H_

#include <dmlc/registry.h>
#include <dmlc/parameter.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./base.h"
#include "./operator.h"

#if DMLC_USE_CXX11
#include <functional>
#endif

namespace mxnet {
namespace op {

/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define ASSIGN_DISPATCH(out, req, exp)  \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }

/*!
* \brief Maximum ndim supported for special operators like broadcasting with non contiguous lhs/rhs
*/
#define MXNET_SPECIAL_MAX_NDIM 5

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_UTIL_H_
