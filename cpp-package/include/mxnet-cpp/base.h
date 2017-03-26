/*!
*  Copyright (c) 2016 by Contributors
* \file base.h
* \brief base definitions for mxnetcpp
* \author Chuntao Hong, Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_BASE_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_BASE_H_

#include <cstdlib>
#include "mxnet/c_api.h"
#include "nnvm/c_api.h"

namespace mxnet {
namespace cpp {

typedef unsigned index_t;

enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
  * \brief perform an inplace write,
  * Target shares memory with one of input arguments.
  * This option only happen when
  */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_BASE_H_
