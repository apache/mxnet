/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator_common.cc
 * \brief implementation of common internal functions.
 */

#include "./operator_common.h"

namespace mxnet {
namespace op {

/*! \brief helper static function to read TShape. */
void String2TShape(const std::string& str, TShape* shape) {
  std::istringstream iss(str);
  iss >> *shape;
}

/*! \brief helper static function to write TShape. */
void TShape2String(const TShape& shape, std::string* str) {
  str->clear();
  std::ostringstream oss;
  oss << shape;
  *str = oss.str();
}

}  // namespace op
}  // namespace mxnet
