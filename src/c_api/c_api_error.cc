/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include <nnvm/c_api.h>
#include "./c_api_common.h"

const char *MXGetLastError() {
  return NNGetLastError();
}

void MXAPISetLastError(const char* msg) {
  NNAPISetLastError(msg);
}
