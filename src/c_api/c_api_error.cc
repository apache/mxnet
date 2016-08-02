/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include "./c_api_common.h"

struct ErrorEntry {
  std::string last_error;
};

typedef dmlc::ThreadLocalStore<ErrorEntry> MXAPIErrorStore;

const char *MXGetLastError() {
  return MXAPIErrorStore::Get()->last_error.c_str();
}

void MXAPISetLastError(const char* msg) {
  MXAPIErrorStore::Get()->last_error = msg;
}
