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
 *  Copyright (c) 2018 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef MXNET_C_API_ERROR_H_
#define MXNET_C_API_ERROR_H_

/*!
 * \brief Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END() or API_END_HANDLE_ERROR()
 * The finally clause contains procedure to cleanup states when an error happens.
 */
#define MX_API_BEGIN()                                                         \
  try {                                                                        \
    on_enter_api(__FUNCTION__);
#define MX_API_END()                                                           \
  }                                                                            \
  catch (const std::exception &_except_) {                                     \
    on_exit_api();                                                             \
    return MXAPIHandleException(_except_);                                     \
  }                                                                            \
  on_exit_api();                                                               \
  return 0; // NOLINT(*)
#define MX_API_END_HANDLE_ERROR(Finalize)                                      \
  }                                                                            \
  catch (const std::exception &_except_) {                                     \
    Finalize;                                                                  \
    on_exit_api();                                                             \
    return MXAPIHandleException(_except_);                                     \
  }                                                                            \
  on_exit_api();                                                               \
  return 0; // NOLINT(*)
/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void MXAPISetLastError(const char* msg);
/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int MXAPIHandleException(const std::exception &e) {
  MXAPISetLastError(e.what());
  return -1;
}

namespace mxnet {
extern void on_enter_api(const char *function);
extern void on_exit_api();
}
#endif  // MXNET_C_API_ERROR_H_
