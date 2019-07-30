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
 * Copyright (c) 2015 by Contributors
 * \file library.cc
 * \brief Dynamically loading accelerator library
 * and accessing its functions
 */

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "library.h"


#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
/*!
 * \brief Retrieve the system error message for the last-error code
 * \param err string that gets the error message
 */
void win_err(char **err) {
  uint32_t dw = GetLastError();
  FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(err),
        0, NULL);
}
#endif


/*!
 * \brief Loads the dynamic shared library file
 * \param path library file location
 * \return handle a pointer for the loaded library, nullptr if loading unsuccessful
 */
void* load_lib(const char* path) {
  void *handle;
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  handle = LoadLibrary(path);
  if (!handle) {
    char *lpMsgBuf;
    win_err(&lpMsgBuf);
    LOG(FATAL) << "Error loading library: '" << path << "'\n" << lpMsgBuf;
    LocalFree(lpMsgBuf);
    return nullptr;
  }
#else
  handle = dlopen(path, RTLD_LAZY);
  if (!handle) {
    LOG(FATAL) << "Error loading library: '" << path << "'\n" << dlerror();
    return nullptr;
  }
#endif  // _WIN32 or _WIN64 or __WINDOWS__

  return handle;
}

/*!
 * \brief Obtains address of given function in the loaded library
 * \param handle pointer for the loaded library
 * \param func function pointer that gets output address
 * \param name function name to be fetched
 */
void get_sym(void* handle, void** func, char* name) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  *func = GetProcAddress((HMODULE)handle, name);
  if (!(*func)) {
    char *lpMsgBuf;
    win_err(&lpMsgBuf);
    LOG(FATAL) << "Error getting function '" << name << "' from library\n" << lpMsgBuf;
    LocalFree(lpMsgBuf);
  }
#else
  *func = dlsym(handle, name);
  if (!(*func)) {
    LOG(FATAL) << "Error getting function '" << name << "' from library\n" << dlerror();
  }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
}
