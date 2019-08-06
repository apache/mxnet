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
 * \file libtest.cc
 * \brief This test checks if the library is implemented correctly
 * and does not involve dynamic loading of library into MXNet
 * This test is supposed to be run before test.py
 */

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <iostream>
#include "lib_api.h"

#define MXNET_VERSION 10500

int main(void) {
  // Get a handle to the library.
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  HINSTANCE handle;
  handle = LoadLibrary(TEXT("mylib.dll"));
#else
  void *handle;
  handle = dlopen("mylib.so", RTLD_LAZY);
#endif

  if (!handle) {
    std::cerr << "Unable to load library" << std::endl;
    return 1;
  }

  // get initialize function address from the library
  initialize_t init_lib;
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  init_lib = (initialize_t) GetProcAddress(handle, MXLIB_INITIALIZE_STR);
#else
  init_lib = (initialize_t) dlsym(handle, MXLIB_INITIALIZE_STR);
#endif

  if (!init_lib) {
    std::cerr << "Unable to get function 'intialize' from library" << std::endl;
    return 1;
  }

  // Call the function.
  (init_lib)(MXNET_VERSION);

  // Deallocate memory.
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif

  return 0;
}
