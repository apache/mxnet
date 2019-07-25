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
 * \brief Tests if the library is implemented correctly
 */

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdio.h>
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

  // If the handle is valid, try to get the function address.
  if (!handle) {
    printf("Unable to load library\n");
    return 1;
  }

  initialize_t func;
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  func = (initialize_t) GetProcAddress(handle, INITIALIZE_STR);
#else
  func = (initialize_t) dlsym(handle, INITIALIZE_STR);
#endif

  // If the function address is valid, call the function.
  if (!func) {
    printf("Unable to get function 'intialize' from library\n");
    return 1;
  }

  // Call the function.
  (func)(MXNET_VERSION);

  // Deallocate memory.
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif

  return 0;
}
