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

#include <dlfcn.h>

#include <iostream>


int main(void) {
  dlerror();
  void *mx;
  mx = dlopen("libmxnet.so", RTLD_LAZY | RTLD_GLOBAL);
  
  if (!mx) {
    std::cerr << "Unable to load libmxnet.so" << std::endl;
    char* err = dlerror();
    if(err)
      std::cerr << err << std::endl;
    return 1;
  }

  // Get a handle to the library.
  void *handle;
  handle = dlopen("libmin_ex.so", RTLD_LAZY);

  if (!handle) {
    std::cerr << "Unable to load library" << std::endl;
    char* err = dlerror();
    if(err)
      std::cerr << err << std::endl;
    return 1;
  }

  // get initialize function address from the library
  void* init_lib = dlsym(handle, "initialize");

  if (!init_lib) {
    std::cerr << "Unable to get function 'intialize' from library" << std::endl;
    return 1;
  }

  dlclose(handle);

  return 0;
}
