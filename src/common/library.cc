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

#include "../../include/mxnet/library.h"

/*
Loads the dynamic shared library file
Parameter: Library file location
Returns: handle for the loaded library, NULL if loading unsuccessful
*/
void* load_lib(const char* path) {
  void *handle;
  handle = dlopen(path, RTLD_LAZY);

  if (!handle) {
    LOG(FATAL) << "Error loading accelerator library: '" << path
               << "'\n" << dlerror();
    return nullptr;
  }
  return handle;
}

/*
Obtains address of given function in the loaded library
Parameters
- handle: handle for the loaded library
- func: function pointer that gets output address
- name: function name to be fetched
*/
void get_sym(void* handle, void** func, char* name) {
  *func = dlsym(handle, name);
  if (!(*func)) {
    LOG(FATAL) << "Error getting function '" << name
               << "' from accelerator library\n" << dlerror();
  }
}
