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
 * \file library.h
 * \brief Defining library loading functions
 */
#ifndef MXNET_COMMON_LIBRARY_H_
#define MXNET_COMMON_LIBRARY_H_

#include <iostream>
#include <map>
#include <string>
#include "dmlc/io.h"

// map of libraries loaded
static std::map<std::string, void*> loaded_libs;

void* load_lib(const char* path);
void close_lib(void* handle);
void get_sym(void* handle, void** func, char* name);

/*!
 * \brief a templated function that fetches from the library
 * a function pointer of any given datatype and name
 * \param T a template parameter for data type of function pointer
 * \param lib library handle
 * \param func_name function name to search for in the library
 * \return func a function pointer
 */
template<typename T>
T get_func(void *lib, char *func_name) {
  T func;
  get_sym(lib, reinterpret_cast<void**>(&func), func_name);
  if (!func)
    LOG(FATAL) << "Unable to get function '" << func_name << "' from library";
  return func;
}

#endif  // MXNET_COMMON_LIBRARY_H_
