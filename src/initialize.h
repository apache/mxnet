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
 * \file initialize.h
 * \brief Library initialization
 */

#include <cstdlib>
#include <map>
#include <string>

#include "dmlc/io.h"

#ifndef MXNET_INITIALIZE_H_
#define MXNET_INITIALIZE_H_

namespace mxnet {

void pthread_atfork_prepare();
void pthread_atfork_parent();
void pthread_atfork_child();

/**
 * Perform library initialization and control multiprocessing behaviour.
 */
class LibraryInitializer {
 public:
  typedef std::map<std::string, void*> loaded_libs_t;
  static LibraryInitializer* Get() {
    static LibraryInitializer inst;
    return &inst;
  }

  /**
   * Library initialization. Called on library loading via constructor attributes or
   * C++ static initialization.
   */
  LibraryInitializer();

  ~LibraryInitializer();

  /**
   * @return true if the current pid doesn't match the one that initialized the library
   */
  bool was_forked() const;

  // Library loading
  bool lib_is_loaded(const std::string& path) const;
  void* lib_load(const char* path);
  void lib_close(void* handle, const std::string& libpath);
  static void get_sym(void* handle, void** func, const char* name);

  /**
   * Original pid of the process which first loaded and initialized the library
   */
  size_t original_pid_;
  size_t mp_worker_nthreads_;
  size_t cpu_worker_nthreads_;
  size_t omp_num_threads_;
  size_t mp_cv_num_threads_;

  // Actual code for the atfork handlers as member functions.
  void atfork_prepare();
  void atfork_parent();
  void atfork_child();

 private:
  /**
   * Pthread atfork handlers are used to reset the concurrency state of modules like CustomOperator
   * and Engine when forking. When forking only the thread that forks is kept alive and memory is
   * copied to the new process so state is inconsistent. This call install the handlers.
   * Has no effect on Windows.
   *
   * https://pubs.opengroup.org/onlinepubs/009695399/functions/pthread_atfork.html
   */
  void install_pthread_atfork_handlers();

  /**
   * Sets the interface and threading layer for Intel® oneAPI MKL at run time.
   * Use with the Single Dynamic Library.
   */
  void init_mkl_dynamic_library();
  /**
   * Install signal handlers (UNIX). Has no effect on Windows.
   */
  void install_signal_handlers();

  void close_open_libs();

  loaded_libs_t loaded_libs_;
};

/*!
 * \brief fetches from the library a function pointer of any given datatype and name
 * \param T a template parameter for data type of function pointer
 * \param lib library handle
 * \param func_name function name to search for in the library
 * \return func a function pointer
 */
template <typename T>
T get_func(void* lib, const char* func_name) {
  T func;
  LibraryInitializer::Get()->get_sym(lib, reinterpret_cast<void**>(&func), func_name);
  if (!func)
    LOG(FATAL) << "Unable to get function '" << func_name << "' from library";
  return func;
}

}  // namespace mxnet
#endif  // MXNET_INITIALIZE_H_
