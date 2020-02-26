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
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include "initialize.h"
#include <signal.h>
#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include "./engine/openmp.h"
#include "./operator/custom/custom-inl.h"
#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV
#include "common/utils.h"
#include "engine/openmp.h"


#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
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
        nullptr,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(err),
        0, nullptr);
}
#else
#include <dlfcn.h>
#endif

namespace mxnet {

#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
static void SegfaultLogger(int sig) {
  fprintf(stderr, "\nSegmentation fault: %d\n\n", sig);
  fprintf(stderr, "%s", dmlc::StackTrace().c_str());
  exit(-1);
}
#endif

// pthread_atfork handlers, delegated to LibraryInitializer members.

void pthread_atfork_prepare() {
  LibraryInitializer* library_initializer = LibraryInitializer::Get();
  library_initializer->atfork_prepare();
}

void pthread_atfork_parent() {
  LibraryInitializer* library_initializer = LibraryInitializer::Get();
  library_initializer->atfork_parent();
}

void pthread_atfork_child() {
  LibraryInitializer* library_initializer = LibraryInitializer::Get();
  library_initializer->atfork_child();
}

// LibraryInitializer member functions

LibraryInitializer::LibraryInitializer()
  : original_pid_(common::current_process_id()),
    mp_worker_nthreads_(dmlc::GetEnv("MXNET_MP_WORKER_NTHREADS", 1)),
    cpu_worker_nthreads_(dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 1)),
    mp_cv_num_threads_(dmlc::GetEnv("MXNET_MP_OPENCV_NUM_THREADS", 0)) {
  dmlc::InitLogging("mxnet");
  engine::OpenMP::Get();   // force OpenMP initialization
  install_signal_handlers();
  install_pthread_atfork_handlers();
}

LibraryInitializer::~LibraryInitializer() {
  close_open_libs();
}

bool LibraryInitializer::lib_is_loaded(const std::string& path) const {
  return loaded_libs.count(path) > 0;
}

/*!
 * \brief Loads the dynamic shared library file
 * \param path library file location
 * \return handle a pointer for the loaded library, throws dmlc::error if library can't be loaded
 */
void* LibraryInitializer::lib_load(const char* path) {
  void *handle = nullptr;
  // check if library was already loaded
  if (!lib_is_loaded(path)) {
    // if not, load it
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    handle = LoadLibrary(path);
    if (!handle) {
      char *err_msg = nullptr;
      win_err(&err_msg);
      LOG(FATAL) << "Error loading library: '" << path << "'\n" << err_msg;
      LocalFree(err_msg);
      return nullptr;
    }
#else
    handle = dlopen(path, RTLD_LAZY);
    if (!handle) {
      LOG(FATAL) << "Error loading library: '" << path << "'\n" << dlerror();
      return nullptr;
    }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
    // then store the pointer to the library
    loaded_libs[path] = handle;
  } else {
    handle = loaded_libs.at(path);
  }
  return handle;
}

/*!
 * \brief Closes the loaded dynamic shared library file
 * \param handle library file handle
 */
void LibraryInitializer::lib_close(void* handle) {
  std::string libpath;
  for (const auto& l : loaded_libs) {
    if (l.second == handle) {
      libpath = l.first;
      break;
    }
  }
  CHECK(!libpath.empty());
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  FreeLibrary((HMODULE)handle);
#else
  if (dlclose(handle)) {
    LOG(WARNING) << "LibraryInitializer::lib_close: couldn't close library at address: " << handle
        << " loaded from: '" << libpath << "': " << dlerror();
  }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
  loaded_libs.erase(libpath);
}

/*!
 * \brief Obtains address of given function in the loaded library
 * \param handle pointer for the loaded library
 * \param func function pointer that gets output address
 * \param name function name to be fetched
 */
void LibraryInitializer::get_sym(void* handle, void** func, char* name) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  *func = GetProcAddress((HMODULE)handle, name);
  if (!(*func)) {
    char *err_msg = nullptr;
    win_err(&err_msg);
    LOG(FATAL) << "Error getting function '" << name << "' from library\n" << err_msg;
    LocalFree(err_msg);
  }
#else
  *func = dlsym(handle, name);
  if (!(*func)) {
    LOG(FATAL) << "Error getting function '" << name << "' from library\n" << dlerror();
  }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
}

bool LibraryInitializer::was_forked() const {
  return common::current_process_id() != original_pid_;
}

void LibraryInitializer::atfork_prepare() {
  using op::custom::CustomOperator;
  CustomOperator::Get()->Stop();
  Engine::Get()->Stop();
}

void LibraryInitializer::atfork_parent() {
  using op::custom::CustomOperator;
  Engine::Get()->Start();
  CustomOperator::Get()->Start();
}

void LibraryInitializer::atfork_child() {
  using op::custom::CustomOperator;
  // Conservative thread management for multiprocess workers
  this->cpu_worker_nthreads_ = this->mp_worker_nthreads_;
#if MXNET_USE_OPENCV && !__APPLE__
  cv::setNumThreads(mp_cv_num_threads_);
#endif  // MXNET_USE_OPENCV
  engine::OpenMP::Get()->initialize_process();
  engine::OpenMP::Get()->set_thread_max(1);
  engine::OpenMP::Get()->set_enabled(false);
  Engine::Get()->Start();
  CustomOperator::Get()->Start();
}


void LibraryInitializer::install_pthread_atfork_handlers() {
#ifndef _WIN32
  engine::OpenMP::Get()->initialize_process();  // force omp to set its atfork handler first
  pthread_atfork(pthread_atfork_prepare, pthread_atfork_parent, pthread_atfork_child);
#endif
}

void LibraryInitializer::install_signal_handlers() {
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
  struct sigaction sa;
  sigaction(SIGSEGV, nullptr, &sa);
  if (sa.sa_handler == nullptr) {
      signal(SIGSEGV, SegfaultLogger);
  }
#endif
}

void LibraryInitializer::close_open_libs() {
  for (const auto& l : loaded_libs) {
    lib_close(l.second);
  }
}

/**
 * Perform static initialization
 */
#ifdef __GNUC__
// In GCC we use constructor to perform initialization before any static initializer is able to run
__attribute__((constructor)) static void LibraryInitializerEntry() {
#pragma GCC diagnostic ignored "-Wunused-variable"
  volatile LibraryInitializer* library_init = LibraryInitializer::Get();
}
#else
static LibraryInitializer* __library_init = LibraryInitializer::Get();
#endif

}  // namespace mxnet
