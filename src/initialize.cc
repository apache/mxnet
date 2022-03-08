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
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include "initialize.h"

#include <algorithm>
#include <csignal>

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
/*!
 * \brief Retrieve the system error message for the last-error code
 * \param err string that gets the error message
 */
void win_err(char** err) {
  uint32_t dw = GetLastError();
  FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr,
      dw,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<char*>(err),
      0,
      nullptr);
}
#else
#include <cxxabi.h>
#include <dlfcn.h>
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
#include <execinfo.h>
#endif
#include <cerrno>
#endif

#include <dmlc/logging.h>
#include <mxnet/c_api.h>
#include <mxnet/engine.h>

#include "./engine/openmp.h"
#include "./operator/custom/custom-inl.h"
#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV
#include "common/utils.h"
#include "engine/openmp.h"

#if defined(MKL_USE_SINGLE_DYNAMIC_LIBRARY)
#include <mkl.h>
#endif

namespace mxnet {

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
  init_mkl_dynamic_library();
  engine::OpenMP::Get();  // force OpenMP initialization
  install_pthread_atfork_handlers();
}

LibraryInitializer::~LibraryInitializer() = default;

bool LibraryInitializer::lib_is_loaded(const std::string& path) const {
  return loaded_libs_.count(path) > 0;
}

/*!
 * \brief Loads the dynamic shared library file
 * \param path library file location
 * \return handle a pointer for the loaded library, throws dmlc::error if library can't be loaded
 */
void* LibraryInitializer::lib_load(const char* path) {
  void* handle = nullptr;
  // check if library was already loaded
  if (!lib_is_loaded(path)) {
    // if not, load it
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    handle = LoadLibrary(path);
    if (!handle) {
      char* err_msg = nullptr;
      win_err(&err_msg);
      LOG(FATAL) << "Error loading library: '" << path << "'\n" << err_msg;
      LocalFree(err_msg);
      return nullptr;
    }
#else
    /* library loading flags:
     *  RTLD_LAZY - Perform lazy binding. Only resolve symbols as the code that
     *              references them is executed.
     *  RTLD_LOCAL - Symbols defined in this library are not made available to
     *              resolve references in subsequently loaded libraries.
     */
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
      LOG(FATAL) << "Error loading library: '" << path << "'\n" << dlerror();
      return nullptr;
    }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
    // then store the pointer to the library
    loaded_libs_[path] = handle;
  } else {
    handle = loaded_libs_.at(path);
  }
  return handle;
}

/*!
 * \brief Closes the loaded dynamic shared library file
 * \param handle library file handle
 */
void LibraryInitializer::lib_close(void* handle, const std::string& libpath) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  FreeLibrary((HMODULE)handle);
#else
  if (dlclose(handle)) {
    LOG(WARNING) << "LibraryInitializer::lib_close: couldn't close library at address: " << handle
                 << " loaded from: '" << libpath << "': " << dlerror();
  }
#endif  // _WIN32 or _WIN64 or __WINDOWS__
}

/*!
 * \brief Obtains address of given function in the loaded library
 * \param handle pointer for the loaded library
 * \param func function pointer that gets output address
 * \param name function name to be fetched
 */
void LibraryInitializer::get_sym(void* handle, void** func, const char* name) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  *func = GetProcAddress((HMODULE)handle, name);
  if (!(*func)) {
    char* err_msg = nullptr;
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

void LibraryInitializer::init_mkl_dynamic_library() {
#if !(defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__))
#if MKL_USE_SINGLE_DYNAMIC_LIBRARY
#if USE_INT64_TENSOR_SIZE
  int interface = MKL_INTERFACE_ILP64;
#else
  int interface = MKL_INTERFACE_LP64;
#endif
#if defined(__INTEL_LLVM_COMPILER) || defined(__APPLE__)
  mkl_set_threading_layer(MKL_THREADING_INTEL);
#else
  mkl_set_threading_layer(MKL_THREADING_GNU);
  interface += MKL_INTERFACE_GNU;
#endif
  mkl_set_interface_layer(interface);
#endif
#endif
}

#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE

static inline void printStackTrace(FILE* out = stderr, const unsigned int max_frames = 63) {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__WINDOWS__)
  // storage array for stack trace address data
  void* addrlist[max_frames + 1];

  // retrieve current stack addresses
  size_t addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

  if (addrlen < 5) {
    return;
  } else {
    addrlen = std::min(addrlen, dmlc::LogStackTraceLevel());
  }
  fprintf(out, "Stack trace:\n");

  // resolve addresses into strings containing "filename(function+address)",
  // Actually it will be ## program address function + offset
  // this array must be free()-ed
  char** symbollist = backtrace_symbols(addrlist, addrlen);

  size_t funcnamesize = 1024;
  char funcname[1024];

  // iterate over the returned symbol lines. skip the first, it is the
  // address of this function.
  for (unsigned int i = 4; i < addrlen; i++) {
    char* begin_name   = nullptr;
    char* begin_offset = nullptr;
    char* end_offset   = nullptr;

    // find parentheses and +address offset surrounding the mangled name
#ifdef DARWIN
    // OSX style stack trace
    for (char* p = symbollist[i]; *p; ++p) {
      if (*p == '_' && *(p - 1) == ' ') {
        begin_name = p - 1;
      } else if (*p == '+') {
        begin_offset = p - 1;
      }
    }

    if (begin_name && begin_offset && begin_name < begin_offset) {
      *begin_name++   = '\0';
      *begin_offset++ = '\0';

      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():
      int status;
      char* ret = abi::__cxa_demangle(begin_name, &funcname[0], &funcnamesize, &status);
      if (status == 0) {
        funcname = ret;  // use possibly realloc()-ed string
        fprintf(out, "  %-30s %-40s %s\n", symbollist[i], funcname, begin_offset);
      } else {
        // demangling failed. Output function name as a C function with
        // no arguments.
        fprintf(out, "  %-30s %-38s() %s\n", symbollist[i], begin_name, begin_offset);
      }
    } else {
      // couldn't parse the line? print the whole line.
      fprintf(out, "  %-40s\n", symbollist[i]);
    }
#else
    for (char* p = symbollist[i]; *p; ++p) {
      if (*p == '(') {
        begin_name = p;
      } else if (*p == '+') {
        begin_offset = p;
      } else if (*p == ')' && (begin_offset || begin_name)) {
        end_offset = p;
      }
    }

    if (begin_name && end_offset && begin_name < end_offset) {
      *begin_name++ = '\0';
      *end_offset++ = '\0';
      if (begin_offset) {
        *begin_offset++ = '\0';
      }

      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():

      int status  = 0;
      char* ret   = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
      char* fname = begin_name;
      if (status == 0) {
        fname = ret;
      }

      if (begin_offset) {
        fprintf(
            out, "  %-30s ( %-40s  + %-6s) %s\n", symbollist[i], fname, begin_offset, end_offset);
      } else {
        fprintf(out, "  %-30s ( %-40s    %-6s) %s\n", symbollist[i], fname, "", end_offset);
      }
    } else {
      // couldn't parse the line? print the whole line.
      fprintf(out, "  %-40s\n", symbollist[i]);
    }
#endif  // !DARWIN - but is posix
  }
  free(symbollist);
#endif
}

#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, IS_FATAL)                          \
  std::shared_ptr<void(int)> HANDLER_NAME(                                      \
      signal(SIGNAL,                                                            \
             [](int signum) {                                                   \
               if (IS_FATAL) {                                                  \
                 printf("\nFatal Error: %s\n", strsignal(SIGNAL));              \
                 printStackTrace();                                             \
                 signal(signum, SIG_DFL);                                       \
                 raise(signum);                                                 \
               } else {                                                         \
                 switch (signum) {                                              \
                   case SIGSEGV:                                                \
                     LOG(FATAL) << "InternalError: " << strsignal(SIGNAL);      \
                     break;                                                     \
                   case SIGFPE:                                                 \
                     LOG(FATAL) << "FloatingPointError: " << strsignal(SIGNAL); \
                     break;                                                     \
                   case SIGBUS:                                                 \
                     LOG(FATAL) << "IOError: " << strsignal(SIGNAL);            \
                     break;                                                     \
                   default:                                                     \
                     LOG(FATAL) << "RuntimeError: " << strsignal(SIGNAL);       \
                     break;                                                     \
                 }                                                              \
               }                                                                \
             }),                                                                \
      [](auto f) { signal(SIGNAL, f); });

SIGNAL_HANDLER(SIGSEGV, SIGSEGVHandler, true);
SIGNAL_HANDLER(SIGFPE, SIGFPEHandler, false);
SIGNAL_HANDLER(SIGBUS, SIGBUSHandler, false);

#endif

void LibraryInitializer::close_open_libs() {
  for (const auto& l : loaded_libs_) {
    lib_close(l.second, l.first);
  }
  loaded_libs_.clear();
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
