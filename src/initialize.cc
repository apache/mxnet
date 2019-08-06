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
#include "./common/library.h"
#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV
#include "common/utils.h"
#include "engine/openmp.h"

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
    mp_cv_num_threads_(dmlc::GetEnv("MXNET_MP_OPENCV_NUM_THREADS", 0))
{
  dmlc::InitLogging("mxnet");
  engine::OpenMP::Get();   // force OpenMP initialization
  install_signal_handlers();
  install_pthread_atfork_handlers();
}

LibraryInitializer::~LibraryInitializer() {
  close_open_libs();
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
  this->cpu_worker_nthreads_ = this->mp_cv_num_threads_;
#if MXNET_USE_OPENCV && !__APPLE__
  cv::setNumThreads(mp_cv_num_threads_);
#endif  // MXNET_USE_OPENCV
  engine::OpenMP::Get()->set_thread_max(1);
  engine::OpenMP::Get()->set_enabled(false);
  Engine::Get()->Start();
  CustomOperator::Get()->Start();
}


void LibraryInitializer::install_pthread_atfork_handlers() {
#ifndef _WIN32
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
  for (auto const& lib : loaded_libs) {
    close_lib(lib.second);
  }
}

void LibraryInitializer::dynlib_defer_close(const std::string &path, void *handle) {
  loaded_libs.emplace(path, handle);
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
