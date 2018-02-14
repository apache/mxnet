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
#include <signal.h>
#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include "engine/profiler.h"

namespace mxnet {
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
static void SegfaultLogger(int sig) {
  fprintf(stderr, "\nSegmentation fault: %d\n\n", sig);
  fprintf(stderr, "%s", dmlc::StackTrace().c_str());
  exit(-1);
}
#endif

class LibraryInitializer {
 public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
    signal(SIGSEGV, SegfaultLogger);
#endif
#if MXNET_USE_PROFILER
    // ensure profiler's constructor are called before atexit.
    engine::Profiler::Get();
    // DumpProfile will be called before engine's and profiler's destructor.
    std::atexit([](){
      engine::Profiler* profiler = engine::Profiler::Get();
      if (profiler->IsEnableOutput()) {
        profiler->DumpProfile();
      }
    });
#endif
  }

  static LibraryInitializer* Get();
};

LibraryInitializer* LibraryInitializer::Get() {
  static LibraryInitializer inst;
  return &inst;
}

#ifdef __GNUC__
// Don't print an unused variable message since this is intentional
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static LibraryInitializer* __library_init = LibraryInitializer::Get();
}  // namespace mxnet
