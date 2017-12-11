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

void segfault_logger(int sig) {
  const int MAX_STACK_SIZE = 10;
  void *stack[MAX_STACK_SIZE];

  fprintf(stderr, "\nSegmentation fault: %d\n\n", sig);

#if DMLC_LOG_STACK_TRACE
  int nframes = backtrace(stack, MAX_STACK_SIZE);
  fprintf(stderr, "Stack trace returned %d entries:\n", nframes);
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int i = 0; i < nframes; ++i) {
      fprintf(stderr, "[bt] (%d) %s\n", i, msgs[i]);
    }
  }
#endif  // DMLC_LOG_STACK_TRACE

  exit(-1);
}

class LibraryInitializer {
 public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
#if MXNET_USE_SIGNAL_HANDLER
    signal(SIGSEGV, segfault_logger);
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
