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
 * \file engine.cc
 * \brief Implementation of engine.
 */
#include <mxnet/engine.h>
#include <memory>
#include <cstdlib>
#include "./engine_impl.h"
#include <fenv.h>
#include <signal.h>

static void 
float_num_err_handler (int sig, siginfo_t *siginfo, void *context) {
  fprintf(stderr, ">>> Floating point error occur\n");
  switch(siginfo->si_code) {
  case FPE_INTDIV:
    fprintf(stderr, ">>> Integer division by zero\n");
    break;
  case FPE_FLTOVF:
    fprintf(stderr, ">>> Floating overflow trap\n");
    break;
  case FPE_FLTDIV:
    fprintf(stderr, ">>> Floating/decimal division by zero\n");
    break;
  }
  assert(0);
}

namespace mxnet {
namespace engine {
inline Engine* CreateEngine() {
  const char *type = getenv("MXNET_ENGINE_TYPE");
  const bool default_engine = (type == nullptr);
  if (type == nullptr) type = "ThreadedEnginePerDevice";
  std::string stype = type;

  struct sigaction act;
  memset(&act, '\0', sizeof(act));
  act.sa_sigaction = &float_num_err_handler;
  act.sa_flags = SA_SIGINFO;
  if (sigaction(SIGFPE, &act, NULL) < 0) {
    fprintf(stderr, "Cannot register signal handler\n");
    exit(-1);
  }
  feenableexcept(FE_ALL_EXCEPT);

  Engine *ret = nullptr;
  #if MXNET_PREDICT_ONLY == 0
  if (stype == "NaiveEngine") {
    ret = CreateNaiveEngine();
  } else if (stype == "ThreadedEngine") {
    ret = CreateThreadedEnginePooled();
  } else if (stype == "ThreadedEnginePerDevice") {
    ret = CreateThreadedEnginePerDevice();
  }
  #else
  ret = CreateNaiveEngine();
  #endif

  int e = std::fetestexcept(FE_ALL_EXCEPT);
  if (e & FE_DIVBYZERO) {
    LOG(FATAL) << "divide by zero" << type;
  }
  if (ret ==nullptr) {
    LOG(FATAL) << "Cannot find Engine " << type;
  }
  if (!default_engine) {
    LOG(INFO) << "MXNet start using engine: " << type;
  }
  return ret;
}
}  // namespace engine

std::shared_ptr<Engine> Engine::_GetSharedRef() {
  static std::shared_ptr<Engine> sptr(engine::CreateEngine());
  return sptr;
}

Engine* Engine::Get() {
  static Engine *inst = _GetSharedRef().get();
  return inst;
}
}  // namespace mxnet
