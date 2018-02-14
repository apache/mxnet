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
 * Copyright (c) 2017 by Contributors
 * \file test_main.cc
 * \brief operator unit test utility functions
 * \author Chris Olivier
*/
#include <gtest/gtest.h>
#include "mxnet/base.h"

#ifdef USE_BREAKPAD
#include <breakpad/client/linux/handler/exception_handler.h>

static bool dumpCallback(const google_breakpad::MinidumpDescriptor& descriptor,
                         void* context, bool succeeded) {
  printf("Dump path: %s\n", descriptor.path());
  return succeeded;
}
#endif

namespace mxnet {
namespace test {
bool unitTestsWithCuda = false;
#ifdef NDEBUG
bool debug_output = false;
#else
bool debug_output = false;
#endif
bool quick_test = false;
bool performance_run = false;
bool csv = false;
}  // namespace test
}  // namespace mxnet

#if MXNET_USE_CUDA

static bool checkForWorkingCuda() {
  int count = 0;
  if (cudaSuccess == cudaGetDeviceCount(&count)) {
    if (count == 0) return -1;
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp prop;
      if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
        std::printf("%d.%d ", prop.major, prop.minor);
        return true;
      }
    }
  }
  std::fprintf(stderr, "Warning: Could not find working CUDA driver\n");
  return false;
}
#else
static bool checkForWorkingCuda() {
  return false;
}
#endif

void backtrace_test() {
  CHECK(false) << "backtrace()";
}

int main(int argc, char ** argv) {
#ifdef USE_BREAKPAD
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, dumpCallback, NULL, true, -1);
#endif

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  mxnet::test::unitTestsWithCuda = checkForWorkingCuda();  // auto-determine

  for (int x = 1; x < argc; ++x) {
    // force checks with CUDA
    if (!strcmp(argv[x], "--with-cuda")) {
      // override (ie force attempt CUDA)
      mxnet::test::unitTestsWithCuda = true;
    } else if (!strcmp(argv[x], "--debug")) {
      mxnet::test::debug_output = true;
    } else if (!strcmp(argv[x], "--perf")) {
      mxnet::test::performance_run = true;
    } else if (!strcmp(argv[x], "--csv")) {
      mxnet::test::csv = true;
    } else if (!strcmp(argv[x], "--quick") || !strcmp(argv[x], "-q")) {
      mxnet::test::quick_test = true;
    } else if (!strcmp(argv[x], "--backtrace")) {
        backtrace_test();
        return 0;
    }
  }

  std::cout << std::endl << std::flush;
  return RUN_ALL_TESTS();
}
