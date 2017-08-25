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

namespace mxnet { namespace test {
bool unitTestsWithCuda = false;
#ifdef NDEBUG
bool debugOutput = false;
#else
bool debugOutput = false;
#endif
}}

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
      mxnet::test::debugOutput = true;
    }
  }

  std::cout << std::endl << std::flush;
  return RUN_ALL_TESTS();
}
