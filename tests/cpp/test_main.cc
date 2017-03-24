#include <gtest/gtest.h>

#ifdef USE_BREAKPAD
#include <breakpad/client/linux/handler/exception_handler.h>

static bool dumpCallback(const google_breakpad::MinidumpDescriptor& descriptor, void* context, bool succeeded) {
  printf("Dump path: %s\n", descriptor.path());
  return succeeded;
}
#endif

bool unitTestsWithCuda = false;

int main(int argc, char ** argv) {

#ifdef USE_BREAKPAD
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, dumpCallback, NULL, true, -1);
#endif

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  for(int x = 1; x < argc; ++x)
  {
    // force checks with CUDA
    if(!strcmp(argv[x], "--with-cuda"))
    {
      unitTestsWithCuda = true;
    }
  }

  return RUN_ALL_TESTS();
}
