#include <gtest/gtest.h>

bool unitTestsWithCuda = false;

int main(int argc, char ** argv) {
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
