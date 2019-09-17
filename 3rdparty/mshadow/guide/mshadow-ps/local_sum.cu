#include "./local_sum-inl.h"
int main(int argc, char *argv[]) {
  return Run<mshadow::gpu>(argc, argv);
}
