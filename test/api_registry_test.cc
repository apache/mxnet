// Copyright (c) 2015 by Contributors
// dummy code to test layer interface
// used to demonstrate how interface can be used
#include <mxnet/registry.h>

int main(int argc, char *argv[]) {
  auto fadd = mxnet::Registry<mxnet::NArrayFunctionEntry>::Find("Plus");
  printf("f.name=%s\n", fadd->name.c_str());
  return 0;
}
