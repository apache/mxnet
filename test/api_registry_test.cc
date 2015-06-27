// dummy code to test layer interface
// used to demonstrate how interface can be used
#include <mxnet/api_registry.h>

int main(int argc, char *argv[]) {
  auto fadd = mxnet::FunctionRegistry::Find("Plus");
  printf("f.name=%s\n", fadd->name.c_str());
  return 0;
}
