#include "mxnet_acc.h"
#include <string>
#include <cstring>

extern "C" void getAccName(char* s) {
  std::string name = "myacc";
  strcpy(s,name.c_str());
}
