#include <iostream>
#include "mxnet/lib_api.h"

using namespace mxnet::ext;

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported";
    return MX_FAIL;
  }
}
