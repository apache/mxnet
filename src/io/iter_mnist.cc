/*!
 * Copyright (c) 2015 by Contributors
 * \file iter_mnist.cc
 * \brief register mnist iterator
 * \author Tianjun Xiao
*/
#include <mxnet/registry.h> 
#include "./iter_mnist-inl.h"

namespace mxnet {
namespace io {

DMLC_REGISTER_PARAMETER(MNISTParam);
REGISTER_IO_ITER(mnist, MNISTIterator);

}  // namespace io
}  // namespace mxnet
