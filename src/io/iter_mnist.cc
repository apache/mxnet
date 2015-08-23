/*!
 * Copyright (c) 2015 by Contributors
 * \file iter_mnist.cc
 * \brief register mnist iterator
 * \author Tianjun Xiao
*/
#include "./iter_mnist-inl.h"

namespace mxnet {
namespace io {

DMLC_REGISTER_PARAMETER(MNISTParam);
MXNET_REGISTER_IO_ITER(MNIST, MNISTIterator)
    .describe("Create MNISTIterator")
    .add_arguments(MNISTParam::__FIELDS__());

}  // namespace io
}  // namespace mxnet
