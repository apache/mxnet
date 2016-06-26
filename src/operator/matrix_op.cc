/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cc
 * \brief CPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile CPU version
#include "./matrix_op-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(TransposeParam);
DMLC_REGISTER_PARAMETER(ExpandDimParam);
DMLC_REGISTER_PARAMETER(SimpleCropParam);
DMLC_REGISTER_PARAMETER(SliceParam);
DMLC_REGISTER_PARAMETER(FlipParam);
}  // op
}  // mxnet
