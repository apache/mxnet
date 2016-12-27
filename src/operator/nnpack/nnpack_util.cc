/*!
 * Copyright (c) 2016 by Contributors
 * \file nnpack_util.cc
 * \brief
 * \author Wei Wu
*/

#if MXNET_USE_NNPACK == 1
#include "nnpack_util.h"

namespace mxnet {
namespace op {

NNPACKInitialize nnpackinitialize;

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_NNPACK
