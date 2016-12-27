/*!
 * Copyright (c) 2016 by Contributors
 * \file nnpack_util.cc
 * \brief
 * \author Wei Wu
*/

#if MXNET_USE_NNPACK == 1
#include "nnpack_util.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_NNPACK == 1
NNPACKInitialize nnpackinitialize;
#endif

}  // namespace op
}  // namespace mxnet
