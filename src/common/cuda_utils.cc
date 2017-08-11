/*!
 * \file cuda_utils.cc
 * \brief CUDA debugging utilities and common functions.
 */

#include <mxnet/base.h>
#include "cuda_utils.h"

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {

const double CublasType<double>::one = 1.0;
const double CublasType<double>::zero = 0.0;

const float CublasType<float>::one = 1.0f;
const float CublasType<float>::zero = 0.0f;

const mshadow::half::half_t CublasType<mshadow::half::half_t>::one = 1.0f;
const mshadow::half::half_t CublasType<mshadow::half::half_t>::zero = 0.0f;

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
