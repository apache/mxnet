/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuation of mxnet
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_
#include <dmlc/base.h>
#include <mshadow/tensor.h>

/*!
 *\brief whether to use opencv support
 */
#ifndef MXNET_USE_OPENCV
#define MXNET_USE_OPENCV 1
#endif

/*!
 *\brief whether to use cudnn library for convolution
 */
#ifndef MXNET_USE_CUDNN
#define MXNET_USE_CUDNN 0
#endif

/*! \brief namespace of mxnet */
namespace mxnet {
typedef mshadow::cpu cpu;
typedef mshadow::gpu gpu;
/*! \brief index type usually use unsigned */
typedef mshadow::index_t index_t;
/*! \brief data type that will be used to store ndarray */
typedef mshadow::default_real_t real_t;

}  // namespace mxnet
#endif  // MXNET_BASE_H_
