/*!
 *  Copyright (c) 2016 by Contributors
 * \file cv_api.h
 * \brief C API for opencv
 * \author Junyuan Xie
 */
#ifndef PLUGIN_OPENCV_CV_API_H_
#define PLUGIN_OPENCV_CV_API_H_

#include <mxnet/c_api.h>

MXNET_DLL int MXCVImdecode(
  const unsigned char *img,
  const mx_uint len,
  const int flag,
  NDArrayHandle *out);

MXNET_DLL int MXCVResize(
  NDArrayHandle src,
  const mx_uint w,
  const mx_uint h,
  const int interpolation,
  NDArrayHandle *out);

MXNET_DLL int MXCVcopyMakeBorder(
  NDArrayHandle src,
  const int top,
  const int bot,
  const int left,
  const int right,
  const int type,
  const double value,
  NDArrayHandle *out);

#endif  // PLUGIN_OPENCV_CV_API_H_
