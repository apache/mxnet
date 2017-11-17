/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
