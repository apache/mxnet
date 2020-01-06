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
 *  Copyright (c) 2019 by Contributors
 * \file opencv_compatibility.h
 * \brief To be compatible with multiple versions of opencv
 */
#ifndef MXNET_IO_OPENCV_COMPATIBILITY_H_
#define MXNET_IO_OPENCV_COMPATIBILITY_H_

#if MXNET_USE_OPENCV
#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 4
#include <opencv2/opencv.hpp>
#define CV_RGB2GRAY cv::COLOR_RGB2GRAY
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY

#define CV_GRAY2RGB cv::COLOR_GRAY2RGB
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR

#define CV_RGB2HLS cv::COLOR_RGB2HLS
#define CV_BGR2HLS cv::COLOR_BGR2HLS

#define CV_HLS2RGB cv::COLOR_HLS2RGB
#define CV_HLS2BGR cv::COLOR_HLS2BGR

#define CV_RGB2BGR cv::COLOR_RGB2BGR
#define CV_BGR2RGB cv::COLOR_BGR2RGB

#define CV_INTER_LINEAR cv::INTER_LINEAR
#define CV_INTER_NEAREST cv::INTER_NEAREST

#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_IMWRITE_PNG_COMPRESSION cv::IMWRITE_PNG_COMPRESSION
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY

#endif  // CV_VERSION_MAJOR >= 4

#endif  // MXNET_USE_OPENCV

#endif  // MXNET_IO_OPENCV_COMPATIBILITY_H_
