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
 * \file image_utils.h
 * \brief the image operator utility function implementation
 * \author Jake Lee
 */

#ifndef MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_

#include <vector>
#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace op {
namespace image {

enum ImageLayout {H, W, C};
enum BatchImageLayout {N, kH, kW, kC};

struct SizeParam {
  int height;
  int width;
  SizeParam() {
    height = 0;
    width = 0;
  }
  SizeParam(int height_, int width_) {
    height = height_;
    width = width_;
  }
};  // struct SizeParam

}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_
