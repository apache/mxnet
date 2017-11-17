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
* \file image_common.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_IMAGE_COMMON_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_COMMON_H_

#include <mxnet/base.h>

namespace mxnet {
namespace op {

/**
* @brief convert TBlob to cv::Mat
* @param input @see TBlob
* @param hight
* @param weight
* @param channel
* @return
*/
static cv::Mat mat_convert(TBlob input, int hight, int weight, int channel) {
  cv::Mat m;
  switch (input.type_flag_) {
    case mshadow::kFloat32: {
      typedef float DType;
      m = cv::Mat(hight, weight, CV_MAKETYPE(CV_32F, channel), input.dptr<DType>());
    }
    break;
    case mshadow::kFloat64: {
      typedef double DType;
      m = cv::Mat(hight, weight, CV_MAKETYPE(CV_64F, channel), input.dptr<DType>());
    }
    break;
    case mshadow::kFloat16: {
      typedef mshadow::half::half_t DType;
      LOG(FATAL) << "not support type enum " << input.type_flag_;
    }
    break;
    case mshadow::kUint8: {
      typedef uint8_t DType;
      m = cv::Mat(hight, weight, CV_MAKETYPE(CV_8U, channel), input.dptr<DType>());
    }
    break;
    case mshadow::kInt8: {
      typedef int8_t DType;
      m = cv::Mat(hight, weight, CV_MAKETYPE(CV_8S, channel), input.dptr<DType>());
    }
    break;
    case mshadow::kInt32: {
      typedef int32_t DType;
      m = cv::Mat(hight, weight, CV_MAKETYPE(CV_32S, channel), input.dptr<DType>());
    }
    break;
    case mshadow::kInt64: {
      typedef int64_t DType;
      LOG(FATAL) << "not support type enum " << input.type_flag_;
    }
    break;
    default:
      LOG(FATAL) << "Unknown type enum " << input.type_flag_;
  }
  return m;
}
}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_IMAGE_IMAGE_COMMON_H_

