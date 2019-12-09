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
 *  Copyright (c) 2017 by Contributors
 * \file export.h
 * \brief Export module that takes charge of code generation and document
 *  Generation for functions exported from R-side
 */

#ifndef MXNET_RCPP_IM2REC_H_
#define MXNET_RCPP_IM2REC_H_

#include <Rcpp.h>
#include <string>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 4
#include <opencv2/opencv.hpp>
#define CV_IMWRITE_PNG_COMPRESSION cv::IMWRITE_PNG_COMPRESSION
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#endif  // CV_VERSION_MAJOR >= 4

namespace mxnet {
namespace R {

class IM2REC {
 public:
  /*!
   * \brief Export the generated file into path.
   * \param path The path to be exported.
   */
  static void im2rec(const std::string & image_lst, const std::string & root,
                     const std::string & output_rec,
                     int label_width = 1, int pack_label = 0, int new_size = -1, int nsplit = 1,
                     int partid = 0, int center_crop = 0, int quality = 95,
                     int color_mode = 1, int unchanged = 0,
                     int inter_method = 1, std::string encoding = ".jpg");
  // intialize the Rcpp module
  static void InitRcppModule();

 public:
  // get the singleton of exporter
  static IM2REC* Get();
  /*! \brief The scope of current module to export */
  Rcpp::Module* scope_;
};

}  // namespace R
}  // namespace mxnet

#endif  // MXNET_RCPP_IM2REC_H_
