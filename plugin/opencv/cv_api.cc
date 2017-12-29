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
#include <dmlc/base.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <opencv2/opencv.hpp>
#include "cv_api.h"
#include "../../src/c_api/c_api_common.h"


using namespace mxnet;
// http://www.64lines.com/jpeg-width-height
// Gets the JPEG size from the array of data passed to the function, file reference: http://www.obrador.com/essentialjpeg/headerinfo.htm
bool get_jpeg_size(const unsigned char* data, mx_uint data_size, mx_uint *width, mx_uint *height) {
  // Check for valid JPEG image
  mx_uint i = 0;  // Keeps track of the position within the file
  if (data[i] == 0xFF && data[i+1] == 0xD8 && data[i+2] == 0xFF && data[i+3] == 0xE0) {
    i += 4;
    // Check for valid JPEG header (null terminated JFIF)
    if (data[i+2] == 'J' && data[i+3] == 'F' && data[i+4] == 'I'
        && data[i+5] == 'F' && data[i+6] == 0x00) {
      // Retrieve the block length of the first block since
      // the first block will not contain the size of file
      uint16_t block_length = data[i] * 256 + data[i+1];
      while (i < data_size) {
        i+=block_length;  // Increase the file index to get to the next block
        if (i >= data_size) return false;  // Check to protect against segmentation faults
        if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
        if (data[i+1] == 0xC0) {
          // 0xFFC0 is the "Start of frame" marker which contains the file size
          // The structure of the 0xFFC0 block is quite simple
          // [0xFFC0][ushort length][uchar precision][ushort x][ushort y]
          *height = data[i+5]*256 + data[i+6];
          *width = data[i+7]*256 + data[i+8];
          return true;
        } else {
          i+=2;  // Skip the block marker
          block_length = data[i] * 256 + data[i+1];  // Go to the next block
        }
      }
      return false;  // If this point is reached then no size was found
    } else {
      return false;  // Not a valid JFIF string
    }
  } else {
    return false;  // Not a valid SOI header
  }
}

bool get_png_size(const unsigned char* data, mx_uint data_size, mx_uint *width, mx_uint *height) {
  if (data[0] == 0x89 && data[1] == 0x50 && data[2] ==0x4E && data[3] == 0x47) {
    unsigned char const* p = data + 16;
    *width = ((p[0]*256 + p[1])*256 + p[2])*256 + p[3];
    p += 4;
    *height = ((p[0]*256 + p[1])*256 + p[2])*256 + p[3];
    return true;
  } else {
    return false;
  }
}

MXNET_DLL int MXCVImdecode(const unsigned char *img, const mx_uint len,
                           const int flag, NDArrayHandle *out) {
  API_BEGIN();
  mx_uint dims[3];
  CHECK_GE(flag, 0) << "flag must be 0 (grayscale) or 1 (colored).";
  dims[2] = flag == 0 ? 1 : 3;
  if (get_jpeg_size(img, len, dims+1, dims)) {
  } else if (get_png_size(img, len, dims+1, dims)) {
  } else {
    LOG(FATAL) << "Only supports png and jpg.";
  }
  NDArray ndout(TShape(dims, dims+3), Context::CPU(), true, mshadow::kUint8);
  unsigned char *img_cpy = new unsigned char[len];
  memcpy(img_cpy, img, sizeof(unsigned char)*len);
  Engine::Get()->PushSync([=](RunContext ctx){
      ndout.CheckAndAlloc();
      cv::Mat buf(1, len, CV_8U, img_cpy);
      cv::Mat dst(dims[0], dims[1], flag == 0 ? CV_8U : CV_8UC3, ndout.data().dptr_);
#if (CV_MAJOR_VERSION > 3 || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3))
      cv::imdecode(buf, flag | cv::IMREAD_IGNORE_ORIENTATION, &dst);
#else
      cv::imdecode(buf, flag, &dst);
#endif
      CHECK(!dst.empty());
      delete[] img_cpy;
    }, ndout.ctx(), {}, {ndout.var()});
  NDArray *tmp = new NDArray();
  *tmp = ndout;
  *out = tmp;
  API_END();
}


MXNET_DLL int MXCVResize(NDArrayHandle src, const mx_uint w, const mx_uint h,
                         const int interpolation, NDArrayHandle *out) {
  API_BEGIN();
  NDArray ndsrc = *static_cast<NDArray*>(src);
  CHECK_EQ(ndsrc.shape().ndim(), 3);
  CHECK_EQ(ndsrc.ctx(), Context::CPU());
  CHECK_EQ(ndsrc.dtype(), mshadow::kUint8);

  mx_uint dims[3] = {h, w, ndsrc.shape()[2]};
  NDArray ndout(TShape(dims, dims+3), Context::CPU(), true, mshadow::kUint8);

  Engine::Get()->PushSync([=](RunContext ctx){
      ndout.CheckAndAlloc();
      cv::Mat buf(ndsrc.shape()[0], ndsrc.shape()[1],
                  dims[2] == 3 ? CV_8UC3 : CV_8U, ndsrc.data().dptr_);
      cv::Mat dst(h, w, dims[2] == 3 ? CV_8UC3 : CV_8U, ndout.data().dptr_);
      cv::resize(buf, dst, cv::Size(w, h), 0, 0, interpolation);
      CHECK(!dst.empty());
    }, ndout.ctx(), {ndsrc.var()}, {ndout.var()});
  NDArray *tmp = new NDArray();
  *tmp = ndout;
  *out = tmp;
  API_END();
}

MXNET_DLL int MXCVcopyMakeBorder(NDArrayHandle src,
                                 const int top,
                                 const int bot,
                                 const int left,
                                 const int right,
                                 const int type,
                                 const double value,
                                 NDArrayHandle *out) {
  API_BEGIN();
  NDArray ndsrc = *static_cast<NDArray*>(src);
  CHECK_EQ(ndsrc.shape().ndim(), 3);
  CHECK_EQ(ndsrc.ctx(), Context::CPU());
  CHECK_EQ(ndsrc.dtype(), mshadow::kUint8);

  int h = ndsrc.shape()[0], w = ndsrc.shape()[1], c = ndsrc.shape()[2];
  mx_uint dims[3] = {top+h+bot, left+w+right, c};
  NDArray ndout(TShape(dims, dims+3), Context::CPU(), true, mshadow::kUint8);

  Engine::Get()->PushSync([=](RunContext ctx){
      ndout.CheckAndAlloc();
      cv::Mat buf(h, w, c == 3 ? CV_8UC3 : CV_8U, ndsrc.data().dptr_);
      cv::Mat dst(top+h+bot, left+w+right, c == 3 ? CV_8UC3 : CV_8U, ndout.data().dptr_);
      cv::copyMakeBorder(buf, dst, top, bot, left, right, type, cv::Scalar(value));
      CHECK(!dst.empty());
    }, ndout.ctx(), {ndsrc.var()}, {ndout.var()});
  NDArray *tmp = new NDArray();
  *tmp = ndout;
  *out = tmp;
  API_END();
}
