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
 * \file optimizer_op-inl.h
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/tuple.h>

#include <fstream>
#include <cstring>

#include "../operator/elemwise_op_common.h"

#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace io {

// http://www.64lines.com/jpeg-width-height
// Gets the JPEG size from the array of data passed to the function,
// file reference: http://www.obrador.com/essentialjpeg/headerinfo.htm
bool get_jpeg_size(const uint8_t* data, uint32_t data_size, int64_t *width, int64_t *height) {
  // Check for valid JPEG image
  uint32_t i = 0;  // Keeps track of the position within the file
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
        uint8_t m = data[i+1];
        if (m == 0xC0 || (m >= 0xC1 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC)) {
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

bool get_png_size(const uint8_t* data, uint32_t data_size, int64_t *width, int64_t *height) {
  if (data[0] == 0x89 && data[1] == 0x50 && data[2] ==0x4E && data[3] == 0x47) {
    uint8_t const* p = data + 16;
    *width = ((p[0]*256 + p[1])*256 + p[2])*256 + p[3];
    p += 4;
    *height = ((p[0]*256 + p[1])*256 + p[2])*256 + p[3];
    return true;
  } else {
    return false;
  }
}

struct ImdecodeParam : public dmlc::Parameter<ImdecodeParam> {
  int flag;
  bool to_rgb;
  DMLC_DECLARE_PARAMETER(ImdecodeParam) {
    DMLC_DECLARE_FIELD(flag)
    .set_lower_bound(0)
    .set_default(1)
    .describe("Convert decoded image to grayscale (0) or color (1).");
    DMLC_DECLARE_FIELD(to_rgb)
    .set_default(true)
    .describe("Whether to convert decoded image to mxnet's default RGB format "
              "(instead of opencv's default BGR).");
  }
};

DMLC_REGISTER_PARAMETER(ImdecodeParam);

struct ImreadParam : public dmlc::Parameter<ImreadParam> {
  std::string filename;
  int flag;
  bool to_rgb;
  DMLC_DECLARE_PARAMETER(ImreadParam) {
    DMLC_DECLARE_FIELD(filename)
    .describe("Name of the image file to be loaded.");
    DMLC_DECLARE_FIELD(flag)
    .set_lower_bound(0)
    .set_default(1)
    .describe("Convert decoded image to grayscale (0) or color (1).");
    DMLC_DECLARE_FIELD(to_rgb)
    .set_default(true)
    .describe("Whether to convert decoded image to mxnet's default RGB format "
              "(instead of opencv's default BGR).");
  }
};

DMLC_REGISTER_PARAMETER(ImreadParam);


#if MXNET_USE_OPENCV
void ImdecodeImpl(int flag, bool to_rgb, void* data, size_t size,
                  NDArray* out) {
  cv::Mat buf(1, size, CV_8U, data);
  cv::Mat dst;
  if (out->is_none()) {
    cv::Mat res = cv::imdecode(buf, flag);
    if (res.empty()) {
      LOG(INFO) << "Decoding failed. Invalid image file.";
      *out = NDArray();
      return;
    }
    *out = NDArray(mshadow::Shape3(res.rows, res.cols, flag == 0 ? 1 : 3),
                   Context::CPU(), false, mshadow::kUint8);
    dst = cv::Mat(out->shape()[0], out->shape()[1], flag == 0 ? CV_8U : CV_8UC3,
                  out->data().dptr_);
    res.copyTo(dst);
    CHECK(!dst.empty()) << "Failed copying buffer to output.";
  } else {
    dst = cv::Mat(out->shape()[0], out->shape()[1], flag == 0 ? CV_8U : CV_8UC3,
                out->data().dptr_);
#if (CV_MAJOR_VERSION > 3 || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3))
    cv::imdecode(buf, flag | cv::IMREAD_IGNORE_ORIENTATION, &dst);
    CHECK(!dst.empty()) << "Decoding failed. Invalid image file.";
#elif(CV_MAJOR_VERSION > 2 || (CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4))
    cv::imdecode(buf, flag, &dst);
    CHECK(!dst.empty()) << "Decoding failed. Invalid image file.";
#else
    cv::Mat tmp = cv::imdecode(buf, flag);
    CHECK(!tmp.empty()) << "Decoding failed. Invalid image file.";
    tmp.copyTo(dst);
    CHECK(!dst.empty()) << "Failed copying buffer to output.";
#endif
  }
  CHECK_EQ(static_cast<void*>(dst.ptr()), out->data().dptr_);
  if (to_rgb && flag != 0) {
    cv::cvtColor(dst, dst, CV_BGR2RGB);
  }
}
#endif  // MXNET_USE_OPENCV

void Imdecode(const nnvm::NodeAttrs& attrs,
              const std::vector<NDArray>& inputs,
              std::vector<NDArray>* outputs) {
#if MXNET_USE_OPENCV
  const auto& param = nnvm::get<ImdecodeParam>(attrs.parsed);

  CHECK_EQ(inputs[0].ctx().dev_mask(), Context::kCPU) << "Only supports cpu input";
  CHECK_EQ(inputs[0].dtype(), mshadow::kUint8) << "Input needs to be uint8 buffer";
  inputs[0].WaitToRead();

  uint8_t* str_img = inputs[0].data().dptr<uint8_t>();
  size_t len = inputs[0].shape().Size();
  TShape oshape(3);
  oshape[2] = param.flag == 0 ? 1 : 3;
  if (get_jpeg_size(str_img, len, &oshape[1], &oshape[0])) {
  } else if (get_png_size(str_img, len, &oshape[1], &oshape[0])) {
  } else {
    (*outputs)[0] = NDArray();
    ImdecodeImpl(param.flag, param.to_rgb, str_img, len, &((*outputs)[0]));
    return;
  }

  const NDArray& ndin = inputs[0];
  NDArray& ndout = (*outputs)[0];
  ndout = NDArray(oshape, Context::CPU(), true, mshadow::kUint8);
  Engine::Get()->PushSync([ndin, ndout, str_img, len, param](RunContext ctx){
      ImdecodeImpl(param.flag, param.to_rgb, str_img, len,
                   const_cast<NDArray*>(&ndout));
    }, ndout.ctx(), {ndin.var()}, {ndout.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE("Imdecode"));
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image io.";
#endif  // MXNET_USE_OPENCV
}

void Imread(const nnvm::NodeAttrs& attrs,
            const std::vector<NDArray>& inputs,
            std::vector<NDArray>* outputs) {
#if MXNET_USE_OPENCV
  const auto& param = nnvm::get<ImreadParam>(attrs.parsed);

  std::ifstream file(param.filename, std::ios::binary | std::ios::ate);
  // if file is not open we get bad alloc after tellg
  CHECK(file.is_open()) << "Imread: '" << param.filename
      << "' couldn't open file: " << strerror(errno);
  size_t fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::shared_ptr<uint8_t> buff(new uint8_t[fsize], std::default_delete<uint8_t[]>());
  file.read(reinterpret_cast<char*>(buff.get()), fsize);
  CHECK(file.good()) << "Failed reading image file: '" << param.filename << "' "
            << strerror(errno);

  TShape oshape(3);
  oshape[2] = param.flag == 0 ? 1 : 3;
  if (get_jpeg_size(buff.get(), fsize, &oshape[1], &oshape[0])) {
  } else if (get_png_size(buff.get(), fsize, &oshape[1], &oshape[0])) {
  } else {
    (*outputs)[0] = NDArray();
    ImdecodeImpl(param.flag, param.to_rgb, buff.get(), fsize, &((*outputs)[0]));
    return;
  }

  NDArray& ndout = (*outputs)[0];
  ndout = NDArray(oshape, Context::CPU(), true, mshadow::kUint8);
  Engine::Get()->PushSync([ndout, buff, fsize, param](RunContext ctx){
      ImdecodeImpl(param.flag, param.to_rgb, buff.get(), fsize,
                   const_cast<NDArray*>(&ndout));
    }, ndout.ctx(), {}, {ndout.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE("Imread"));
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image io.";
#endif  // MXNET_USE_OPENCV
}


struct ResizeParam : public dmlc::Parameter<ResizeParam> {
  int w;
  int h;
  int interp;
  DMLC_DECLARE_PARAMETER(ResizeParam) {
    DMLC_DECLARE_FIELD(w)
    .set_lower_bound(1)
    .describe("Width of resized image.");
    DMLC_DECLARE_FIELD(h)
    .set_lower_bound(1)
    .describe("Height of resized image.");
    DMLC_DECLARE_FIELD(interp)
    .set_default(1)
    .describe("Interpolation method (default=cv2.INTER_LINEAR).");
  }
};
DMLC_REGISTER_PARAMETER(ResizeParam);

inline bool ResizeShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  const auto& param = nnvm::get<ResizeParam>(attrs.parsed);
  if (ishape->size() != 1 || (*ishape)[0].ndim() != 3) return false;

  oshape->clear();
  oshape->push_back(mshadow::Shape3(param.h, param.w, (*ishape)[0][2]));
  return true;
}

inline void Imresize(const nnvm::NodeAttrs& attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "imresize doesn't support fp16";
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[2]);
  const auto& param = nnvm::get<ResizeParam>(attrs.parsed);
  cv::Mat buf(inputs[0].shape_[0], inputs[0].shape_[1], cv_type, inputs[0].dptr_);
  cv::Mat dst(outputs[0].shape_[0], outputs[0].shape_[1], cv_type, outputs[0].dptr_);
  cv::resize(buf, dst, cv::Size(param.w, param.h), 0, 0, param.interp);
  CHECK(!dst.empty());
  CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image io.";
#endif  // MXNET_USE_OPENCV
}


struct MakeBorderParam : public dmlc::Parameter<MakeBorderParam> {
  int top, bot, left, right;
  int type;
  double value;
  nnvm::Tuple<double> values;
  DMLC_DECLARE_PARAMETER(MakeBorderParam) {
    DMLC_DECLARE_FIELD(top)
    .describe("Top margin.");
    DMLC_DECLARE_FIELD(bot)
    .describe("Bottom margin.");
    DMLC_DECLARE_FIELD(left)
    .describe("Left margin.");
    DMLC_DECLARE_FIELD(right)
    .describe("Right margin.");
    DMLC_DECLARE_FIELD(type)
    .set_default(0)
    .describe("Filling type (default=cv2.BORDER_CONSTANT).");
    DMLC_DECLARE_FIELD(value)
    .set_default(0.0)
    .describe("(Deprecated! Use ``values`` instead.) Fill with single value.");
    DMLC_DECLARE_FIELD(values)
    .set_default({})
    .describe("Fill with value(RGB[A] or gray), up to 4 channels.");
  }
};
DMLC_REGISTER_PARAMETER(MakeBorderParam);

inline bool MakeBorderShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  const auto& param = nnvm::get<MakeBorderParam>(attrs.parsed);
  if (ishape->size() != 1 || (*ishape)[0].ndim() != 3) return false;

  oshape->clear();
  oshape->push_back(
    mshadow::Shape3((*ishape)[0][0]+param.top+param.bot,
                    (*ishape)[0][1]+param.left+param.right,
                    (*ishape)[0][2]));
  return true;
}

inline void copyMakeBorder(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "imresize doesn't support fp16";
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[2]);
  const auto& param = nnvm::get<MakeBorderParam>(attrs.parsed);
  cv::Mat buf(inputs[0].shape_[0], inputs[0].shape_[1], cv_type, inputs[0].dptr_);
  cv::Mat dst(outputs[0].shape_[0], outputs[0].shape_[1], cv_type, outputs[0].dptr_);
  cv::Scalar color(param.value, param.value, param.value);
  if (param.values.ndim() > 0) {
    color = cv::Scalar(cv::Vec<double, 4>(param.values.begin()));
  }
  cv::copyMakeBorder(buf, dst, param.top, param.bot, param.left, param.right, param.type, color);
  CHECK(!dst.empty());
  CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image io.";
#endif  // MXNET_USE_OPENCV
}

NNVM_REGISTER_OP(_cvimdecode)
.describe("Decode image with OpenCV. \n"
          "Note: return image in RGB by default, "
          "instead of OpenCV's default BGR.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(op::ParamParser<ImdecodeParam>)
.set_attr<FNDArrayFunction>("FNDArrayFunction", Imdecode)
.add_argument("buf", "NDArray", "Buffer containing binary encoded image")
.add_arguments(ImdecodeParam::__FIELDS__());

NNVM_REGISTER_OP(_cvimread)
.describe("Read and decode image with OpenCV. \n"
          "Note: return image in RGB by default, "
          "instead of OpenCV's default BGR.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(op::ParamParser<ImreadParam>)
.set_attr<FNDArrayFunction>("FNDArrayFunction", Imread)
.add_arguments(ImreadParam::__FIELDS__());

NNVM_REGISTER_OP(_cvimresize)
.describe("Resize image with OpenCV. \n")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(op::ParamParser<ResizeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ResizeShape)
.set_attr<nnvm::FInferType>("FInferType", op::ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Imresize)
.add_argument("src", "NDArray", "source image")
.add_arguments(ResizeParam::__FIELDS__());

NNVM_REGISTER_OP(_cvcopyMakeBorder)
.describe("Pad image border with OpenCV. \n")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(op::ParamParser<MakeBorderParam>)
.set_attr<nnvm::FInferShape>("FInferShape", MakeBorderShape)
.set_attr<nnvm::FInferType>("FInferType", op::ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", copyMakeBorder)
.add_argument("src", "NDArray", "source image")
.add_arguments(MakeBorderParam::__FIELDS__());

}  // namespace io
}  // namespace mxnet
