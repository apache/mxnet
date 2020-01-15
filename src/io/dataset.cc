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
 *  Copyright (c) 2015 by Contributors
 * \file dataset.cc
 * \brief High performance datasets implementation
 */
#include <dmlc/parameter.h>
#include <mxnet/io.h>
#include <mxnet/ndarray.h>

#include <string>
#include <vector>

#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
  #include "./opencv_compatibility.h"
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace io {
struct ImageSequenceDatasetParam : public dmlc::Parameter<ImageSequenceDatasetParam> {
    /*! \brief the list of absolute image paths, separated by \0 characters */
    std::string img_list;
    /*! \brief If flag is 0, always convert to grayscale(1 channel). 
    * If flag is 1, always convert to colored (3 channels).
    * */
    int flag;
    // declare parameters
    DMLC_DECLARE_PARAMETER(ImageSequenceDatasetParam) {
        DMLC_DECLARE_FIELD(img_list)
            .describe("The list of image absolute paths.");
        DMLC_DECLARE_FIELD(flag).set_default(1)
            .describe("If 1, always convert to colored, if 0 always convert to grayscale.");
    }
};  // struct ImageSequenceDatasetParam

DMLC_REGISTER_PARAMETER(ImageSequenceDatasetParam);

class ImageSequenceDataset : public Dataset {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      std::vector<std::pair<std::string, std::string> > kwargs_left;
      param_.InitAllowUnknown(kwargs);
      img_list_ = dmlc::Split(param_.img_list, ';');
    }

    uint64_t GetLen() const {
      return img_list_.size();
    }

    int GetOutputSize() const {
      return 1;
    }

    NDArray GetItem(uint64_t idx, int n) {
#if MXNET_USE_OPENCV
      CHECK_LT(idx, img_list_.size())
        << "GetItem index: " << idx << " out of bound: " << img_list_.size();
      CHECK_EQ(n, 0) << "ImageSequenceDataset only produce one output";
      cv::Mat res = cv::imread(img_list_[idx], param_.flag);
      const int n_channels = res.channels();
      NDArray ret;
      if (n_channels == 1) {
        ret = SwapImageChannels<1>(res);
      } else if (n_channels == 3) {
        ret = SwapImageChannels<3>(res);
      } else if (n_channels == 4) {
        ret = SwapImageChannels<4>(res);
      }
      return ret;
#else
    LOG(FATAL) << "Opencv is needed for image decoding.";
#endif
    };

  private:
    /*! \brief parameters */
    ImageSequenceDatasetParam param_;
    /*! \brief image list */
    std::vector<std::string> img_list_;
    /*! \brief image process buffer */
    std::vector<uint8_t> buffer_;

#if MXNET_USE_OPENCV
    template<int n_channels>
    NDArray SwapImageChannels(cv::Mat &img) {
      int swap_indices[n_channels]; // NOLINT(*)
      if (n_channels == 1) {
        swap_indices[0] = 0;
      } else if (n_channels == 3) {
        swap_indices[0] = 2;
        swap_indices[1] = 1;
        swap_indices[2] = 0;
      } else if (n_channels == 4) {
        swap_indices[0] = 2;
        swap_indices[1] = 1;
        swap_indices[2] = 0;
        swap_indices[3] = 3;
      }

      std::size_t size = img.rows * img.cols * n_channels;
      if (buffer_.size() < size) {
        // increase buffer size only when it's less than the new image
        // otherwise stay unchanged, in case next image is larger
        buffer_.resize(size);
      }

      // swap channels while copying elements into buffer
      for (int i = 0; i < img.rows; ++i) {
        const uint8_t* im_data = img.ptr<uint8_t>(i);
        uint8_t* buffer_row = buffer_.data() + i * img.cols * n_channels;
        for (int j = 0; j < img.cols; ++j) {
          uint8_t* buffer_col = buffer_row + j * n_channels;
          for (int k = 0; k < n_channels; ++k) {
            buffer_col[k] = im_data[swap_indices[k]];
          }
        }
      }

      // sync copy into ndarray
      TShape arr_shape = TShape({img.rows, img.cols, n_channels});
      NDArray arr(arr_shape, mxnet::Context::CPU(0), true, mshadow::kUint8);
      arr.SyncCopyFromCPU(buffer_.data(), size);
      return arr;
    }
#endif
};

MXNET_REGISTER_IO_DATASET(ImageSequenceDataset)
 .describe("Image Sequence Dataset")
 .add_arguments(ImageSequenceDatasetParam::__FIELDS__())
 .set_body([]() {
     return new ImageSequenceDataset();
});

}  // namespace io
}  // namespace mxnet