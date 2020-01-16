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
    /*! \brief the path separator character, by default it's \n */
    char path_sep;
    /*! \brief If flag is 0, always convert to grayscale(1 channel). 
    * If flag is 1, always convert to colored (3 channels).
    * If flag is -1, keep channels unchanged.
    */
    int flag;
    // declare parameters
    DMLC_DECLARE_PARAMETER(ImageSequenceDatasetParam) {
        DMLC_DECLARE_FIELD(img_list)
            .describe("The list of image absolute paths.");
        DMLC_DECLARE_FIELD(path_sep).set_default('|')
            .describe("The path separator for joined image paths.");
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
      img_list_ = dmlc::Split(param_.img_list, param_.path_sep);
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
        uint8_t* buffer_data = buffer_.data() + i * img.cols * n_channels;
        for (int j = 0; j < img.cols; ++j) {
          for (int k = 0; k < n_channels; ++k) {
            buffer_data[k] = im_data[swap_indices[k]];
          }
          im_data += n_channels;
          buffer_data += n_channels;
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

struct NDArrayDatasetParam : public dmlc::Parameter<NDArrayDatasetParam> {
    /*! \brief the source ndarray */
    std::intptr_t arr;
    // declare parameters
    DMLC_DECLARE_PARAMETER(NDArrayDatasetParam) {
        DMLC_DECLARE_FIELD(arr)
            .describe("Pointer to NDArray.");
    }
};  // struct NDArrayDatasetParam

DMLC_REGISTER_PARAMETER(NDArrayDatasetParam);

class NDArrayDataset : public Dataset {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      std::vector<std::pair<std::string, std::string> > kwargs_left;
      param_.InitAllowUnknown(kwargs);
      data_ = *(static_cast<NDArray*>(reinterpret_cast<void*>(param_.arr)));
      if (data_.shape().ndim() < 1) {
        LOG(FATAL) << "NDArray with no dim is not iterable";
      }
      size_ = data_.shape().begin()[0];
    }

    uint64_t GetLen() const {
      return size_;;
    }

    int GetOutputSize() const {
      return 1;
    }

    NDArray GetItem(uint64_t idx, int n) {
      CHECK_LT(idx, size_)
        << "GetItem index: " << idx << " out of bound: " << size_;
      CHECK_EQ(n, 0) << "NDArrayDataset only produce one output";
      NDArray ret = data_.Slice(idx, idx + 1);
      return ret;
    };

  private:
    /*! \brief parameters */
    NDArrayDatasetParam param_;
    /*! \brief stored ndarray */
    NDArray data_;
    /*! \brief stored ndarray shape */
    int64_t size_;
};    // class NDArrayDataset

MXNET_REGISTER_IO_DATASET(NDArrayDataset)
 .describe("Single NDArray Dataset")
 .add_arguments(NDArrayDatasetParam::__FIELDS__())
 .set_body([]() {
     return new NDArrayDataset();
});

struct TupleDatasetParam : public dmlc::Parameter<TupleDatasetParam> {
    /*! \brief the source ndarray */
    Tuple<std::intptr_t> datasets;
    // declare parameters
    DMLC_DECLARE_PARAMETER(TupleDatasetParam) {
        DMLC_DECLARE_FIELD(datasets)
            .describe("A small set of pointers to other c++ datasets.");
    }
};  // struct TupleDatasetParam

DMLC_REGISTER_PARAMETER(TupleDatasetParam);

class TupleDataset : public Dataset {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      std::vector<std::pair<std::string, std::string> > kwargs_left;
      param_.InitAllowUnknown(kwargs);
      auto childs = param_.datasets;
      childs_.reserve(childs.ndim());
      item_size_ = 0;
      size_t child_cnt = 0;
      for (auto child : childs) {
        auto d = static_cast<Dataset*>(reinterpret_cast<void*>(child));
        if (child_cnt == 0) {
          size_ = d->GetLen();
        } else {
          CHECK_EQ(size_, d->GetLen())
            << "All child dataset of TupleDataset must be identical "
            << "Given mismatch: " << size_ << " vs " << d->GetLen();
        }
        childs_.emplace_back(d);
        // generate lookup table for indexing
        int old_size = item_size_;
        item_size_ += d->GetOutputSize();
        for (int i = old_size; i < item_size_; ++i) {
          idx_map_[i] = std::make_pair(child_cnt, i - old_size);
        }
        child_cnt++;
      }
    }

    uint64_t GetLen() const {
      return size_;;
    }

    int GetOutputSize() const {
      return item_size_;
    }

    NDArray GetItem(uint64_t idx, int n) {
      CHECK_LT(idx, size_)
        << "GetItem index: " << idx << " out of bound: " << size_;
      CHECK_GE(n, 0) << "Getting negative item is forbidden";
      CHECK_LT(n, item_size_) << "Item index out of bound: " << n << " vs total " << item_size_;
      auto new_idx = idx_map_[n];
      return childs_[new_idx.first]->GetItem(idx, new_idx.second);
    };

  private:
    /*! \brief parameters */
    TupleDatasetParam param_;
    /*! \brief stored child datasets */
    std::vector<Dataset*> childs_;
    /*! \brief overall dataset size, equals to all child datasets */
    uint64_t size_;
    /*! \brief overall item output size, which is the sum of childs' */
    int item_size_;
    /*! \brief a table to get the corresponding child dataset and data index */
    std::unordered_map<int, std::pair<int, int> > idx_map_;
};   // class TupleDataset

MXNET_REGISTER_IO_DATASET(TupleDataset)
 .describe("Tuple like Dataset that combine a bunch of datasets")
 .add_arguments(TupleDatasetParam::__FIELDS__())
 .set_body([]() {
     return new TupleDataset();
});
}  // namespace io
}  // namespace mxnet