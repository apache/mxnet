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
#include <mxnet/tensor_blob.h>

#include "../imperative/cached_op.h"

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
    /*! \brief the path separator character, by default it's ; */
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

    std::vector<NDArray> GetItem(uint64_t idx, std::vector<int> &is_scalar) const {
      is_scalar.resize(1);
      is_scalar[0] = 0;
#if MXNET_USE_OPENCV
      CHECK_LT(idx, img_list_.size())
        << "GetItem index: " << idx << " out of bound: " << img_list_.size();
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
      return std::vector<NDArray>({ret});
#else
    LOG(FATAL) << "Opencv is needed for image decoding.";
#endif
    };

  private:
    /*! \brief parameters */
    ImageSequenceDatasetParam param_;
    /*! \brief image list */
    std::vector<std::string> img_list_;

#if MXNET_USE_OPENCV
    template<int n_channels>
    NDArray SwapImageChannels(cv::Mat &img) const {
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

      TShape arr_shape = TShape({img.rows, img.cols, n_channels});
      NDArray arr(arr_shape, mxnet::Context::CPU(0), true, mshadow::kUint8);
      auto ptr = static_cast<uint8_t*>(arr.data().dptr_);

      // swap channels while copying elements into buffer
      for (int i = 0; i < img.rows; ++i) {
        const uint8_t* im_data = img.ptr<uint8_t>(i);
        uint8_t* buffer_data = ptr + i * img.cols * n_channels;
        for (int j = 0; j < img.cols; ++j) {
          for (int k = 0; k < n_channels; ++k) {
            buffer_data[k] = im_data[swap_indices[k]];
          }
          im_data += n_channels;
          buffer_data += n_channels;
        }
      }
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
      // std::vector<std::pair<std::string, std::string> > kwargs_left;
      param_.InitAllowUnknown(kwargs);
      data_ = *(static_cast<NDArray*>(reinterpret_cast<void*>(param_.arr)));
      if (data_.shape().ndim() < 1) {
        LOG(FATAL) << "NDArray with no dim is not iterable";
      }
      size_ = data_.shape().begin()[0];
    }

    uint64_t GetLen() const {
      return size_;
    }

    std::vector<NDArray> GetItem(uint64_t idx, std::vector<int> &is_scalar) const {
      is_scalar.resize(1);
      CHECK_LT(idx, size_)
        << "GetItem index: " << idx << " out of bound: " << size_;
      NDArray ret = data_.Slice(idx, idx + 1);
      if (ret.shape().ndim() > 1) {
        // remove first dim to be consistent with numpy
        TShape new_shape;
        new_shape.assign(ret.shape().begin() + 1, ret.shape().end());
        ret = ret.Reshape(new_shape);
        is_scalar[0] = 0;
      } else {
        if (data_.shape().ndim() == 1) {
          is_scalar[0] = 1;
        }
      }
      return std::vector<NDArray>({ret});
    };

  private:
    /*! \brief parameters */
    NDArrayDatasetParam param_;
    /*! \brief stored ndarray */
    NDArray data_;
    /*! \brief stored ndarray shape */
    int64_t size_;
};  // class NDArrayDataset

MXNET_REGISTER_IO_DATASET(NDArrayDataset)
 .describe("Single NDArray Dataset")
 .add_arguments(NDArrayDatasetParam::__FIELDS__())
 .set_body([]() {
     return new NDArrayDataset();
});

struct GroupDatasetParam : public dmlc::Parameter<GroupDatasetParam> {
    /*! \brief the source ndarray */
    Tuple<std::intptr_t> datasets;
    // declare parameters
    DMLC_DECLARE_PARAMETER(GroupDatasetParam) {
        DMLC_DECLARE_FIELD(datasets)
            .describe("A small set of pointers to other c++ datasets.");
    }
};  // struct GroupDatasetParam

DMLC_REGISTER_PARAMETER(GroupDatasetParam);

class GroupDataset : public Dataset {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      std::vector<std::pair<std::string, std::string> > kwargs_left;
      param_.InitAllowUnknown(kwargs);
      auto childs = param_.datasets;
      childs_.reserve(childs.ndim());
      size_t child_cnt = 0;
      for (auto child : childs) {
        auto d = *static_cast<DatasetPtr*>(reinterpret_cast<void*>(child));
        if (child_cnt == 0) {
          size_ = d->GetLen();
        } else {
          CHECK_EQ(size_, d->GetLen())
            << "All child dataset of GroupDataset must be identical "
            << "Given mismatch: " << size_ << " vs " << d->GetLen();
        }
        childs_.emplace_back(d);
        child_cnt++;
      }
    }

    uint64_t GetLen() const {
      return size_;
    }

    std::vector<NDArray> GetItem(uint64_t idx, std::vector<int> &is_scalar) const {
      CHECK_LT(idx, size_)
        << "GetItem index: " << idx << " out of bound: " << size_;
      std::vector<NDArray> ret;
      is_scalar.clear();
      for (auto child : childs_) {
        std::vector<int> temp_scalar;
        auto v = child->GetItem(idx, temp_scalar);
        ret.insert(ret.end(), v.begin(), v.end());
        for (size_t j = 0; j < v.size(); ++j) {
          is_scalar.emplace_back(temp_scalar[j]);
        }
      }
    };

  private:
    /*! \brief parameters */
    GroupDatasetParam param_;
    /*! \brief stored child datasets */
    std::vector<DatasetPtr> childs_;
    /*! \brief overall dataset size, equals to all child datasets */
    uint64_t size_;
};   // class GroupDataset

MXNET_REGISTER_IO_DATASET(GroupDataset)
 .describe("Grouped Dataset that combine a bunch of datasets")
 .add_arguments(GroupDatasetParam::__FIELDS__())
 .set_body([]() {
     return new GroupDataset();
});

struct LazyTransformDatasetParam : public dmlc::Parameter<LazyTransformDatasetParam> {
    /*! \brief the source ndarray */
    std::intptr_t cached_op;
    /*! \brief internal dataset */
    std::intptr_t dataset;
    /*! \brief indices for items that needs transformation */
    Tuple<int> transform_indices;
    // declare parameters
    DMLC_DECLARE_PARAMETER(LazyTransformDatasetParam) {
        DMLC_DECLARE_FIELD(cached_op)
            .describe("Pointer to cached transform function.");
        DMLC_DECLARE_FIELD(dataset)
            .describe("Pointer to internal dataset.");
        DMLC_DECLARE_FIELD(transform_indices).set_default(Tuple<int>({}))
            .describe("The indices for dataset items that need to be transformed/processed. "
                      "If `transform_indices` is empty(default), then all items will be processed.");
    }
};  // struct LazyTransformDatasetParam

DMLC_REGISTER_PARAMETER(LazyTransformDatasetParam);

class LazyTransformDataset : public Dataset {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      cached_op_ = *static_cast<CachedOpPtr*>(reinterpret_cast<void*>(param_.cached_op));
      base_data_ = *static_cast<DatasetPtr*>(reinterpret_cast<void*>(param_.dataset));
    }

    uint64_t GetLen() const {
      return base_data_->GetLen();
    }

    std::vector<NDArray> GetItem(uint64_t idx, std::vector<int> &is_scalar) const {
      return std::vector<NDArray>();
    };

  private:
    /*! \brief parameters */
    LazyTransformDatasetParam param_;
    /*! \brief stored cached op */
    CachedOpPtr cached_op_;
    /*! \brief internal dataset */
    DatasetPtr base_data_;
};   // class LazyTransformDataset

MXNET_REGISTER_IO_DATASET(LazyTransformDataset)
 .describe("Dataset that apply lazy transformation to internal dataset")
 .add_arguments(LazyTransformDatasetParam::__FIELDS__())
 .set_body([]() {
     return new LazyTransformDataset();
});
}  // namespace io
}  // namespace mxnet