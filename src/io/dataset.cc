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
 *  Copyright (c) 2020 by Contributors
 * \file dataset.cc
 * \brief High performance datasets implementation
 */
#include <dmlc/parameter.h>
#include <dmlc/recordio.h>
#include <dmlc/io.h>
#include <mxnet/io.h>
#include <mxnet/ndarray.h>
#include <mxnet/tensor_blob.h>

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include "../imperative/cached_op.h"
#include "../imperative/naive_cached_op.h"
#include "../ndarray/ndarray_function.h"

#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#include "./opencv_compatibility.h"
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace io {

struct RecordFileDatasetParam : public dmlc::Parameter<RecordFileDatasetParam> {
  std::string rec_file;
  std::string idx_file;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RecordFileDatasetParam) {
      DMLC_DECLARE_FIELD(rec_file)
          .describe("The absolute path of record file.");
      DMLC_DECLARE_FIELD(idx_file)
          .describe("The path of the idx file.");
  }
};  // struct RecordFileDatasetParam

DMLC_REGISTER_PARAMETER(RecordFileDatasetParam);

class RecordFileDataset final : public Dataset {
 public:
  explicit RecordFileDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    param_.InitAllowUnknown(kwargs);
    // read and process idx file
    dmlc::Stream *idx_stream = dmlc::Stream::Create(param_.idx_file.c_str(), "r");
    dmlc::istream is(idx_stream);
    size_t key, idx;
    while (is >> key >> idx) {
      idx_[key] = idx;
    }
    delete idx_stream;
  }

  uint64_t GetLen() const override {
    return idx_.size();
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* ret) override {
    ret->resize(1);
    auto& out = (*ret)[0];
    static thread_local std::unique_ptr<dmlc::Stream> stream;
    static thread_local std::unique_ptr<dmlc::RecordIOReader> reader;
    if (!reader) {
      auto s = dmlc::Stream::Create(param_.rec_file.c_str(), "r");
      stream.reset(s);
      reader = std::make_unique<dmlc::RecordIOReader>(s);
    }
    size_t pos = idx_[static_cast<size_t>(idx)];
    reader->Seek(pos);
    static thread_local std::string read_buff;
    if (reader->NextRecord(&read_buff)) {
      const char *buf = read_buff.c_str();
      const size_t size = read_buff.size();
      out = NDArray(TShape({static_cast<dim_t>(size)}), Context::CPU(), false, mshadow::kInt8);
      TBlob dst = out.data();
      RunContext rctx{Context::CPU(), nullptr, nullptr, false};
      mxnet::ndarray::Copy<cpu, cpu>(
        TBlob(const_cast<void*>(reinterpret_cast<const void*>(buf)),
          out.shape(), cpu::kDevMask, out.dtype(), 0),
          &dst, Context::CPU(), Context::CPU(), rctx);
    }
    return true;
  }

 private:
  /*! \brief parameters */
  RecordFileDatasetParam param_;
  /*! \brief indices */
  std::unordered_map<size_t, size_t> idx_;
};

MXNET_REGISTER_IO_DATASET(RecordFileDataset)
  .describe("MXNet Record File Dataset")
  .add_arguments(RecordFileDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new RecordFileDataset(kwargs);
});

struct ImageRecordFileDatasetParam : public dmlc::Parameter<ImageRecordFileDatasetParam> {
  std::string rec_file;
  std::string idx_file;
  int flag;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecordFileDatasetParam) {
      DMLC_DECLARE_FIELD(rec_file)
          .describe("The absolute path of record file.");
      DMLC_DECLARE_FIELD(idx_file)
          .describe("The path of the idx file.");
      DMLC_DECLARE_FIELD(flag).set_default(1)
          .describe("If 1, always convert to colored, if 0 always convert to grayscale.");
  }
};  // struct ImageRecordFileDatasetParam

DMLC_REGISTER_PARAMETER(ImageRecordFileDatasetParam);

#if MXNET_USE_OPENCV
template<int n_channels>
void SwapImageChannels(const cv::Mat &img, NDArray* arr) {
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
  if (arr->is_none() || arr->shape() != arr_shape || arr->ctx() != mxnet::Context::CPU(0) ||
      arr->dtype() != mshadow::kUint8 || arr->storage_type() != kDefaultStorage) {
    *arr = NDArray(arr_shape, mxnet::Context::CPU(0), false, mshadow::kUint8);
  }
  auto ptr = static_cast<uint8_t*>(arr->data().dptr_);

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
}
#endif

/*! \brief Struct for unpack recordio header */
#pragma pack(1)
struct IRHeader {
  uint32_t flag;
  float label;
  uint64_t id;
  uint64_t id2;
};  // struct IRHeader

class ImageRecordFileDataset : public Dataset {
 public:
  explicit ImageRecordFileDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    param_.InitAllowUnknown(kwargs);
    base_ = std::make_shared<RecordFileDataset>(kwargs);
  }

  uint64_t GetLen() const override {
    return base_->GetLen();
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* ret) override {
    CHECK_LT(idx, GetLen());
    std::vector<NDArray> raw;
    if (!base_->GetItem(idx, &raw)) return false;
    CHECK_EQ(raw.size(), 1U) << "RecordFileDataset should return size 1 NDArray vector";
    uint8_t *s = reinterpret_cast<uint8_t*>(raw[0].data().dptr_);
    size_t size = raw[0].shape().Size();
    CHECK_GT(size, sizeof(IRHeader)) << "Invalid size of bytes from Record File";
    IRHeader header;
    std::memcpy(&header, s, sizeof(header));
    size -= sizeof(header);
    s += sizeof(header);
    NDArray label = NDArray(Context::CPU(), mshadow::default_type_flag);
    RunContext rctx{Context::CPU(), nullptr, nullptr, false};
    if (header.flag > 0) {
      auto label_shape = header.flag <= 1 ? TShape(0, 1) : TShape({header.flag});
      label.ReshapeAndAlloc(label_shape);
      TBlob dst = label.data();
      mxnet::ndarray::Copy<cpu, cpu>(
        TBlob(reinterpret_cast<void*>(s), label.shape(), cpu::kDevMask, label.dtype(), 0),
        &dst, Context::CPU(), Context::CPU(), rctx);
      s += sizeof(float) * header.flag;
      size -= sizeof(float) * header.flag;
    } else {
      // label is a scalar with ndim() == 0
      label.ReshapeAndAlloc(TShape(0, 1));
      TBlob dst = label.data();
      *(dst.dptr<float>()) = header.label;
    }
    ret->resize(2);
    (*ret)[1] = label;
#if MXNET_USE_OPENCV
    cv::Mat buf(1, size, CV_8U, s);
    cv::Mat res = cv::imdecode(buf, param_.flag);
    CHECK(!res.empty()) << "Decoding failed. Invalid image file.";
    const int n_channels = res.channels();
    if (n_channels == 1) {
      SwapImageChannels<1>(res, &(ret->at(0)));
    } else if (n_channels == 3) {
      SwapImageChannels<3>(res, &(ret->at(0)));
    } else if (n_channels == 4) {
      SwapImageChannels<4>(res, &(ret->at(0)));
    }
    return true;
#else
  LOG(FATAL) << "Opencv is needed for image decoding.";
#endif
  return false;  // should not reach here
  }

 private:
  /*! \brief parameters */
  ImageRecordFileDatasetParam param_;
  /*! \brief base recordIO reader */
  std::shared_ptr<RecordFileDataset> base_;
};

MXNET_REGISTER_IO_DATASET(ImageRecordFileDataset)
  .describe("MXNet Image Record File Dataset")
  .add_arguments(ImageRecordFileDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new ImageRecordFileDataset(kwargs);
});

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

class ImageSequenceDataset final : public Dataset {
 public:
  explicit ImageSequenceDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    param_.InitAllowUnknown(kwargs);
    img_list_ = dmlc::Split(param_.img_list, param_.path_sep);
  }

  uint64_t GetLen() const override {
    return img_list_.size();
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* ret) override {
#if MXNET_USE_OPENCV
    CHECK_LT(idx, img_list_.size())
      << "GetItem index: " << idx << " out of bound: " << img_list_.size();
    cv::Mat res = cv::imread(img_list_[idx], param_.flag);
    CHECK(!res.empty()) << "Decoding failed. Invalid image file.";
    const int n_channels = res.channels();
    ret->resize(1);
    if (n_channels == 1) {
      SwapImageChannels<1>(res, &(ret->at(0)));
    } else if (n_channels == 3) {
      SwapImageChannels<3>(res, &(ret->at(0)));
    } else if (n_channels == 4) {
      SwapImageChannels<4>(res, &(ret->at(0)));
    }
    return true;
#else
  LOG(FATAL) << "Opencv is needed for image decoding.";
#endif
  return false;
  }

 private:
  /*! \brief parameters */
  ImageSequenceDatasetParam param_;
  /*! \brief image list */
  std::vector<std::string> img_list_;
};

MXNET_REGISTER_IO_DATASET(ImageSequenceDataset)
  .describe("Image Sequence Dataset")
  .add_arguments(ImageSequenceDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new ImageSequenceDataset(kwargs);
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

class NDArrayDataset final : public Dataset {
 public:
  explicit NDArrayDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    data_ = *(static_cast<NDArray*>(reinterpret_cast<void*>(param_.arr)));
    if (data_.shape().ndim() < 1) {
      LOG(FATAL) << "NDArray with no dim is not iterable";
    }
    size_ = data_.shape().begin()[0];
  }

  uint64_t GetLen() const override {
    return size_;
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* rets) override {
    CHECK_LT(idx, size_)
      << "GetItem index: " << idx << " out of bound: " << size_;
    rets->resize(1);
    auto& ret = (*rets)[0];
    ret = data_.Slice(idx, idx + 1);
    if (ret.shape().ndim() > 1) {
      // remove first dim to be consistent with numpy
      TShape new_shape;
      new_shape.assign(ret.shape().begin() + 1, ret.shape().end());
      ret = ret.Reshape(new_shape);
    } else {
      if (data_.shape().ndim() == 1) {
        // scalar
        TShape new_shape(0, 1);
        ret = ret.Reshape(new_shape);
      }
    }
    return true;
  }

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
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new NDArrayDataset(kwargs);
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

class GroupDataset final : public Dataset {
 public:
  explicit GroupDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    param_.InitAllowUnknown(kwargs);
    auto childs = param_.datasets;
    childs_.reserve(childs.ndim());
    size_t child_cnt = 0;
    for (auto child : childs) {
      auto d = *static_cast<std::shared_ptr<Dataset>*>(reinterpret_cast<void*>(child));
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

  uint64_t GetLen() const override {
    return size_;
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* ret) override {
    CHECK_LT(idx, size_)
      << "GetItem index: " << idx << " out of bound: " << size_;
    ret->clear();
    for (const auto& child : childs_) {
      std::vector<NDArray> temp_ret;
      if (!child->GetItem(idx, &temp_ret)) return false;
      ret->insert(ret->end(), temp_ret.begin(), temp_ret.end());
    }
    return true;
  }

 private:
  /*! \brief parameters */
  GroupDatasetParam param_;
  /*! \brief stored child datasets */
  std::vector<std::shared_ptr<Dataset>> childs_;
  /*! \brief overall dataset size, equals to all child datasets */
  uint64_t size_;
};   // class GroupDataset

MXNET_REGISTER_IO_DATASET(GroupDataset)
  .describe("Grouped Dataset that combine a bunch of datasets")
  .add_arguments(GroupDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new GroupDataset(kwargs);
});

struct IndexedDatasetParam : public dmlc::Parameter<IndexedDatasetParam> {
  /*! \brief the base dataset */
  std::intptr_t base;
  /*! \brief the indices */
  Tuple<uint64_t> indices;
  // declare parameters
  DMLC_DECLARE_PARAMETER(IndexedDatasetParam) {
      DMLC_DECLARE_FIELD(base)
          .describe("Pointer to the internal c++ dataset that is going to be indexed.");
      DMLC_DECLARE_FIELD(indices)
          .describe("The indices for the internal dataset. Output[i] will be base[indices[i]].");
  }
};  // struct IndexedDatasetParam

DMLC_REGISTER_PARAMETER(IndexedDatasetParam);

class IndexedDataset final : public Dataset {
 public:
  explicit IndexedDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    base_data_ = *static_cast<std::shared_ptr<Dataset>*>(reinterpret_cast<void*>(param_.base));
  }

  uint64_t GetLen() const override {
    return param_.indices.ndim();
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* ret) override {
    CHECK_GT(param_.indices.ndim(), idx) << "IndexError: " << idx
      << " from total: " << param_.indices.ndim();
    auto new_idx = param_.indices[idx];
    CHECK_GT(base_data_->GetLen(), new_idx) << "IndexError: " << new_idx
      << " from original dataset with size: " << base_data_->GetLen();
    return base_data_->GetItem(new_idx, ret);
  }

 private:
  /*! \brief parameters */
  IndexedDatasetParam param_;
  /*! \brief stored child dataset */
  std::shared_ptr<Dataset> base_data_;
};   // class IndexedDataset

MXNET_REGISTER_IO_DATASET(IndexedDataset)
  .describe("Grouped Dataset that combine a bunch of datasets")
  .add_arguments(IndexedDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new IndexedDataset(kwargs);
});

struct LazyTransformDatasetParam : public dmlc::Parameter<LazyTransformDatasetParam> {
  /*! \brief the source ndarray */
  std::intptr_t cached_op;
  /*! \brief internal dataset */
  std::intptr_t dataset;
  /*! \brief indices for items that needs transformation */
  Tuple<int> transform_indices;
  /*! \brief is_scalar information for outputs */
  Tuple<int> scalar_outputs;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LazyTransformDatasetParam) {
      DMLC_DECLARE_FIELD(cached_op)
          .describe("Pointer to cached transform function.");
      DMLC_DECLARE_FIELD(dataset)
          .describe("Pointer to internal dataset.");
      DMLC_DECLARE_FIELD(transform_indices).set_default(Tuple<int>({}))
          .describe("The indices for dataset items that need to be transformed/processed. "
                    "If `transform_indices` is empty(default), "
                    "then all items will be processed.");
      DMLC_DECLARE_FIELD(scalar_outputs)
          .describe("Indicate whether outputs are scalars, the size must match the output size.");
  }
};  // struct LazyTransformDatasetParam

DMLC_REGISTER_PARAMETER(LazyTransformDatasetParam);

class LazyTransformDataset final : public Dataset {
 public:
  LazyTransformDataset(const LazyTransformDataset& other) {
    this->param_ = other.param_;
    this->pass_through_indices_ = other.pass_through_indices_;
    this->use_input_indices_ = other.use_input_indices_;
    this->num_outputs_ = other.num_outputs_;
    this->cached_op_ = std::make_shared<NaiveCachedOp>(
      other.cached_op_->sym_, other.cached_op_->flags_);
    this->base_data_ = other.base_data_;
  }

  explicit LazyTransformDataset(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    auto op = *static_cast<CachedOpPtr*>(reinterpret_cast<void*>(param_.cached_op));
    cached_op_ = std::make_shared<NaiveCachedOp>(op->sym_, op->flags_);
    base_data_ = *static_cast<std::shared_ptr<Dataset>*>(reinterpret_cast<void*>(param_.dataset));

    // use first item to calculate size info
    CHECK_GT(GetLen(), 0)
      << "LazyTransformDataset expect the base dataset to have at least 1 item";
    std::vector<NDArray> inputs;
    CHECK(base_data_->GetItem(0, &inputs));
    // check output size
    CHECK_EQ(param_.scalar_outputs.ndim(), cached_op_->num_outputs())
      << "Output scalar info size: " << param_.scalar_outputs.ndim() << " vs. output size: "
      << cached_op_->num_outputs() << " mismatch!";
    // check input size
    if (param_.transform_indices.ndim() == 0) {
      std::vector<int> default_indices;
      default_indices.reserve(cached_op_->num_inputs());
      for (size_t i = 0; i < cached_op_->num_inputs(); ++i) {
        default_indices.emplace_back(static_cast<int>(i));
      }
      use_input_indices_ = default_indices;
    } else {
      use_input_indices_ = std::vector<int>(param_.transform_indices.begin(),
                                            param_.transform_indices.end());
    }
    CHECK_EQ(use_input_indices_.size(), cached_op_->num_inputs())
      << "Mismatched transform indices and transform inputs: " << use_input_indices_.size()
      << " vs. " << cached_op_->num_inputs();
    auto num_inputs = use_input_indices_.size();
    CHECK_GE(inputs.size(), num_inputs)
      << "LazyTransformDataset input size " << inputs.size()
      << " smaller than transform input size: "
      << num_inputs;
    pass_through_indices_.clear();
    for (size_t i = 0; i < inputs.size(); ++i) {
      // filling output ndarray from unaltered inputs, transformed outputs are already inserted
      if (std::find(use_input_indices_.begin(),
                    use_input_indices_.end(), i) == use_input_indices_.end()) {
        pass_through_indices_.emplace_back(i);
      }
    }
    num_outputs_ = inputs.size() + cached_op_->num_outputs() - cached_op_->num_inputs();
  }

  ~LazyTransformDataset() override = default;

  uint64_t GetLen() const override {
    return base_data_->GetLen();
  }

  bool GetItem(uint64_t idx, std::vector<NDArray>* outputs) override {
    std::vector<NDArray> inputs;
    if (!base_data_->GetItem(idx, &inputs)) return false;
    outputs->reserve(num_outputs_);
    outputs->resize(cached_op_->num_outputs());
    for (auto i : pass_through_indices_) {
      outputs->emplace_back(inputs[i]);
    }
    CHECK_EQ(outputs->size(), num_outputs_);
    // workspace for cached op
    std::vector<NDArray*> ndinputs;
    std::vector<NDArray*> ndoutputs;
    ndinputs.reserve(inputs.size());
    for (int use_input_indice : use_input_indices_) {
      ndinputs.emplace_back(&(inputs[use_input_indice]));
    }
    ndoutputs.reserve(cached_op_->num_outputs());
    CHECK_LE(cached_op_->num_outputs(), outputs->size());
    for (size_t i = 0; i < cached_op_->num_outputs(); ++i) {
      ndoutputs.emplace_back(&(outputs->at(i)));
    }

    for (auto & input : inputs) {
      input.WaitToRead();
    }
    CHECK(inputs.size() > 0) << "dataset getitem requires at least one input";
    Context default_ctx = inputs[0].ctx();
    cached_op_->Forward(cached_op_, ndinputs, ndoutputs, default_ctx);
    return true;
  }

 private:
  /*! \brief parameters */
  LazyTransformDatasetParam param_;
  /*! \brief stored cached op */
  NaiveCachedOpPtr cached_op_;
  /*! \brief internal dataset */
  std::shared_ptr<Dataset> base_data_;
  std::vector<int> use_input_indices_;
  std::vector<int> pass_through_indices_;
  size_t num_outputs_;
};   // class LazyTransformDataset

MXNET_REGISTER_IO_DATASET(LazyTransformDataset)
  .describe("Dataset that apply lazy transformation to internal dataset")
  .add_arguments(LazyTransformDatasetParam::__FIELDS__())
  .set_body([](const std::vector<std::pair<std::string, std::string> >& kwargs) {
     return new LazyTransformDataset(kwargs);
});
}  // namespace io
}  // namespace mxnet
