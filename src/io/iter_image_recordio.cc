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
 * \file iter_image_recordio-inl.hpp
 * \brief recordio data iterator
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/common.h>
#include <dmlc/input_split_shuffle.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/recordio.h>
#include <dmlc/threadediter.h>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include "./image_iter_common.h"
#include "./inst_vector.h"
#include "./image_recordio.h"
#include "./image_augmenter.h"
#include "./iter_prefetcher.h"
#include "./iter_normalize.h"
#include "./iter_batchloader.h"

namespace mxnet {
namespace io {
// parser to parse image recordio
template<typename DType>
class ImageRecordIOParser {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);

  // set record to the head
  inline void BeforeFirst(void) {
    return source_->BeforeFirst();
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(std::vector<InstVector<DType>> *out);

 private:
  // magic number to see prng
  static const int kRandMagic = 111;
  /*! \brief parameters */
  ImageRecParserParam param_;
  #if MXNET_USE_OPENCV
  /*! \brief augmenters */
  std::vector<std::vector<std::unique_ptr<ImageAugmenter> > > augmenters_;
  #endif
  /*! \brief random samplers */
  std::vector<std::unique_ptr<common::RANDOM_ENGINE> > prnds_;
  /*! \brief data source */
  std::unique_ptr<dmlc::InputSplit> source_;
  /*! \brief label information, if any */
  std::unique_ptr<ImageLabelMap> label_map_;
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
};

template<typename DType>
inline void ImageRecordIOParser<DType>::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
#if MXNET_USE_OPENCV
  // initialize parameter
  // init image rec param
  param_.InitAllowUnknown(kwargs);
  int maxthread, threadget;
  #pragma omp parallel
  {
    // be conservative, set number of real cores
    maxthread = std::max(omp_get_num_procs() / 2 - 1, 1);
  }
  param_.preprocess_threads = std::min(maxthread, param_.preprocess_threads);
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    threadget = omp_get_num_threads();
  }
  param_.preprocess_threads = threadget;

  std::vector<std::string> aug_names = dmlc::Split(param_.aug_seq, ',');
  augmenters_.clear();
  augmenters_.resize(threadget);
  // setup decoders
  for (int i = 0; i < threadget; ++i) {
    for (const auto& aug_name : aug_names) {
      augmenters_[i].emplace_back(ImageAugmenter::Create(aug_name));
      augmenters_[i].back()->Init(kwargs);
    }
    prnds_.emplace_back(new common::RANDOM_ENGINE((i + 1) * kRandMagic));
  }
  if (param_.path_imglist.length() != 0) {
    label_map_.reset(new ImageLabelMap(param_.path_imglist.c_str(),
      param_.label_width, !param_.verbose));
  }
  CHECK(param_.path_imgrec.length() != 0)
      << "ImageRecordIOIterator: must specify image_rec";

  if (param_.verbose) {
    LOG(INFO) << "ImageRecordIOParser: " << param_.path_imgrec
              << ", use " << threadget << " threads for decoding..";
  }
  source_.reset(dmlc::InputSplit::Create(
      param_.path_imgrec.c_str(), param_.part_index,
      param_.num_parts, "recordio"));
  if (param_.shuffle_chunk_size > 0) {
    if (param_.shuffle_chunk_size > 4096) {
      LOG(INFO) << "Chunk size: " << param_.shuffle_chunk_size
                 << " MB which is larger than 4096 MB, please set "
                    "smaller chunk size";
    }
    if (param_.shuffle_chunk_size < 4) {
      LOG(INFO) << "Chunk size: " << param_.shuffle_chunk_size
                 << " MB which is less than 4 MB, please set "
                    "larger chunk size";
    }
    // 1.1 ratio is for a bit more shuffle parts to avoid boundary issue
    unsigned num_shuffle_parts =
        std::ceil(source_->GetTotalSize() * 1.1 /
                  (param_.num_parts * (param_.shuffle_chunk_size << 20UL)));

    if (num_shuffle_parts > 1) {
      source_.reset(dmlc::InputSplitShuffle::Create(
          param_.path_imgrec.c_str(), param_.part_index,
          param_.num_parts, "recordio", num_shuffle_parts, param_.shuffle_chunk_seed));
    }
    source_->HintChunkSize(param_.shuffle_chunk_size << 17UL);
  } else {
    // use 64 MB chunk when possible
    source_->HintChunkSize(8 << 20UL);
  }
#else
  LOG(FATAL) << "ImageRec need opencv to process";
#endif
}

template<typename DType>
inline bool ImageRecordIOParser<DType>::
ParseNext(std::vector<InstVector<DType>> *out_vec) {
  CHECK(source_ != nullptr);
  dmlc::InputSplit::Blob chunk;
  if (!source_->NextChunk(&chunk)) return false;
#if MXNET_USE_OPENCV
  // save opencv out
  out_vec->resize(param_.preprocess_threads);
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    CHECK(omp_get_num_threads() == param_.preprocess_threads);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(chunk, tid, param_.preprocess_threads);
    ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector<DType> &out = (*out_vec)[tid];
    out.Clear();
    while (reader.NextRecord(&blob)) {
      // Opencv decode and augments
      cv::Mat res;
      rec.Load(blob.dptr, blob.size);
      cv::Mat buf(1, rec.content_size, CV_8U, rec.content);
      switch (param_.data_shape[0]) {
       case 1:
        res = cv::imdecode(buf, 0);
        break;
       case 3:
        res = cv::imdecode(buf, 1);
        break;
       case 4:
        // -1 to keep the number of channel of the encoded image, and not force gray or color.
        res = cv::imdecode(buf, -1);
        CHECK_EQ(res.channels(), 4)
          << "Invalid image with index " << rec.image_index()
          << ". Expected 4 channels, got " << res.channels();
        break;
       default:
        LOG(FATAL) << "Invalid output shape " << param_.data_shape;
      }
      const int n_channels = res.channels();
      for (auto& aug : augmenters_[tid]) {
        res = aug->Process(res, nullptr, prnds_[tid].get());
      }
      out.Push(static_cast<unsigned>(rec.image_index()),
               mshadow::Shape3(n_channels, res.rows, res.cols),
               mshadow::Shape1(param_.label_width));

      mshadow::Tensor<cpu, 3, DType> data = out.data().Back();

      // For RGB or RGBA data, swap the B and R channel:
      // OpenCV store as BGR (or BGRA) and we want RGB (or RGBA)
      std::vector<int> swap_indices;
      if (n_channels == 1) swap_indices = {0};
      if (n_channels == 3) swap_indices = {2, 1, 0};
      if (n_channels == 4) swap_indices = {2, 1, 0, 3};

      for (int i = 0; i < res.rows; ++i) {
        uchar* im_data = res.ptr<uchar>(i);
        for (int j = 0; j < res.cols; ++j) {
          for (int k = 0; k < n_channels; ++k) {
              data[k][i][j] = im_data[swap_indices[k]];
          }
          im_data += n_channels;
        }
      }

      mshadow::Tensor<cpu, 1> label = out.label().Back();
      if (label_map_ != nullptr) {
        mshadow::Copy(label, label_map_->Find(rec.image_index()));
      } else if (rec.label != NULL) {
        CHECK_EQ(param_.label_width, rec.num_label)
          << "rec file provide " << rec.num_label << "-dimensional label "
             "but label_width is set to " << param_.label_width;
        mshadow::Copy(label, mshadow::Tensor<cpu, 1>(rec.label,
                                                     mshadow::Shape1(rec.num_label)));
      } else {
        CHECK_EQ(param_.label_width, 1)
          << "label_width must be 1 unless an imglist is provided "
             "or the rec file is packed with multi dimensional label";
        label[0] = rec.header.label;
      }
      res.release();
    }
  }
#else
  LOG(FATAL) << "Opencv is needed for image decoding and augmenting.";
#endif  // MXNET_USE_OPENCV
  return true;
}

// iterator on image recordio
template<typename DType = real_t>
class ImageRecordIter : public IIterator<DataInst> {
 public:
  ImageRecordIter() : data_(nullptr) { }
  // destructor
  virtual ~ImageRecordIter(void) {
    iter_.Destroy();
    delete data_;
  }
  // constructor
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    // use the kwarg to init parser
    parser_.Init(kwargs);
    // prefetch at most 4 minbatches
    iter_.set_max_capacity(4);
    // init thread iter
    iter_.Init([this](std::vector<InstVector<DType>> **dptr) {
        if (*dptr == nullptr) {
          *dptr = new std::vector<InstVector<DType>>();
        }
        return parser_.ParseNext(*dptr);
      },
      [this]() { parser_.BeforeFirst(); });
    inst_ptr_ = 0;
    rnd_.seed(kRandMagic + param_.seed);
  }
  // before first
  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
    inst_order_.clear();
    inst_ptr_ = 0;
  }

  virtual bool Next(void) {
    while (true) {
      if (inst_ptr_ < inst_order_.size()) {
        std::pair<unsigned, unsigned> p = inst_order_[inst_ptr_];
        out_ = (*data_)[p.first][p.second];
        ++inst_ptr_;
        return true;
      } else {
        if (data_ != nullptr) iter_.Recycle(&data_);
        if (!iter_.Next(&data_)) return false;
        inst_order_.clear();
        for (unsigned i = 0; i < data_->size(); ++i) {
          const InstVector<DType>& tmp = (*data_)[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
        }
        // shuffle instance order if needed
        if (param_.shuffle != 0) {
          std::shuffle(inst_order_.begin(), inst_order_.end(), rnd_);
        }
        inst_ptr_ = 0;
      }
    }
    return false;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  // random magic
  static const int kRandMagic = 111;
  // output instance
  DataInst out_;
  // data ptr
  size_t inst_ptr_;
  // internal instance order
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  // data
  std::vector<InstVector<DType>> *data_;
  // internal parser
  ImageRecordIOParser<DType> parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVector<DType>> > iter_;
  // parameters
  ImageRecordParam param_;
  // random number generator
  common::RANDOM_ENGINE rnd_;
};

// OLD VERSION - DEPRECATED
MXNET_REGISTER_IO_ITER(ImageRecordIter_v1)
.describe(R"code(Iterating on image RecordIO files

.. note::

  ``ImageRecordIter_v1`` is deprecated. Use ``ImageRecordIter`` instead.


Read images batches from RecordIO files with a rich of data augmentation
options.

One can use ``tools/im2rec.py`` to pack individual image files into RecordIO
files.

)code" ADD_FILELINE)
.add_arguments(ImageRecParserParam::__FIELDS__())
.add_arguments(ImageRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.add_arguments(ImageNormalizeParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new ImageNormalizeIter(
                new ImageRecordIter<real_t>())));
  });

// OLD VERSION - DEPRECATED
MXNET_REGISTER_IO_ITER(ImageRecordUInt8Iter_v1)
.describe(R"code(Iterating on image RecordIO files

.. note::

  ``ImageRecordUInt8Iter_v1`` is deprecated. Use ``ImageRecordUInt8Iter`` instead.

This iterator is identical to ``ImageRecordIter`` except for using ``uint8`` as
the data type instead of ``float``.

)code" ADD_FILELINE)
.add_arguments(ImageRecParserParam::__FIELDS__())
.add_arguments(ImageRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new ImageRecordIter<uint8_t>()));
  });
}  // namespace io
}  // namespace mxnet
