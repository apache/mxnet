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
 * \file iter_image_recordio_2.cc
 * \brief new version of recordio data iterator
 */

#include <mxnet/io.h>
#include <dmlc/parameter.h>
#include <dmlc/threadediter.h>
#include <dmlc/input_split_shuffle.h>
#include <dmlc/recordio.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/common.h>
#include <dmlc/timer.h>
#include <type_traits>
#if MXNET_USE_LIBJPEG_TURBO
#include <turbojpeg.h>
#endif
#include "./image_recordio.h"
#include "./image_augmenter.h"
#include "./image_iter_common.h"
#include "./inst_vector.h"
#include "../common/utils.h"

namespace mxnet {
namespace io {
// parser to parse image recordio
template<typename DType>
class ImageRecordIOParser2 {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);

  // set record to the head
  inline void BeforeFirst(void) {
    if (batch_param_.round_batch == 0 || !overflow) {
      n_parsed_ = 0;
      return source_->BeforeFirst();
    } else {
      overflow = false;
    }
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(DataBatch *out);

 private:
#if MXNET_USE_OPENCV
  template<int n_channels>
  void ProcessImage(const cv::Mat& res,
    mshadow::Tensor<cpu, 3, DType>* data_ptr, const bool is_mirrored, const float contrast_scaled,
    const float illumination_scaled);
#if MXNET_USE_LIBJPEG_TURBO
  cv::Mat TJimdecode(cv::Mat buf, int color);
#endif
#endif
  inline unsigned ParseChunk(DType* data_dptr, real_t* label_dptr, const unsigned current_size,
    dmlc::InputSplit::Blob * chunk);
  inline void CreateMeanImg(void);

  // magic number to seed prng
  static const int kRandMagic = 111;
  static const int kRandMagicNormalize = 0;
  /*! \brief parameters */
  ImageRecParserParam param_;
  ImageRecordParam record_param_;
  BatchParam batch_param_;
  ImageNormalizeParam normalize_param_;
  PrefetcherParam prefetch_param_;
  #if MXNET_USE_OPENCV
  /*! \brief augmenters */
  std::vector<std::vector<std::unique_ptr<ImageAugmenter> > > augmenters_;
  #endif
  /*! \brief random samplers */
  std::vector<std::unique_ptr<common::RANDOM_ENGINE> > prnds_;
  common::RANDOM_ENGINE rnd_;
  /*! \brief data source */
  std::unique_ptr<dmlc::InputSplit> source_;
  /*! \brief label information, if any */
  std::unique_ptr<ImageLabelMap> label_map_;
  /*! \brief temporary results */
  std::vector<InstVector<DType>> temp_;
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
  /*! \brief internal instance order */
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  unsigned inst_index_;
  /*! \brief internal counter tracking number of already parsed entries */
  unsigned n_parsed_;
  /*! \brief overflow marker */
  bool overflow;
  /*! \brief unit size */
  std::vector<size_t> unit_size_;
  /*! \brief mean image, if needed */
  mshadow::TensorContainer<cpu, 3> meanimg_;
  // whether to use legacy shuffle
  // (without IndexedRecordIO support)
  bool legacy_shuffle_;
  // whether mean image is ready.
  bool meanfile_ready_;
};

template<typename DType>
inline void ImageRecordIOParser2<DType>::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
#if MXNET_USE_OPENCV
  // initialize parameter
  // init image rec param
  param_.InitAllowUnknown(kwargs);
  record_param_.InitAllowUnknown(kwargs);
  batch_param_.InitAllowUnknown(kwargs);
  normalize_param_.InitAllowUnknown(kwargs);
  prefetch_param_.InitAllowUnknown(kwargs);
  n_parsed_ = 0;
  overflow = false;
  rnd_.seed(kRandMagic + record_param_.seed);
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
      << "ImageRecordIter2: must specify image_rec";

  if (param_.verbose) {
    LOG(INFO) << "ImageRecordIOParser2: " << param_.path_imgrec
              << ", use " << threadget << " threads for decoding..";
  }
  legacy_shuffle_ = false;
  if (param_.path_imgidx.length() != 0) {
    source_.reset(dmlc::InputSplit::Create(
        param_.path_imgrec.c_str(),
        param_.path_imgidx.c_str(),
        param_.part_index,
        param_.num_parts, "indexed_recordio",
        record_param_.shuffle,
        record_param_.seed,
        batch_param_.batch_size));
  } else {
    source_.reset(dmlc::InputSplit::Create(
        param_.path_imgrec.c_str(), param_.part_index,
        param_.num_parts, "recordio"));
    if (record_param_.shuffle)
      legacy_shuffle_ = true;
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
      source_->HintChunkSize(64 << 20UL);
    }
  }
  // Normalize init
  if (!std::is_same<DType, uint8_t>::value) {
    meanimg_.set_pad(false);
    meanfile_ready_ = false;
    if (normalize_param_.mean_img.length() != 0) {
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(normalize_param_.mean_img.c_str(), "r", true));
      if (fi.get() == nullptr) {
        this->CreateMeanImg();
      } else {
        fi.reset(nullptr);
        if (param_.verbose) {
          LOG(INFO) << "Load mean image from " << normalize_param_.mean_img;
        }
        // use python compatible ndarray store format
        std::vector<NDArray> data;
        std::vector<std::string> keys;
        {
          std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(normalize_param_.mean_img.c_str(),
                                                                "r"));
          NDArray::Load(fi.get(), &data, &keys);
        }
        CHECK_EQ(data.size(), 1)
          << "Invalid mean image file format";
        data[0].WaitToRead();
        mshadow::Tensor<cpu, 3> src = data[0].data().get<cpu, 3, real_t>();
        meanimg_.Resize(src.shape_);
        mshadow::Copy(meanimg_, src);
        meanfile_ready_ = true;
        if (param_.verbose) {
          LOG(INFO) << "Load mean image from " << normalize_param_.mean_img << " completed";
        }
      }
    }
  }
#else
  LOG(FATAL) << "ImageRec need opencv to process";
#endif
}

template<typename DType>
inline bool ImageRecordIOParser2<DType>::ParseNext(DataBatch *out) {
  if (overflow) {
    return false;
  }
  CHECK(source_ != nullptr);
  dmlc::InputSplit::Blob chunk;
  unsigned current_size = 0;
  out->index.resize(batch_param_.batch_size);

  // InitBatch
  if (out->data.size() == 0) {
    // This assumes that DataInst given by
    // InstVector contains only 2 elements in
    // data vector (operator[] implementation)
    out->data.resize(2);
    unit_size_.resize(2);

    std::vector<index_t> shape_vec;
    shape_vec.push_back(batch_param_.batch_size);
    for (index_t dim = 0; dim < param_.data_shape.ndim(); ++dim) {
      shape_vec.push_back(param_.data_shape[dim]);
    }
    TShape data_shape(shape_vec.begin(), shape_vec.end());

    shape_vec.clear();
    shape_vec.push_back(batch_param_.batch_size);
    shape_vec.push_back(param_.label_width);
    TShape label_shape(shape_vec.begin(), shape_vec.end());

    out->data.at(0) = NDArray(data_shape, Context::CPUPinned(0), false,
      mshadow::DataType<DType>::kFlag);
    out->data.at(1) = NDArray(label_shape, Context::CPUPinned(0), false,
      mshadow::DataType<real_t>::kFlag);
    unit_size_[0] = param_.data_shape.Size();
    unit_size_[1] = param_.label_width;
  }

  while (current_size < batch_param_.batch_size) {
    // int n_to_copy;
    unsigned n_to_out = 0;
    if (n_parsed_ == 0) {
      if (source_->NextBatch(&chunk, batch_param_.batch_size)) {
        inst_order_.clear();
        inst_index_ = 0;
        DType* data_dptr = static_cast<DType*>(out->data[0].data().dptr_);
        real_t* label_dptr = static_cast<real_t*>(out->data[1].data().dptr_);
        if (!legacy_shuffle_) {
          n_to_out = ParseChunk(data_dptr, label_dptr, current_size, &chunk);
        } else {
          n_to_out = ParseChunk(NULL, NULL, batch_param_.batch_size, &chunk);
        }
        // Count number of parsed images that do not fit into current out
        n_parsed_ = inst_order_.size();
        // shuffle instance order if needed
        if (legacy_shuffle_) {
          std::shuffle(inst_order_.begin(), inst_order_.end(), rnd_);
        }
      } else {
        if (current_size == 0) {
          return false;
        }
        CHECK(!overflow) << "number of input images must be bigger than the batch size";
        if (batch_param_.round_batch != 0) {
          overflow = true;
          source_->BeforeFirst();
        } else {
          current_size = batch_param_.batch_size;
        }
        out->num_batch_padd = batch_param_.batch_size - current_size;
        n_to_out = 0;
      }
    } else {
      int n_to_copy = std::min(n_parsed_, batch_param_.batch_size - current_size);
      n_parsed_ -= n_to_copy;
      // Copy
      #pragma omp parallel for num_threads(param_.preprocess_threads)
      for (int i = 0; i < n_to_copy; ++i) {
        std::pair<unsigned, unsigned> place = inst_order_[inst_index_ + i];
        const DataInst& batch = temp_[place.first][place.second];
        for (unsigned j = 0; j < batch.data.size(); ++j) {
          CHECK_EQ(unit_size_[j], batch.data[j].Size());
          MSHADOW_TYPE_SWITCH(out->data[j].data().type_flag_, dtype, {
          mshadow::Copy(
              out->data[j].data().FlatTo1D<cpu, dtype>().Slice((current_size + i) * unit_size_[j],
                (current_size + i + 1) * unit_size_[j]),
              batch.data[j].get_with_shape<cpu, 1, dtype>(mshadow::Shape1(unit_size_[j])));
          });
        }
      }
      n_to_out = n_to_copy;
      inst_index_ += n_to_copy;
    }

    current_size += n_to_out;
  }
  return true;
}

#if MXNET_USE_OPENCV
template<typename DType>
template<int n_channels>
void ImageRecordIOParser2<DType>::ProcessImage(const cv::Mat& res,
  mshadow::Tensor<cpu, 3, DType>* data_ptr, const bool is_mirrored, const float contrast_scaled,
  const float illumination_scaled) {
  float RGBA_MULT[4] = { 0 };
  float RGBA_BIAS[4] = { 0 };
  float RGBA_MEAN[4] = { 0 };
  mshadow::Tensor<cpu, 3, DType>& data = (*data_ptr);
  if (!std::is_same<DType, uint8_t>::value) {
    RGBA_MULT[0] = contrast_scaled / normalize_param_.std_r;
    RGBA_MULT[1] = contrast_scaled / normalize_param_.std_g;
    RGBA_MULT[2] = contrast_scaled / normalize_param_.std_b;
    RGBA_MULT[3] = contrast_scaled / normalize_param_.std_a;
    RGBA_BIAS[0] = illumination_scaled / normalize_param_.std_r;
    RGBA_BIAS[1] = illumination_scaled / normalize_param_.std_g;
    RGBA_BIAS[2] = illumination_scaled / normalize_param_.std_b;
    RGBA_BIAS[3] = illumination_scaled / normalize_param_.std_a;
    if (!meanfile_ready_) {
      RGBA_MEAN[0] = normalize_param_.mean_r;
      RGBA_MEAN[1] = normalize_param_.mean_g;
      RGBA_MEAN[2] = normalize_param_.mean_b;
      RGBA_MEAN[3] = normalize_param_.mean_a;
    }
  }

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

  DType RGBA[n_channels] = {};
  for (int i = 0; i < res.rows; ++i) {
    const uchar* im_data = res.ptr<uchar>(i);
    for (int j = 0; j < res.cols; ++j) {
      for (int k = 0; k < n_channels; ++k) {
        RGBA[k] = im_data[swap_indices[k]];
      }
      if (!std::is_same<DType, uint8_t>::value) {
        // normalize/mirror here to avoid memory copies
        // logic from iter_normalize.h, function SetOutImg
        for (int k = 0; k < n_channels; ++k) {
          if (meanfile_ready_) {
            RGBA[k] = (RGBA[k] - meanimg_[k][i][j]) * RGBA_MULT[k] + RGBA_BIAS[k];
          } else {
            RGBA[k] = (RGBA[k] - RGBA_MEAN[k]) * RGBA_MULT[k] + RGBA_BIAS[k];
          }
        }
      }
      for (int k = 0; k < n_channels; ++k) {
        // mirror here to avoid memory copies
        // logic from iter_normalize.h, function SetOutImg
        if (is_mirrored) {
          data[k][i][res.cols - j - 1] = RGBA[k];
        } else {
          data[k][i][j] = RGBA[k];
        }
      }
      im_data += n_channels;
    }
  }
}

#if MXNET_USE_LIBJPEG_TURBO

bool is_jpeg(unsigned char * file) {
  if ((file[0] == 255) && (file[1] == 216)) {
    return true;
  } else {
    return false;
  }
}

template<typename DType>
cv::Mat ImageRecordIOParser2<DType>::TJimdecode(cv::Mat image, int color) {
  unsigned char* jpeg = image.ptr();
  size_t jpeg_size = image.rows * image.cols;

  if (!is_jpeg(jpeg)) {
    // If it is not JPEG then fall back to OpenCV
    return cv::imdecode(image, color);
  }

  tjhandle handle = tjInitDecompress();
  int h, w, subsamp;
  int err = tjDecompressHeader2(handle,
                                jpeg,
                                jpeg_size,
                                &w, &h, &subsamp);
  if (err != 0) {
    // If it is a malformed JPEG then fall back to OpenCV
    return cv::imdecode(image, color);
  }
  cv::Mat ret = cv::Mat(h, w, color ? CV_8UC3 : CV_8UC1);
  err = tjDecompress2(handle,
                      jpeg,
                      jpeg_size,
                      ret.ptr(),
                      w,
                      0,
                      h,
                      color ? TJPF_BGR : TJPF_GRAY,
                      0);
  if (err != 0) {
    // If it is a malformed JPEG then fall back to OpenCV
    return cv::imdecode(image, color);
  }
  tjDestroy(handle);
  return ret;
}
#endif
#endif

// Returns the number of images that are put into output
template<typename DType>
inline unsigned ImageRecordIOParser2<DType>::ParseChunk(DType* data_dptr, real_t* label_dptr,
  const unsigned current_size, dmlc::InputSplit::Blob * chunk) {
  temp_.resize(param_.preprocess_threads);
#if MXNET_USE_OPENCV
  // save opencv out
  dmlc::RecordIOChunkReader reader(*chunk, 0, 1);
  unsigned gl_idx = current_size;
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    CHECK(omp_get_num_threads() == param_.preprocess_threads);
    unsigned int tid = omp_get_thread_num();
    // dmlc::RecordIOChunkReader reader(*chunk, tid, param_.preprocess_threads);
    ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector<DType> &out_tmp = temp_[tid];
    out_tmp.Clear();
    while (true) {
      bool reader_has_data;
      unsigned idx;
      #pragma omp critical
      {
        reader_has_data = reader.NextRecord(&blob);
        if (reader_has_data) {
          idx = gl_idx++;
          if (idx >= batch_param_.batch_size) {
            inst_order_.push_back(std::make_pair(tid, out_tmp.Size()));
          }
        }
      }
      if (!reader_has_data) break;
      // Opencv decode and augments
      cv::Mat res;
      rec.Load(blob.dptr, blob.size);
      cv::Mat buf(1, rec.content_size, CV_8U, rec.content);
      switch (param_.data_shape[0]) {
       case 1:
#if MXNET_USE_LIBJPEG_TURBO
        res = TJimdecode(buf, 0);
#else
        res = cv::imdecode(buf, 0);
#endif
        break;
       case 3:
#if MXNET_USE_LIBJPEG_TURBO
        res = TJimdecode(buf, 1);
#else
        res = cv::imdecode(buf, 1);
#endif
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
      mshadow::Tensor<cpu, 3, DType> data;
      if (idx < batch_param_.batch_size) {
        data = mshadow::Tensor<cpu, 3, DType>(data_dptr + idx*unit_size_[0],
          mshadow::Shape3(n_channels, res.rows, res.cols));
      } else {
        out_tmp.Push(static_cast<unsigned>(rec.image_index()),
                 mshadow::Shape3(n_channels, res.rows, res.cols),
                 mshadow::Shape1(param_.label_width));
        data = out_tmp.data().Back();
      }

      std::uniform_real_distribution<float> rand_uniform(0, 1);
      std::bernoulli_distribution coin_flip(0.5);
      bool is_mirrored = (normalize_param_.rand_mirror && coin_flip(*(prnds_[tid])))
                         || normalize_param_.mirror;
      float contrast_scaled = 1;
      float illumination_scaled = 0;
      if (!std::is_same<DType, uint8_t>::value) {
        contrast_scaled =
          (rand_uniform(*(prnds_[tid])) * normalize_param_.max_random_contrast * 2
          - normalize_param_.max_random_contrast + 1)*normalize_param_.scale;
        illumination_scaled =
          (rand_uniform(*(prnds_[tid])) * normalize_param_.max_random_illumination * 2
          - normalize_param_.max_random_illumination) * normalize_param_.scale;
      }
      // For RGB or RGBA data, swap the B and R channel:
      // OpenCV store as BGR (or BGRA) and we want RGB (or RGBA)
      if (n_channels == 1) {
        ProcessImage<1>(res, &data, is_mirrored, contrast_scaled, illumination_scaled);
      } else if (n_channels == 3) {
        ProcessImage<3>(res, &data, is_mirrored, contrast_scaled, illumination_scaled);
      } else if (n_channels == 4) {
        ProcessImage<4>(res, &data, is_mirrored, contrast_scaled, illumination_scaled);
      }

      mshadow::Tensor<cpu, 1, real_t> label;
      if (idx < batch_param_.batch_size) {
        label = mshadow::Tensor<cpu, 1, real_t>(label_dptr + idx*unit_size_[1],
          mshadow::Shape1(param_.label_width));
      } else {
        label = out_tmp.label().Back();
      }

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
  return (std::min(batch_param_.batch_size, gl_idx) - current_size);
#else
  LOG(FATAL) << "Opencv is needed for image decoding and augmenting.";
  return 0;
#endif
}

// create mean image.
template<typename DType>
inline void ImageRecordIOParser2<DType>::CreateMeanImg(void) {
    if (param_.verbose) {
      LOG(INFO) << "Cannot find " << normalize_param_.mean_img
                << ": create mean image, this will take some time...";
    }
    double start = dmlc::GetTime();
    dmlc::InputSplit::Blob chunk;
    size_t imcnt = 0;  // NOLINT(*)
    while (source_->NextChunk(&chunk)) {
      inst_order_.clear();
      // Parse chunk w/o putting anything in out
      ParseChunk(NULL, NULL, batch_param_.batch_size, &chunk);
      for (unsigned i = 0; i < inst_order_.size(); ++i) {
        std::pair<unsigned, unsigned> place = inst_order_[i];
        mshadow::Tensor<cpu, 3> outimg =
          temp_[place.first][place.second].data[0].template get<cpu, 3, real_t>();
        if (imcnt == 0) {
          meanimg_.Resize(outimg.shape_);
          mshadow::Copy(meanimg_, outimg);
        } else {
          meanimg_ += outimg;
        }
        imcnt += 1;
        double elapsed = dmlc::GetTime() - start;
        if (imcnt % 10000L == 0 && param_.verbose) {
          LOG(INFO) << imcnt << " images processed, " << elapsed << " sec elapsed";
        }
      }
    }
    meanimg_ *= (1.0f / imcnt);
    // save as mxnet python compatible format.
    TBlob tmp = meanimg_;
    {
      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(normalize_param_.mean_img.c_str(), "w"));
      NDArray::Save(fo.get(),
                    {NDArray(tmp, 0)},
                    {"mean_img"});
    }
    if (param_.verbose) {
      LOG(INFO) << "Save mean image to " << normalize_param_.mean_img << "..";
    }
    meanfile_ready_ = true;
    this->BeforeFirst();
}

template<typename DType = real_t>
class ImageRecordIter2 : public IIterator<DataBatch> {
 public:
    ImageRecordIter2() : out_(nullptr) { }

    virtual ~ImageRecordIter2(void) {
      iter_.Destroy();
    }

    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      prefetch_param_.InitAllowUnknown(kwargs);
      parser_.Init(kwargs);
      // maximum prefetch threaded iter internal size
      const int kMaxPrefetchBuffer = 16;
      // init thread iter
      iter_.set_max_capacity(kMaxPrefetchBuffer);
      // init thread iter
      iter_.Init([this](DataBatch **dptr) {
          if (*dptr == nullptr) {
            *dptr = new DataBatch();
          }
          return parser_.ParseNext(*dptr);
          },
          [this]() { parser_.BeforeFirst(); });
    }

    virtual void BeforeFirst(void) {
      iter_.BeforeFirst();
    }

    // From iter_prefetcher.h
    virtual bool Next(void) {
      if (out_ != nullptr) {
        recycle_queue_.push(out_); out_ = nullptr;
      }
      // do recycle
      if (recycle_queue_.size() == prefetch_param_.prefetch_buffer) {
        DataBatch *old_batch =  recycle_queue_.front();
        // can be more efficient on engine
        for (NDArray& arr : old_batch->data) {
          arr.WaitToWrite();
        }
        recycle_queue_.pop();
        iter_.Recycle(&old_batch);
      }
      return iter_.Next(&out_);
    }

    virtual const DataBatch &Value(void) const {
      return *out_;
    }

 private:
    /*! \brief Backend thread */
    dmlc::ThreadedIter<DataBatch> iter_;
    /*! \brief Parameters */
    PrefetcherParam prefetch_param_;
    /*! \brief output data */
    DataBatch *out_;
    /*! \brief queue to be recycled */
    std::queue<DataBatch*> recycle_queue_;
    /* \brief parser */
    ImageRecordIOParser2<DType> parser_;
};

MXNET_REGISTER_IO_ITER(ImageRecordIter)
.describe(R"code(Iterates on image RecordIO files

Reads batches of images from .rec RecordIO files. One can use ``im2rec.py`` tool
(in tools/) to pack raw image files into RecordIO files. This iterator is less
flexible to customization but is fast and has lot of language bindings. To
iterate over raw images directly use ``ImageIter`` instead (in Python).

Example::

  data_iter = mx.io.ImageRecordIter(
    path_imgrec="./sample.rec", # The target record file.
    data_shape=(3, 227, 227), # Output data shape; 227x227 region will be cropped from the original image.
    batch_size=4, # Number of items per batch.
    resize=256 # Resize the shorter edge to 256 before cropping.
    # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
    )
  # You can now use the data_iter to access batches of images.
  batch = data_iter.next() # first batch.
  images = batch.data[0] # This will contain 4 (=batch_size) images each of 3x227x227.
  # process the images
  ...
  data_iter.reset() # To restart the iterator from the beginning.

)code" ADD_FILELINE)
.add_arguments(ImageRecParserParam::__FIELDS__())
.add_arguments(ImageRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.add_arguments(ImageNormalizeParam::__FIELDS__())
.set_body([]() {
    return new ImageRecordIter2<real_t>();
    });

MXNET_REGISTER_IO_ITER(ImageRecordUInt8Iter)
.describe(R"code(Iterating on image RecordIO files

This iterator is identical to ``ImageRecordIter`` except for using ``uint8`` as
the data type instead of ``float``.

)code" ADD_FILELINE)
.add_arguments(ImageRecParserParam::__FIELDS__())
.add_arguments(ImageRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.set_body([]() {
    return new ImageRecordIter2<uint8_t>();
  });
}  // namespace io
}  // namespace mxnet
