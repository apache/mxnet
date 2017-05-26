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
  inline void ParseChunk(dmlc::InputSplit::Blob * chunk);
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
      }
    }
  }
#else
  LOG(FATAL) << "ImageRec need opencv to process";
#endif
}

template<typename DType>
inline bool ImageRecordIOParser2<DType>::ParseNext(DataBatch *out) {
  if (overflow)
    return false;
  CHECK(source_ != nullptr);
  dmlc::InputSplit::Blob chunk;
  unsigned current_size = 0;
  out->index.resize(batch_param_.batch_size);
  while (current_size < batch_param_.batch_size) {
    int n_to_copy;
    if (n_parsed_ == 0) {
      if (source_->NextChunk(&chunk)) {
        inst_order_.clear();
        inst_index_ = 0;
        ParseChunk(&chunk);
        unsigned n_read = 0;
        for (unsigned i = 0; i < temp_.size(); ++i) {
          const InstVector<DType>& tmp = temp_[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
          n_read += tmp.Size();
        }
        n_to_copy = std::min(n_read, batch_param_.batch_size - current_size);
        n_parsed_ = n_read - n_to_copy;
        // shuffle instance order if needed
        if (record_param_.shuffle != 0) {
          std::shuffle(inst_order_.begin(), inst_order_.end(), rnd_);
        }
      } else {
        if (current_size == 0) return false;
        CHECK(!overflow) << "number of input images must be bigger than the batch size";
        if (batch_param_.round_batch != 0) {
          overflow = true;
          source_->BeforeFirst();
        } else {
          current_size = batch_param_.batch_size;
        }
        out->num_batch_padd = batch_param_.batch_size - current_size;
        n_to_copy = 0;
      }
    } else {
      n_to_copy = std::min(n_parsed_, batch_param_.batch_size - current_size);
      n_parsed_ -= n_to_copy;
    }

    // InitBatch
    if (out->data.size() == 0 && n_to_copy != 0) {
      std::pair<unsigned, unsigned> place = inst_order_[inst_index_];
      const DataInst& first_batch = temp_[place.first][place.second];
      out->data.resize(first_batch.data.size());
      unit_size_.resize(first_batch.data.size());
      for (size_t i = 0; i < out->data.size(); ++i) {
        TShape src_shape = first_batch.data[i].shape_;
        int src_type_flag = first_batch.data[i].type_flag_;
        // init object attributes
        std::vector<index_t> shape_vec;
        shape_vec.push_back(batch_param_.batch_size);
        for (index_t dim = 0; dim < src_shape.ndim(); ++dim) {
          shape_vec.push_back(src_shape[dim]);
        }
        TShape dst_shape(shape_vec.begin(), shape_vec.end());
        auto dtype = prefetch_param_.dtype
          ? prefetch_param_.dtype.value()
          : first_batch.data[i].type_flag_;
        out->data.at(i) = NDArray(dst_shape, Context::CPU(), false , src_type_flag);
        unit_size_[i] = src_shape.Size();
      }
    }

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
    inst_index_ += n_to_copy;
    current_size += n_to_copy;
  }
  return true;
}

template<typename DType>
inline void ImageRecordIOParser2<DType>::ParseChunk(dmlc::InputSplit::Blob * chunk) {
  temp_.resize(param_.preprocess_threads);
#if MXNET_USE_OPENCV
  // save opencv out
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    CHECK(omp_get_num_threads() == param_.preprocess_threads);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(*chunk, tid, param_.preprocess_threads);
    ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector<DType> &out = temp_[tid];
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

      std::uniform_real_distribution<float> rand_uniform(0, 1);
      std::bernoulli_distribution coin_flip(0.5);
      bool is_mirrored = (normalize_param_.rand_mirror && coin_flip(*(prnds_[tid])))
                         || normalize_param_.mirror;
      float contrast_scaled;
      float illumination_scaled;
      if (!std::is_same<DType, uint8_t>::value) {
        contrast_scaled =
          (rand_uniform(*(prnds_[tid])) * normalize_param_.max_random_contrast * 2
          - normalize_param_.max_random_contrast + 1)*normalize_param_.scale;
        illumination_scaled =
          (rand_uniform(*(prnds_[tid])) * normalize_param_.max_random_illumination * 2
          - normalize_param_.max_random_illumination) * normalize_param_.scale;
      }
      for (int i = 0; i < res.rows; ++i) {
        uchar* im_data = res.ptr<uchar>(i);
        for (int j = 0; j < res.cols; ++j) {
          DType RGBA[4];
          for (int k = 0; k < n_channels; ++k) {
            RGBA[k] = im_data[swap_indices[k]];
          }
          if (!std::is_same<DType, uint8_t>::value) {
            // normalize/mirror here to avoid memory copies
            // logic from iter_normalize.h, function SetOutImg

            if (normalize_param_.mean_r > 0.0f || normalize_param_.mean_g > 0.0f ||
                normalize_param_.mean_b > 0.0f || normalize_param_.mean_a > 0.0f) {
              // subtract mean per channel
              RGBA[0] -= normalize_param_.mean_r;
              if (n_channels >= 3) {
                RGBA[1] -= normalize_param_.mean_g;
                RGBA[2] -= normalize_param_.mean_b;
              }
              if (n_channels == 4) {
                RGBA[3] -= normalize_param_.mean_a;
              }
              for (int k = 0; k < n_channels; ++k) {
                RGBA[k] = RGBA[k] * contrast_scaled + illumination_scaled;
              }
            } else if (!meanfile_ready_ || normalize_param_.mean_img.length() == 0) {
              // do not subtract anything
              for (int k = 0; k < n_channels; ++k) {
                RGBA[k] = RGBA[k] * normalize_param_.scale;
              }
            } else {
              CHECK(meanfile_ready_);
              for (int k = 0; k < n_channels; ++k) {
                  RGBA[k] = (RGBA[k] - meanimg_[k][i][j]) * contrast_scaled + illumination_scaled;
              }
            }
          }
          for (int k = 0; k < n_channels; ++k) {
            if (!std::is_same<DType, uint8_t>::value) {
              // normalize/mirror here to avoid memory copies
              // logic from iter_normalize.h, function SetOutImg
              if (is_mirrored) {
                data[k][i][res.cols - j - 1] = RGBA[k];
              } else {
                data[k][i][j] = RGBA[k];
              }
            } else {
              // do not do normalization in Uint8 reader
              data[k][i][j] = RGBA[k];
            }
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
      ParseChunk(&chunk);
      inst_order_.clear();
      for (unsigned i = 0; i < temp_.size(); ++i) {
        const InstVector<DType>& tmp = temp_[i];
        for (unsigned j = 0; j < tmp.Size(); ++j) {
          inst_order_.push_back(std::make_pair(i, j));
        }
      }
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
