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
#include "./inst_vector.h"
#include "./image_recordio.h"
#include "./image_augmenter.h"
#include "./image_iter_common.h"
#include "./iter_prefetcher.h"
#include "./iter_normalize.h"
#include "./iter_batchloader.h"

namespace mxnet {
namespace io {
/*! \brief data structure to hold labels for image detection tasks
 *  support arbitrary label_width
 */
class ImageDetLabelMap {
 public:
  /*!
   * \brief initialize the label list into memory
   * \param path_imglist path to the image list
   * \param label_width predefined label_width, -1 for arbitrary width
   */
  explicit ImageDetLabelMap(const char *path_imglist,
                            int label_width,
                            bool silent) {
    image_index_.clear();
    label_.clear();
    idx2label_.clear();
    dmlc::InputSplit *fi = dmlc::InputSplit::Create
        (path_imglist, 0, 1, "text");
    dmlc::InputSplit::Blob rec;
    while (fi->NextRecord(&rec)) {
      // quick manual parsing
      char *p = reinterpret_cast<char*>(rec.dptr);
      char *end = p + rec.size;
      // skip space
      while (isspace(*p) && p != end) ++p;
      image_index_.push_back(static_cast<size_t>(atol(p)));
      size_t start_pos = label_.size();
      if (label_width > 0) {
        // provided label_width > 0, require width check
        for (int i = 0; i < label_width; ++i) {
          // skip till space
          while (!isspace(*p) && p != end) ++p;
          // skip space
          while (isspace(*p) && p != end) ++p;
          CHECK(p != end) << "Bad ImageList format";
          label_.push_back(static_cast<real_t>(atof(p)));
        }
        CHECK_EQ(label_.size() - start_pos, label_width);
      } else {
        // arbitrary label width for each sample
        while (!isspace(*p) && p != end) ++p;
        while (isspace(*p) && p != end) ++p;
        char *curr = p;
        CHECK(curr != end) << "Bad ImageList format";
        while (!isspace(*p) && p != end) ++p;
        while (isspace(*p) && p != end) ++p;
        char *next = p;
        while (next != end) {
          label_.push_back(static_cast<real_t>(atof(curr)));
          curr = next;
          while (!isspace(*next) && next != end) ++next;
          while (isspace(*next) && next != end) ++next;
        }
        // skip the last one which should be the image_path
        CHECK_GT(label_.size(), start_pos) << "Bad ImageList format: empty label";
      }
      // record label start_pos and width in map
      idx2label_[image_index_.back()] = std::pair<size_t, size_t>(
        start_pos, label_.size() - start_pos);
    }
    delete fi;
    if (!silent) {
      LOG(INFO) << "Loaded ImageList from " << path_imglist << ' '
                << image_index_.size() << " Image records";
    }
  }

  /*! \brief find a label for corresponding index, return vector as copy */
  inline std::vector<float> FindCopy(size_t imid) const {
    std::unordered_map<size_t, std::pair<size_t, size_t> >::const_iterator it
        = idx2label_.find(imid);
    CHECK(it != idx2label_.end()) << "fail to find imagelabel for id " << imid;
    const real_t *ptr = dmlc::BeginPtr(label_) + it->second.first;
    return std::vector<float>(ptr, ptr + it->second.second);
  }

  /*! \brief Iterate through all labels, find the Maximum width of labels */
  inline size_t MaxLabelWidth() const {
    size_t max_width = 0;
    for (auto i : idx2label_) {
      size_t width = i.second.second;
      if (width > max_width) max_width = width;
    }
    return max_width;
  }

 private:
  /*! \brief vector storing image indices */
  std::vector<size_t> image_index_;
  /*! \brief vectors storing raw labels in 1D */
  std::vector<real_t> label_;
  /*! \brief map storing image index to pair<label_start_pos, label_end_pos> */
  std::unordered_map<size_t, std::pair<size_t, size_t> > idx2label_;
};  // class ImageDetLabelMap

// Define image record parser parameters
struct ImageDetRecParserParam : public dmlc::Parameter<ImageDetRecParserParam> {
  /*! \brief path to image list */
  std::string path_imglist;
  /*! \brief path to image recordio */
  std::string path_imgrec;
  /*! \brief a sequence of names of image augmenters, seperated by , */
  std::string aug_seq;
  /*! \brief label-width, use -1 for variable width */
  int label_width;
  /*! \brief input shape */
  TShape data_shape;
  /*! \brief number of threads */
  int preprocess_threads;
  /*! \brief whether to remain silent */
  bool verbose;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  /*! \brief the size of a shuffle chunk*/
  size_t shuffle_chunk_size;
  /*! \brief the seed for chunk shuffling*/
  int shuffle_chunk_seed;
  /*! \brief pad label to specified length, -1 for auto estimate in whole dataset */
  int label_pad_width;
  /*! \brief labe padding value */
  float label_pad_value;

  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageDetRecParserParam) {
    DMLC_DECLARE_FIELD(path_imglist).set_default("")
        .describe("Dataset Param: Path to image list.");
    DMLC_DECLARE_FIELD(path_imgrec).set_default("./data/imgrec.rec")
        .describe("Dataset Param: Path to image record file.");
    DMLC_DECLARE_FIELD(aug_seq).set_default("det_aug_default")
        .describe("Augmentation Param: the augmenter names to represent"\
                  " sequence of augmenters to be applied, seperated by comma." \
                  " Additional keyword parameters will be seen by these augmenters."
                  " Make sure you don't use normal augmenters for detection tasks.");
    DMLC_DECLARE_FIELD(label_width).set_default(-1)
        .describe("Dataset Param: How many labels for an image, -1 for variable label size.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Dataset Param: Shape of each instance generated by the DataIter.");
    DMLC_DECLARE_FIELD(preprocess_threads).set_lower_bound(1).set_default(4)
        .describe("Backend Param: Number of thread to do preprocessing.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Auxiliary Param: Whether to output parser information.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
    DMLC_DECLARE_FIELD(shuffle_chunk_size).set_default(0)
        .describe("the size(MB) of the shuffle chunk, used with shuffle=True,"\
                  " it can enable global shuffling");
    DMLC_DECLARE_FIELD(shuffle_chunk_seed).set_default(0)
        .describe("the seed for chunk shuffling");
    DMLC_DECLARE_FIELD(label_pad_width).set_default(0)
        .describe("pad output label width if set larger than 0, -1 for auto estimate");
    DMLC_DECLARE_FIELD(label_pad_value).set_default(-1.f)
        .describe("label padding value if enabled");
  }
};

// parser to parse image recordio
template<typename DType>
class ImageDetRecordIOParser {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);

  // set record to the head
  inline void BeforeFirst(void) {
    return source_->BeforeFirst();
  }
  // parse next set of records, return an array of
  // instance vector to the user
  virtual inline bool ParseNext(std::vector<InstVector<DType>> *out);

 protected:
  // magic number to see prng
  static const int kRandMagic = 233;
  /*! \brief parameters */
  ImageDetRecParserParam param_;
  #if MXNET_USE_OPENCV
  /*! \brief augmenters */
  std::vector<std::vector<std::unique_ptr<ImageAugmenter> > > augmenters_;
  #endif
  /*! \brief random samplers */
  std::vector<std::unique_ptr<common::RANDOM_ENGINE> > prnds_;
  /*! \brief data source */
  std::unique_ptr<dmlc::InputSplit> source_;
  /*! \brief label information, if any */
  std::unique_ptr<ImageDetLabelMap> label_map_;
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
};

template<typename DType>
inline void ImageDetRecordIOParser<DType>::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
#if MXNET_USE_OPENCV
  // initialize parameter
  // init image rec param
  param_.InitAllowUnknown(kwargs);
  int maxthread, threadget;
  #pragma omp parallel
  {
    // be conservative, set number of real cores - 1
    maxthread = std::max(omp_get_num_procs() - 1, 1);
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
    label_map_.reset(new ImageDetLabelMap(param_.path_imglist.c_str(),
      param_.label_width, !param_.verbose));
  }
  CHECK(param_.path_imgrec.length() != 0)
      << "ImageDetRecordIOIterator: must specify image_rec";

  if (param_.verbose) {
    LOG(INFO) << "ImageDetRecordIOParser: " << param_.path_imgrec
              << ", use " << threadget << " threads for decoding..";
  }
  source_.reset(dmlc::InputSplit::Create(
      param_.path_imgrec.c_str(), param_.part_index,
      param_.num_parts, "recordio"));

  // estimate padding width for labels
  int max_label_width = 0;
  if (label_map_ != nullptr) {
    max_label_width = label_map_->MaxLabelWidth();
  } else {
    // iterate through recordio
    dmlc::InputSplit::Blob chunk;
    while (source_->NextChunk(&chunk)) {
      #pragma omp parallel num_threads(param_.preprocess_threads)
      {
        CHECK(omp_get_num_threads() == param_.preprocess_threads);
        int max_width = 0;
        int tid = omp_get_thread_num();
        dmlc::RecordIOChunkReader reader(chunk, tid, param_.preprocess_threads);
        ImageRecordIO rec;
        dmlc::InputSplit::Blob blob;
        while (reader.NextRecord(&blob)) {
          rec.Load(blob.dptr, blob.size);
          if (rec.label != NULL) {
            if (param_.label_width > 0) {
              CHECK_EQ(param_.label_width, rec.num_label)
                << "rec file provide " << rec.num_label << "-dimensional label "
                   "but label_width is set to " << param_.label_width;
            }
            // update max value
            max_width = std::max(max_width, rec.num_label);
          } else {
            LOG(FATAL) << "Not enough label packed in img_list or rec file.";
          }
        }
        #pragma omp critical
        {
          max_label_width = std::max(max_label_width, max_width);
        }
      }
    }
  }
  if (max_label_width > param_.label_pad_width) {
    if (param_.label_pad_width > 0) {
      LOG(FATAL) << "ImageDetRecordIOParser: label_pad_width: "
        << param_.label_pad_width << " smaller than estimated width: "
        << max_label_width;
    }
    param_.label_pad_width = max_label_width;
  }
  if (param_.verbose) {
    LOG(INFO) << "ImageDetRecordIOParser: " << param_.path_imgrec
              << ", label padding width: " << param_.label_pad_width;
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
  LOG(FATAL) << "ImageDetRec need opencv to process";
#endif
}

template<typename DType>
inline bool ImageDetRecordIOParser<DType>::
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
      // load label before augmentations
      std::vector<float> label_buf;
      if (this->label_map_ != nullptr) {
        label_buf = label_map_->FindCopy(rec.image_index());
      } else if (rec.label != NULL) {
        if (param_.label_width > 0) {
          CHECK_EQ(param_.label_width, rec.num_label)
            << "rec file provide " << rec.num_label << "-dimensional label "
               "but label_width is set to " << param_.label_width;
        }
        label_buf.assign(rec.label, rec.label + rec.num_label);
      } else {
        LOG(FATAL) << "Not enough label packed in img_list or rec file.";
      }
      for (auto& aug : this->augmenters_[tid]) {
        res = aug->Process(res, &label_buf, this->prnds_[tid].get());
      }
      out.Push(static_cast<unsigned>(rec.image_index()),
               mshadow::Shape3(n_channels, param_.data_shape[1], param_.data_shape[2]),
               mshadow::Shape1(param_.label_pad_width + 4));

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
      label = param_.label_pad_value;
      // store info for real data_shape and label_width
      label[0] = res.channels();
      label[1] = res.rows;
      label[2] = res.cols;
      label[3] = label_buf.size();
      mshadow::Copy(label.Slice(4, 4 + label_buf.size()),
        mshadow::Tensor<cpu, 1>(dmlc::BeginPtr(label_buf),
        mshadow::Shape1(label_buf.size())));
      res.release();
    }
  }
#else
      LOG(FATAL) << "Opencv is needed for image decoding and augmenting.";
#endif
  return true;
}

// Define image record parameters
struct ImageDetRecordParam: public dmlc::Parameter<ImageDetRecordParam> {
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief random seed */
  int seed;
  /*! \brief whether to remain silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageDetRecordParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(false)
        .describe("Augmentation Param: Whether to shuffle data.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Auxiliary Param: Whether to output information.");
  }
};

// iterator on image recordio
template<typename DType = real_t>
class ImageDetRecordIter : public IIterator<DataInst> {
 public:
  ImageDetRecordIter() : data_(nullptr) { }
  // destructor
  virtual ~ImageDetRecordIter(void) {
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
  static const int kRandMagic = 233;
  // output instance
  DataInst out_;
  // data ptr
  size_t inst_ptr_;
  // internal instance order
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  // data
  std::vector<InstVector<DType>> *data_;
  // internal parser
  ImageDetRecordIOParser<DType> parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVector<DType>> > iter_;
  // parameters
  ImageDetRecordParam param_;
  // random number generator
  common::RANDOM_ENGINE rnd_;
};

DMLC_REGISTER_PARAMETER(ImageDetRecParserParam);
DMLC_REGISTER_PARAMETER(ImageDetRecordParam);

MXNET_REGISTER_IO_ITER(ImageDetRecordIter)
.describe("Create iterator for image detection dataset packed in recordio.")
.add_arguments(ImageDetRecParserParam::__FIELDS__())
.add_arguments(ImageDetRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultDetAugParams())
.add_arguments(ImageDetNormalizeParam::__FIELDS__())
.set_body([]() {
  return new PrefetcherIter(
        new BatchLoader(
            new ImageDetNormalizeIter(
                new ImageDetRecordIter<real_t>())));
});
}  // namespace io
}  // namespace mxnet
