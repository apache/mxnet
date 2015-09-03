/*!
 * \file iter_image_recordio-inl.hpp
 * \brief recordio data
iterator
 */
#include <cstdlib>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/recordio.h>
#include <dmlc/threadediter.h>
#include <unordered_map>
#include <vector>
#include "./inst_vector.h"
#include "./image_recordio.h"
#include "./image_augmenter.h"
#include "../utils/decoder.h"
namespace mxnet {
namespace io {
/*! \brief data structure to hold labels for images */
class ImageLabelMap {
 public:
  /*!
   * \brief initialize the label list into memory
   * \param path_imglist path to the image list
   * \param label_width predefined label_width
   */
  explicit ImageLabelMap(const char *path_imglist,
                         mshadow::index_t label_width,
                         bool silent) {
    label_width_ = label_width;
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
      for (size_t i = 0; i < label_width_; ++i) {
        // skip till space
        while (!isspace(*p) && p != end) ++p;
        // skip space
        while (isspace(*p) && p != end) ++p;
        CHECK(p != end) << "Bad ImageList format";
        label_.push_back(static_cast<real_t>(atof(p)));
      }
    }
    delete fi;
    // be careful not to resize label_ afterwards
    idx2label_.reserve(image_index_.size());
    for (size_t i = 0; i < image_index_.size(); ++i) {
      idx2label_[image_index_[i]] = BeginPtr(label_) + i * label_width_;
    }
    if (!silent) {
      LOG(INFO) << "Loaded ImageList from " << path_imglist << ' '
                << image_index_.size() << " Image records";
    }
  }
  /*! \brief find a label for corresponding index */
  inline mshadow::Tensor<cpu, 1> Find(size_t imid) const {
    std::unordered_map<size_t, real_t*>::const_iterator it
        = idx2label_.find(imid);
    CHECK(it != idx2label_.end()) << "fail to find imagelabel for id " << imid;
    return mshadow::Tensor<cpu, 1>(it->second, mshadow::Shape1(label_width_));
  }

 private:
  // label with_
  mshadow::index_t label_width_;
  // image index of each record
  std::vector<size_t> image_index_;
  // real label content
  std::vector<real_t> label_;
  // map index to label
  std::unordered_map<size_t, real_t*> idx2label_;
};

// Define image record parser parameters
struct ImageRecParserParam : public dmlc::Parameter<ImageRecParserParam> {
  /*! \brief path to image list */
  std::string path_imglist_;
  /*! \brief path to image recordio */
  std::string path_imgrec_;
  /*! \brief number of threads */
  int nthread_;
  /*! \brief whether to remain silent */
  bool silent_;
  /*! \brief number of distributed worker */
  int dist_num_worker_, dist_worker_rank_;
  /*! \brief label-width */
  int label_width_;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecParserParam) {
    DMLC_DECLARE_FIELD(path_imglist_).set_default("")
        .describe("Path to image list.");
    DMLC_DECLARE_FIELD(path_imagrec_).set_default("./data/imgrec.rec")
        .describe("Path to image record file.");
    DMLC_DECLARE_FIELD(nthread_).set_lower_bound(1).set_default(4)
        .describe("Number of thread to do parsing.");
    DMLC_DECLARE_FIELD(label_width_).set_lower_bound(1).set_default(1)
        .describe("How many labels for an image.");
    DMLC_DECLARE_FIELD(silent_).set_default(false)
        .describe("Whether to output parser information.");
    DMLC_DECLARE_FIELD(dist_num_worker_).set_lower_bound(1).set_default(1)
        .describe("Dist worker number.");
    DMLC_DECLARE_FIELD(dist_worker_rank_).set_default(0)
        .describe("Dist worker rank.");
  }
};

// parser to parse image recordio
class ImageRecordIOParser {
 public:
  ImageRecordIOParser(void)
      : source_(NULL),
        label_map_(NULL) {
  }
  ~ImageRecordIOParser(void) {
    // can be NULL
    delete label_map_;
    delete source_;
    for (size_t i = 0; i < augmenters_.size(); ++i) {
      delete augmenters_[i];
    }
    for (size_t i = 0; i < prnds_.size(); ++i) {
      delete prnds_[i];
    }
  }
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);
  
  // set record to the head
  inline void BeforeFirst(void) {
    return source_->BeforeFirst();
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(std::vector<InstVector> *out);
 private:
  // magic nyumber to see prng
  static const int kRandMagic = 111;
  /*! \brief parameters */
  ImageRecParserParam param_; 
  /*! \brief augmenters */
  std::vector<ImageAugmenter*> augmenters_;
  /*! \brief random samplers */
  std::vector<common::RANDOM_ENGINE*> prnds_;
  /*! \brief data source */
  dmlc::InputSplit *source_;
  /*! \brief label information, if any */
  ImageLabelMap *label_map_;
};

inline void ImageRecordIOParser::Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
  // initialize parameter
  std::vector<std::pair<std::string, std::string> > kwargs_left;
  // init image rec param
  kwargs_left = param_.InitAllowUnknown(kwargs);
  int maxthread, threadget;
  #pragma omp parallel
  {
    maxthread = std::max(omp_get_num_procs() / 2 - 1, 1);
  }
  param_.nthread_ = std::min(maxthread, param_.nthread_);
  #pragma omp parallel num_threads(param_.nthread_)
  {
    threadget = omp_get_num_threads();
  }
  param_.nthread_ = threadget;
  // setup decoders
  for (int i = 0; i < threadget; ++i) {
    augmenters_.push_back(new ImageAugmenter());
    augmenters_[i].init(kwargs_left);
    prnds_.push_back(new common::RANDOM_ENGINE((i + 1) * kRandMagic));
  }
  
  // handling for hadoop
  // TODO, hack
  const char *ps_rank = getenv("PS_RANK");
  if (ps_rank != NULL) {
    param_.dist_worker_rank = atoi(ps_rank);
  }

  if (param_.path_imglist_.length() != 0) {
    label_map_ = new ImageLabelMap(param_.path_imglist_.c_str(),
                                   param_.label_width_, silent_ != 0);
  } else {
    param_.label_width_ = 1;
  }
  CHECK(path_imgrec_.length() != 0)
    << "ImageRecordIOIterator: must specify image_rec";
#if MSHADOW_DIST_PS
    // TODO move to a better place
    param_.dist_num_worker_ = ::ps::RankSize();
    param_.dist_worker_rank_ = ::ps::MyRank();
    LOG(INFO) << "rank " << param_.dist_worker_rank_
              << " in " << param_.dist_num_worker_;
#endif
  source_ = dmlc::InputSplit::Create
      (param_.path_imgrec_.c_str(), param_.dist_worker_rank_,
       param_.dist_num_worker_, "recordio");
  // use 64 MB chunk when possible
  source_->HintChunkSize(8 << 20UL);
}

inline bool ImageRecordIOParser::
ParseNext(std::vector<InstVector> *out_vec) {
  CHECK(source_ != NULL);
  dmlc::InputSplit::Blob chunk;
  if (!source_->NextChunk(&chunk)) return false;
  out_vec->resize(param_.nthread_);
  #pragma omp parallel num_threads(param_.nthread_)
  {
    CHECK(omp_get_num_threads() == param_.nthread_);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(chunk, tid, parser_.nthread_);
    mxnet::ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector &out = (*out_vec)[tid];
    out.Clear();
    while (reader.NextRecord(&blob)) {
      // result holder
      cv::Mat res;
      rec.Load(blob.dptr, blob.size);
      cv::Mat buf(1, rec.content_size, CV_8U, rec.content);
      res = cv::imdecode(buf, 1);
      res = augmenters_[tid]->Process(res, prnds_[tid]);
      out.Push(static_cast<unsigned>(rec.image_index()),
               mshadow::Shape3(3, res.rows, res.cols),
               mshadow::Shape1(param_.label_width_));
      DataInst inst = out.Back();
      for (int i = 0; i < res.rows; ++i) {
        for (int j = 0; j < res.cols; ++j) {
          cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
          inst.data[0][i][j] = bgr[2];
          inst.data[1][i][j] = bgr[1];
          inst.data[2][i][j] = bgr[0];
        }
      }
      if (label_map_ != NULL) {
        mshadow::Copy(inst.label, label_map_->Find(rec.image_index()));
      } else {
        inst.label[0] = rec.header.label;
      }
      res.release();
    }
  }
  return true;
}

// Define image record parameters
struct ImageRecordParam: public dmlc::Parameter<ImageRecordParam> {
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief random seed */
  int seed;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecordParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(true)
        .describe("Whether to shuffle data.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Random Seed.");
  }
};

// iterator on image recordio
class ImageRecordIter : public IIterator<DataInst> {
 public:
  ImageRecordIter()
      : data_(NULL) {
  }
  virtual ~ImageRecordIter(void) {
    iter_.Destroy();
    // data can be NULL
    delete data_;
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init image rec param
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // use the left kwarg to init parser
    parser_.Init(kwargs_left);
    // init thread iter
    iter_.set_max_capacity(4);
    iter_.Init([this](std::vector<InstVector> **dptr) {
        if (*dptr == NULL) {
          *dptr = new std::vector<InstVector>();
        }
        return parser_.ParseNext(*dptr);
      },
      [this]() { parser_.BeforeFirst(); });
    inst_ptr_ = 0;
  }
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
        if (data_ != NULL) iter_.Recycle(&data_);
        if (!iter_.Next(&data_)) return false;
        inst_order_.clear();
        for (unsigned i = 0; i < data_->size(); ++i) {
          const InstVector &tmp = (*data_)[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
        }
        // shuffle instance order if needed
        if (shuffle_ != 0) {
            std::shuffle(inst_order_.begin(), inst_.end(), common::RANDOM_ENGINE(kRandMagic + param_.seed));
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
  // whether shuffle data
  int shuffle_;
  // data ptr
  size_t inst_ptr_;
  // internal instance order
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  // data
  std::vector<InstVector> *data_;
  // internal parser
  ImageRecordIOParser parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVector> > iter_;
  // parameters
  ImageRecParserParam param_;
};
DMLC_REGISTER_PARAMETER(ImageRecParserParam);
DMLC_REGISTER_PARAMETER(ImageRecordParam);
MXNET_REGISTER_IO_ITER(MNISTIter, MNISTIter)
MXNET_REGISTER_IO_CHAINED_ITER(ImageRecordIter, ImageRecordIter, BatchAdaptIter)
    .describe("Create iterator for dataset packed in recordio.")
    .add_arguments(ImageRecordParam::__FIELDS__())
    .add_arguments(ImageRecParserParam::__FIELDS__())
    .add_arguments(BatchParam::__FIELDS__())
    .add_arguments(ImageAugmenterParam::__FIELDS__());
}  // namespace io
}  // namespace mxnet
#endif  // ITER_IMAGE_RECORDIO_INL_HPP_
