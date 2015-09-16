/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_image_recordio-inl.hpp
 * \brief recordio data
iterator
 */
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
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
#include "./iter_batch.h"
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
    this->label_width = label_width;
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
      for (size_t i = 0; i < label_width; ++i) {
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
      idx2label_[image_index_[i]] = dmlc::BeginPtr(label_) + i * label_width;
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
    return mshadow::Tensor<cpu, 1>(it->second, mshadow::Shape1(label_width));
  }

 private:
  // label with_
  mshadow::index_t label_width;
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
  std::string path_imglist;
  /*! \brief path to image recordio */
  std::string path_imgrec;
  /*! \brief number of threads */
  int nthread;
  /*! \brief whether to remain silent */
  bool silent;
  /*! \brief virtually split the data into n parts */
  int num_parts;
  /*! \brief only read the i-th part */
  int part_index;
  /*! \brief label-width */
  int label_width;
  /*! \brief input shape */
  TShape input_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecParserParam) {
    DMLC_DECLARE_FIELD(path_imglist).set_default("")
        .describe("Path to image list.");
    DMLC_DECLARE_FIELD(path_imgrec).set_default("./data/imgrec.rec")
        .describe("Path to image record file.");
    DMLC_DECLARE_FIELD(nthread).set_lower_bound(1).set_default(4)
        .describe("Number of thread to do parsing.");
    DMLC_DECLARE_FIELD(label_width).set_lower_bound(1).set_default(1)
        .describe("How many labels for an image.");
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Whether to output parser information.");
    DMLC_DECLARE_FIELD(num_parts).set_lower_bound(1).set_default(1)
        .describe("virtually split the data into n parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("only read the i-th part");
    index_t input_shape_default[] = {3, 224, 224};
    DMLC_DECLARE_FIELD(input_shape)
        .set_default(TShape(input_shape_default, input_shape_default + 3))
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Input shape of the neural net");
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
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
};

inline void ImageRecordIOParser::Init(
        const std::vector<std::pair<std::string, std::string> >& kwargs) {
  // initialize parameter
  // init image rec param
  param_.InitAllowUnknown(kwargs);
  int maxthread, threadget;
  #pragma omp parallel
  {
    // why ? (muli)
    maxthread = std::max(omp_get_num_procs() / 2 - 1, 1);
  }
  param_.nthread = std::min(maxthread, param_.nthread);
  #pragma omp parallel num_threads(param_.nthread)
  {
    threadget = omp_get_num_threads();
  }
  param_.nthread = threadget;
  // setup decoders
  for (int i = 0; i < threadget; ++i) {
    augmenters_.push_back(new ImageAugmenter());
    augmenters_[i]->Init(kwargs);
    prnds_.push_back(new common::RANDOM_ENGINE((i + 1) * kRandMagic));
  }

  if (param_.path_imglist.length() != 0) {
    label_map_ = new ImageLabelMap(param_.path_imglist.c_str(),
                                   param_.label_width, param_.silent != 0);
  } else {
    param_.label_width = 1;
  }
  CHECK(param_.path_imgrec.length() != 0)
    << "ImageRecordIOIterator: must specify image_rec";

  source_ = dmlc::InputSplit::Create(
      param_.path_imgrec.c_str(), param_.part_index,
      param_.num_parts, "recordio");
  // use 64 MB chunk when possible
  source_->HintChunkSize(8 << 20UL);
}

inline bool ImageRecordIOParser::
ParseNext(std::vector<InstVector> *out_vec) {
  CHECK(source_ != NULL);
  dmlc::InputSplit::Blob chunk;
  if (!source_->NextChunk(&chunk)) return false;
  out_vec->resize(param_.nthread);
  #pragma omp parallel num_threads(param_.nthread)
  {
    CHECK(omp_get_num_threads() == param_.nthread);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(chunk, tid, param_.nthread);
    ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // image data
    InstVector &out = (*out_vec)[tid];
    out.Clear();
    while (reader.NextRecord(&blob)) {
      rec.Load(blob.dptr, blob.size);
      out.Push(static_cast<unsigned>(rec.image_index()),
               mshadow::Shape3(param_.input_shape[0], param_.input_shape[1], param_.input_shape[2]),
               mshadow::Shape1(param_.label_width));
      DataInst inst = out.Back();
      // turn datainst into tensor
      mshadow::Tensor<mshadow::cpu, 3> data = inst.data[0].get<mshadow::cpu, 3, float>();
      mshadow::Tensor<mshadow::cpu, 1> label = inst.data[1].get<mshadow::cpu, 1, float>();
      augmenters_[tid]->Process(rec.content, rec.content_size, &img_, prnds_[tid]);
      mshadow::Copy(data, img_);
      if (label_map_ != NULL) {
        mshadow::Copy(label, label_map_->Find(rec.image_index()));
      } else {
        label[0] = rec.header.label;
      }
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
  /*! \brief mean file string*/
  std::string mean_img;
  /*! \brief whether to remain silent */
  bool silent;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecordParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(true)
        .describe("Whether to shuffle data.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Random Seed.");
    DMLC_DECLARE_FIELD(mean_img).set_default("./data/mean.bin")
        .describe("Path to image mean file.");
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Whether to output information.");
  }
};


// iterator on image recordio
class ImageRecordIter : public IIterator<DataInst> {
 public:
  ImageRecordIter() : data_(NULL) { }
  virtual ~ImageRecordIter(void) {
    iter_.Destroy();
    delete data_;
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    // use the kwarg to init parser
    parser_.Init(kwargs);
    // prefetch at most 4 minbatches
    iter_.set_max_capacity(4);
    // init thread iter
    iter_.Init([this](std::vector<InstVector> **dptr) {
        if (*dptr == NULL) {
          *dptr = new std::vector<InstVector>();
        }
        return parser_.ParseNext(*dptr);
      },
      [this]() { parser_.BeforeFirst(); });
    // Check Meanfile
    if (param_.mean_img.length() != 0) {
      dmlc::Stream *fi =
          dmlc::Stream::Create(param_.mean_img.c_str(), "r", true);
      if (fi == NULL) {
        this->CreateMeanImg();
      } else {
        delete fi;
      }
    }
    inst_ptr_ = 0;
    shuffle_ = param_.shuffle;
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
            std::shuffle(inst_order_.begin(), inst_order_.end(), \
                    common::RANDOM_ENGINE(kRandMagic + param_.seed));
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
  inline void CreateMeanImg(void) {
    if (param_.silent == 0) {
      printf("cannot find %s: create mean image, this will take some time...\n",
              param_.mean_img.c_str());
    }
    time_t start = time(NULL);
    uint64_t elapsed = 0;
    size_t imcnt = 1;
    this->BeforeFirst();
    CHECK(this->Next()) << "input iterator failed.";
    // Get the first data
    mshadow::Tensor<mshadow::cpu, 3> img_tensor = out_.data[0].get<mshadow::cpu, 3, float>();
    meanimg_.Resize(img_tensor.shape_);
    mshadow::Copy(meanimg_, img_tensor);
    while (this->Next()) {
      mshadow::Tensor<mshadow::cpu, 3> img_tensor = out_.data[0].get<mshadow::cpu, 3, float>();
      meanimg_ += img_tensor; imcnt += 1;
      elapsed = (uint64_t)(time(NULL) - start);
      if (imcnt % 1000 == 0 && param_.silent == 0) {
        printf("\r                                                               \r");
        printf("[%8lu] images processed, %ld sec elapsed", imcnt, elapsed);
        fflush(stdout);
      }
    }
    meanimg_ *= (1.0f / imcnt);

    dmlc::Stream *fo = dmlc::Stream::Create(param_.mean_img.c_str(), "w");
    meanimg_.SaveBinary(*fo);
    delete fo;
    if (param_.silent == 0) {
      printf("save mean image to %s..\n", param_.mean_img.c_str());
    }
  }

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
  ImageRecordParam param_;
  // mean image
  mshadow::TensorContainer<cpu, 3> meanimg_;
};

DMLC_REGISTER_PARAMETER(ImageRecParserParam);
DMLC_REGISTER_PARAMETER(ImageRecordParam);
MXNET_REGISTER_IO_CHAINED_ITER(ImageRecordIter, ImageRecordIter, BatchAdaptIter)
    .describe("Create iterator for dataset packed in recordio.")
    .add_arguments(ImageRecordParam::__FIELDS__())
    .add_arguments(ImageRecParserParam::__FIELDS__())
    .add_arguments(BatchParam::__FIELDS__())
    .add_arguments(ImageAugmentParam::__FIELDS__());
}  // namespace io
}  // namespace mxnet
