/*!
 *  Copyright (c) 2016 by Contributors
 * \file iter_feature.cc
 * \brief feature recordio data iterator
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/common.h>
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
#include "./iter_prefetcher.h"
#include "./iter_batchloader.h"

namespace mxnet {
namespace io {


// Define feature record parser parameters
struct FeatureParserParam : public dmlc::Parameter<FeatureParserParam> {
  /*! \brief path to feature bin file, from numpy.tofile */
  std::string path_featurebin;
  /*! \brief label count */
  int label_count;
  /*! \brief feature count per image */
  int feature_dim;
  /*! \brief number of threads */
  int preprocess_threads;
  /*! \brief whether to remain silent */
  bool verbose;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;

  // declare parameters
  DMLC_DECLARE_PARAMETER(FeatureParserParam) {
    DMLC_DECLARE_FIELD(path_featurebin).set_default("./data/feature.rec")
        .describe("Dataset Param: Path to packed feature rec file");
    DMLC_DECLARE_FIELD(label_count).set_lower_bound(1).set_default(1)
        .describe("Dataset Param: How many labels for an image.");
    DMLC_DECLARE_FIELD(feature_dim).set_lower_bound(1).set_default(1024)
        .describe("Dataset Param: the dimension of the feature array.");
    DMLC_DECLARE_FIELD(preprocess_threads).set_lower_bound(1).set_default(4)
        .describe("Backend Param: Number of thread to do preprocessing.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Auxiliary Param: Whether to output parser information.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
  }
};

// parser to parse image recordio
class FeatureIOParser {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);

  // set record to the head
  inline void BeforeFirst(void) {
    return source_->BeforeFirst();
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(std::vector<InstVectorFea> *out);

 private:
  // magic nyumber to see prng
  static const int kRandMagic = 111;
  /*! \brief parameters */
  FeatureParserParam param_;
  #if MXNET_USE_OPENCV
  /*! \brief augmenters */
  std::vector<std::vector<std::unique_ptr<ImageAugmenter> > > augmenters_;
  #endif
  /*! \brief random samplers */
  std::vector<std::unique_ptr<common::RANDOM_ENGINE> > prnds_;
  /*! \brief data source */
  std::unique_ptr<dmlc::InputSplit> source_;
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
};

inline void FeatureIOParser::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
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

  CHECK(param_.path_featurebin.length() != 0)
      << "FeatureIOIterator: must specify a feature_rec file";

  if (param_.verbose) {
    LOG(INFO) << "FeatureIOParser: " << param_.path_featurebin
              << ", use " << threadget << " threads for decoding..";
  }
  source_.reset(dmlc::InputSplit::Create(
      param_.path_featurebin.c_str(), param_.part_index,
      param_.num_parts, "recordio"));
  // use 64 MB chunk when possible
  source_->HintChunkSize(8 << 20UL);
}

inline bool FeatureIOParser::
ParseNext(std::vector<InstVectorFea> *out_vec) {
  bool ret = true;
  CHECK(source_ != nullptr);
  dmlc::InputSplit::Blob chunk;
  if (!source_->NextChunk(&chunk)) return false;

  out_vec->resize(param_.preprocess_threads);
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    CHECK(omp_get_num_threads() == param_.preprocess_threads);
    int tid = omp_get_thread_num();
    dmlc::RecordIOChunkReader reader(chunk, tid, param_.preprocess_threads);
    ImageRecordIO rec;
    dmlc::InputSplit::Blob blob;
    // feature data
    InstVectorFea &out = (*out_vec)[tid];
    out.Clear();
    while (reader.NextRecord(&blob)) {

    rec.Load(blob.dptr, blob.size);
    out.Push(static_cast<unsigned>(rec.image_index()),
             mshadow::Shape1(param_.feature_dim),
             mshadow::Shape1(param_.label_count));

    if (rec.content_size != param_.feature_dim * sizeof(double)) {
        ret = false;
    }
    mshadow::Tensor<cpu, 1> data = out.data().Back();
    double* fea_val = reinterpret_cast<double*>(rec.content);
    for (int i = 0; i < param_.feature_dim; i++) {
        data[i] = static_cast<float>(fea_val[i]);
    }

    mshadow::Tensor<cpu, 1> label = out.label().Back();
    label[0] = rec.header.label;
    }
  }
  return ret;
}

// Define feature record parameters
struct FeaRecordParam: public dmlc::Parameter<FeaRecordParam> {
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief random seed */
  int seed;
  /*! \brief whether to remain silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(FeaRecordParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(false)
        .describe("Augmentation Param: Whether to shuffle data.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Auxiliary Param: Whether to output information.");
  }
};

// iterator on feature recordio
class FeatureRecordIter : public IIterator<DataInst> {
 public:
  FeatureRecordIter() : data_(nullptr) { }
  // destructor
  virtual ~FeatureRecordIter(void) {
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
    iter_.Init([this](std::vector<InstVectorFea> **dptr) {
        if (*dptr == nullptr) {
          *dptr = new std::vector<InstVectorFea>();
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
          const InstVectorFea& tmp = (*data_)[i];
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
  std::vector<InstVectorFea> *data_;
  // internal parser
  FeatureIOParser parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVectorFea> > iter_;
  // parameters
  FeaRecordParam param_;
  // random number generator
  common::RANDOM_ENGINE rnd_;
};

DMLC_REGISTER_PARAMETER(FeatureParserParam);
DMLC_REGISTER_PARAMETER(FeaRecordParam);

MXNET_REGISTER_IO_ITER(FeatureRecordIter)
.describe("Create iterator for feature packed in recordio.")
.add_arguments(FeatureParserParam::__FIELDS__())
.add_arguments(FeaRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
                new FeatureRecordIter()));
  });
}  // namespace io
}  // namespace mxnet
