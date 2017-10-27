/*!
 *  Copyright (c) 2017 by shuqian.qu@hobot.cc
 * \file iter_recordio.h
 * \brief Interface of recordio iter
 */

#ifndef MXNET_IO_ITER_RECORDIO_H_
#define MXNET_IO_ITER_RECORDIO_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/recordio.h>
#include <dmlc/memory_io.h>
#include <dmlc/threadediter.h>
#include <dmlc/input_split_shuffle.h>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include "../common/utils.h"
#include "./iter_threaded_input_split.h"

#if PB_FORMAT_REC
#include "./recordio.pb.h"
namespace mxnet {
namespace io {

namespace record_data_type {
enum RecordDataType {kNdarray, kBinary, kString};
}

struct RecordInst {
  // value
  struct TValue {
    NDArray nd;
    std::string str;
  };
  // record data
  struct RData {
    uint64_t id;
    int type;
    TValue value;
  };
  // extra data
  struct ExData {
    std::string key;
    int type;
    TValue value;
  };

  RecordHead head;
  std::vector<RData> data;
  std::vector<float> label;
  std::vector<ExData> extra;
};

/*!
 * \brief a list
 */
class RecordInstVector {
 public:
  /*! \brief return the number of record */
  inline size_t Size(void) const {
    return record_list_.size();
  }
  // instance
  /* \brief get the i-th record */
  inline RecordInst operator[](size_t i) const {
    return record_list_[i];
  }
  /* \brief get the last (label, example) pair */
  inline RecordInst Back() const {
    return (*this)[Size() - 1];
  }

  inline void Clear(void) {
    record_list_.clear();
  }

  inline void Push(const RecordInst& rec) {
    record_list_.push_back(rec);
  }

  std::vector<RecordInst> record_list_;
};

// Define record parser parameters
struct RecParserParam : public dmlc::Parameter<RecParserParam> {
  /*! \brief the ratio of data sampling*/
  float sampling_ratio;
  DMLC_DECLARE_PARAMETER(RecParserParam) {
    DMLC_DECLARE_FIELD(sampling_ratio).set_default(1.0f)
        .describe("the ratio of data sampling");
  }
};

// parser to parse recordio
class RecordIOParser {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs,
                   int parse_thread);

  // set record to the head
  inline void BeforeFirst(void) { }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(RecordInstVector *out_vec,
                        dmlc::InputSplit::Blob chunk,
                        int tid);

 private:
  // magic number to see prng
  static const int kRandMagic = 111;
  /*! \brief parameters */
  RecParserParam param_;
  /*! \brief random samplers */
  std::vector<std::unique_ptr<common::RANDOM_ENGINE> > prnds_;
};

inline void LoadRecTypeValue(RecordInst::TValue* value,
                          const std::string& origin, int type) {
  switch (type) {
    case record_data_type::kNdarray:
      {
        // NDArray value
        auto val = origin;
        dmlc::MemoryStringStream strm(&val); // NOLINT(*)
        if (!value->nd.Load(&strm)) {
          LOG(FATAL) << "Invalid NDArray serialization format!";
        }
      }
      break;
    case record_data_type::kBinary:
      {
        // string to NDArray type
        value->nd = NDArray(mshadow::Shape1(origin.size()),
                            Context::CPU(),
                            false,
                            mshadow::kUint8);
        auto nd_blob = value->nd.data();
        memcpy(nd_blob.dptr_, origin.c_str(), origin.size());
      }
      break;
    case record_data_type::kString:
      {
        value->str = origin;
      }
      break;
    default:
      {
        LOG(FATAL) << "read data type error:" << type;
      }
      break;
  }
  return;
}

inline void RecordIOParser::Init(
        const std::vector<std::pair<std::string, std::string> >& kwargs,
        int parse_thread) {
  // initialize parameter
  param_.InitAllowUnknown(kwargs);
  for (int i = 0; i < parse_thread; ++i) {
    prnds_.emplace_back(new common::RANDOM_ENGINE((i + 1) * kRandMagic));
  }
}

inline bool RecordIOParser::ParseNext(RecordInstVector *out_data,
                                      dmlc::InputSplit::Blob chunk,
                                      int tid) {
  dmlc::RecordIOChunkReader reader(chunk);
  auto& out = *out_data;
  out.Clear();
  dmlc::InputSplit::Blob blob;
  while (reader.NextRecord(&blob)) {
    if (param_.sampling_ratio < 1.0f) {
      std::uniform_real_distribution<float> dis(0, 1);
      // drop all samples in cat
      if (dis(*prnds_[tid]) > param_.sampling_ratio) continue;
    }
    if (blob.size <= 0) continue;
    RecordInst rec;
    RecordUnit pb_record;
    pb_record.ParseFromString(std::string(static_cast<const char*>(blob.dptr), blob.size));
    if (!pb_record.has_head() || !pb_record.has_body()) continue;
    rec.head = pb_record.head();
    const auto& body = pb_record.body();
    // data
    rec.data.resize(body.data_size());
    for (int i = 0; i < body.data_size(); i++) {
      const auto& data_i = body.data(i);
      rec.data[i].id = data_i.id();
      rec.data[i].type = data_i.type();
      LoadRecTypeValue(&rec.data[i].value, data_i.value(), data_i.type());
    }
    // label
    for (int i = 0; i < body.label_size(); i++) {
      rec.label.push_back(body.label(i));
    }
    // extra
    rec.extra.resize(body.extra_size());
    for (int i = 0; i < body.extra_size(); i++) {
      const auto& extra_i = body.extra(i);
      rec.extra[i].key = extra_i.key();
      rec.extra[i].type = extra_i.type();
      LoadRecTypeValue(&rec.extra[i].value, extra_i.value(), extra_i.type());
    }
    out.Push(rec);
  }
  return true;
}

// RecordDataInstIter parameters
struct RecordDataInstParam : public dmlc::Parameter<RecordDataInstParam> {
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief random seed */
  int seed;
  /*! \brief whether to remain silent */
  bool verbose;
  /*! \brief path to recordio */
  std::string path_imgrec;
  /*! \brief number of threads */
  int preprocess_threads;
  /*! \brief whether to read by chunk*/
  bool chunk_read;
  /*! \brief size of data chunk */
  int chunk_size;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  /*! \brief the size of a shuffle chunk*/
  size_t shuffle_chunk_size;
  /*! \brief the seed for chunk shuffling*/
  int shuffle_chunk_seed;
  /*! \brief source chunk's queue capacity*/
  int source_chunk_queue_capacity;
  /*! \brief record chunk's queue capacity*/
  int record_chunk_queue_capacity;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RecordDataInstParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(false)
        .describe("Augmentation Param: Whether to shuffle data.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Auxiliary Param: Whether to output information.");
    DMLC_DECLARE_FIELD(path_imgrec).set_default("")
        .describe("Dataset Param: Path to record file.");
    DMLC_DECLARE_FIELD(preprocess_threads).set_lower_bound(1).set_default(2)
        .describe("Backend Param: Number of thread to do preprocessing.");
    DMLC_DECLARE_FIELD(chunk_size).set_lower_bound(4).set_default(16)
        .describe("Backend Param: chunk_size.");
    DMLC_DECLARE_FIELD(chunk_read).set_default(true)
        .describe("Backend Param: whether to read by chunk.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
    DMLC_DECLARE_FIELD(shuffle_chunk_size).set_default(0)
        .describe("the size(MB) of the shuffle block, "
                  "it is recommended to use 16 or more when using hdfs");
    DMLC_DECLARE_FIELD(shuffle_chunk_seed).set_default(0)
        .describe("the seed for block shuffling");
    DMLC_DECLARE_FIELD(source_chunk_queue_capacity).set_default(2)
        .describe("source chunk's queue capacity");
    DMLC_DECLARE_FIELD(record_chunk_queue_capacity).set_default(2)
        .describe("record chunk's queue capacity");
  }
};

template <typename ParserType, typename InstVecType, typename InstType = DataInst>
class RecordDataInstIter : public IIterator<InstType> {
 public:
  RecordDataInstIter(): inst_ptr_(0), data_(nullptr), iter_(nullptr) { }
  // destructor
  virtual ~RecordDataInstIter(void) {
    delete iter_;
    iter_ = nullptr;
    delete data_;
    data_ = nullptr;
  }
  // constructor
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);
  // before first
  virtual void BeforeFirst(void) {
    iter_->BeforeFirst();
    inst_order_.clear();
    inst_ptr_ = 0;
  }

  virtual bool Next(void);

  virtual const InstType &Value(void) const {
    return out_;
  }

 private:
  // random magic
  static const int kRandMagic = 11111;
  // output instance
  InstType out_;
  // data ptr
  size_t inst_ptr_;
  // internal instance order
  std::vector<unsigned> inst_order_;
  // data
  InstVecType* data_;
  // internal parser
  ParserType parser_;
  /*! \brief data source */
  std::unique_ptr<dmlc::InputSplit> source_;
  // backend thread
  dmlc::MultiThreadedIter<InstVecType, CommonChunkType>* iter_;
  // parameters
  RecordDataInstParam param_;
  // random number generator
  common::RANDOM_ENGINE rnd_;
};

template <typename ParserType, typename InstVecType, typename InstType>
void RecordDataInstIter<ParserType, InstVecType, InstType>::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
  const auto& unknown_params = param_.InitAllowUnknown(kwargs);
  for (size_t index = 0; index < unknown_params.size(); index++) {
    CHECK(unknown_params[index].first != "shuffle_seed") <<
        "shuffle_seed param is deprecated, please use shuffle_chunk_seed instead!";
  }
  CHECK(param_.path_imgrec.length() != 0)
      << "RecordDataInstIter: must specify path_imgrec";
  // preprocess thread
#if defined(_OPENMP)
  int maxthread = 0;
  int threadget = 0;
  #pragma omp parallel
  {
    // be conservative, set number of real cores
    maxthread = std::max(omp_get_num_procs() - 1, 1);
  }
  param_.preprocess_threads = std::min(maxthread, param_.preprocess_threads);
  #pragma omp parallel num_threads(param_.preprocess_threads)
  {
    threadget = omp_get_num_threads();
  }
  param_.preprocess_threads = threadget;
#endif

  // use the kwarg to init parser
  parser_.Init(kwargs, param_.preprocess_threads);
  // input split
  source_.reset(dmlc::InputSplit::Create(
      param_.path_imgrec.c_str(), param_.part_index,
      param_.num_parts, "recordio"));
  // use 64 MB chunk when possible
  if (param_.shuffle_chunk_size > 0 && param_.shuffle) {
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
    source_->HintChunkSize(param_.shuffle_chunk_size << 20UL);
  } else {
    // use 4 MB chunk when possible
    source_->HintChunkSize(param_.chunk_size << 20UL);
  }

  iter_ = new dmlc::MultiThreadedIter<InstVecType, CommonChunkType>(
          CreateInputSplitThreadedIter(source_.get(),
                                       param_.source_chunk_queue_capacity,
                                       param_.chunk_read),
          param_.preprocess_threads,
          param_.record_chunk_queue_capacity);
  // init thread iter
  iter_->Init([this](InstVecType **dptr, CommonChunkType* chunk, int tid) {
                if (*dptr == nullptr) {
                *dptr = new InstVecType();
                }
                dmlc::InputSplit::Blob chunk_blob;
                chunk_blob.dptr = &(*chunk)[0];
                chunk_blob.size = chunk->size();
                return parser_.ParseNext(*dptr, chunk_blob, tid);
              },
              [this]() { parser_.BeforeFirst(); });
  inst_ptr_ = 0;
  rnd_.seed(kRandMagic + param_.seed);
  if (param_.verbose) {
    LOG(INFO) << "RecordDataInstIter: " << param_.path_imgrec
                  << ", use " << param_.preprocess_threads << " threads for decoding.. ";
  }
}

template <typename ParserType, typename InstVecType, typename InstType>
bool RecordDataInstIter<ParserType, InstVecType, InstType>::Next(void) {
  while (true) {
    if (inst_ptr_ < inst_order_.size()) {
      unsigned p = inst_order_[inst_ptr_];
      out_ = (*data_)[p];
      ++inst_ptr_;
      return true;
    } else {
      if (data_ != nullptr) iter_->Recycle(&data_);
      if (!iter_->Next(&data_)) return false;
      inst_order_.clear();
      for (unsigned i = 0; i < data_->Size(); ++i) {
        inst_order_.push_back(i);
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

// iterator on recordio
class RecordIter : public RecordDataInstIter<RecordIOParser,
    RecordInstVector, RecordInst> {};

/*! \brief typedef the factory function of record iterator */
typedef std::function<IIterator<RecordInst> *()> RecordIteratorFactory;
/*!
 * \brief Registry entry for record iterator factory functions.
 */
struct RecordIteratorReg
    : public dmlc::FunctionRegEntryBase<RecordIteratorReg,
                                        RecordIteratorFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Iterators
//--------------------------------------------------------------
/*!
 * \brief Macro to register Iterators
 *
 * \code
 * // example of registering a record iterator
 * MXNET_REGISTER_RECORD_ITER(RecordIter)
 * .describe("record data iterator")
 * .set_body([]() {
 *     return new RecordIter();
 *   });
 * \endcode
 */
#define MXNET_REGISTER_RECORD_ITER(name)                                    \
  DMLC_REGISTRY_REGISTER(::mxnet::io::RecordIteratorReg, RecordIteratorReg, name)

}  // namespace io
}  // namespace mxnet

#endif  // PB_FORMAT_REC

#endif  // MXNET_IO_ITER_RECORDIO_H_
