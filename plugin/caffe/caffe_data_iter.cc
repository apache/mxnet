/*!
 * Copyright (c) 2015 by Contributors
 * \file caffe_data_iter.cc
 * \brief register mnist iterator
*/
#include <sys/time.h>
#include <caffe/proto/caffe.pb.h>
#include <dmlc/parameter.h>
#include <atomic>

#include "caffe_common.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"
#include "../../src/io/inst_vector.h"
#include "../../src/io/iter_prefetcher.h"

#define CHECK_NEXT_TIMING

#ifdef CHECK_NEXT_TIMING
#define IF_CHECK_TIMING(__t$) __t$
#else
#define IF_CHECK_TIMING(__t$)
#endif

namespace mxnet {
namespace io {

struct CaffeDataParam : public dmlc::Parameter<CaffeDataParam> {
  /*! \brief protobuf text */
  ::caffe::LayerParameter prototxt;
  /*! \brief number of iterations per epoch */
  int num_examples;
  /*! \brief data mode */
  bool flat;

  DMLC_DECLARE_PARAMETER(CaffeDataParam) {
    DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
      .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(flat).set_default(false)
      .describe("Augmentation Param: Whether to flat the data into 1D.");
    DMLC_DECLARE_FIELD(num_examples).set_lower_bound(1).set_default(10000)
      .describe("Number of examples in the epoch.");
  }
};

template<typename Dtype>
class CaffeDataIter : public IIterator<TBlobBatch> {
 public:
  explicit CaffeDataIter(int type_flag) : batch_size_(0), channels_(1), width_(1), height_(1)
                               , type_flag_(type_flag), loc_(0)
  {}
  virtual ~CaffeDataIter(void) {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::map<std::string, std::string> kmap(kwargs.begin(), kwargs.end());
    param_.InitAllowUnknown(kmap);

    // Caffe seems to understand phase inside an "include {}" block
    if (!param_.prototxt.has_phase()) {
      if (param_.prototxt.include().size()) {
        if (param_.prototxt.include(0).has_phase()) {
          param_.prototxt.set_phase(param_.prototxt.include(0).phase());
        }
      }
    }

    std::string type = param_.prototxt.type();
    caffe_data_layer_ = caffe::LayerRegistry<Dtype>::CreateLayer(param_.prototxt);
    CHECK(caffe_data_layer_ != nullptr) << "Failed creating caffe data layer";
    const size_t top_size = param_.prototxt.top_size();
    if (top_size > 0) {
      if (top_size > NR_SUPPORTED_TOP_ITEMS) {
        LOG(WARNING)
          << "Too may \"top\" items, only two (one data, one label) are currently supported";
      }
      top_.reserve(top_size);
      for (size_t x = 0; x < top_size; ++x) {
        ::caffe::Blob<Dtype> *blob = new ::caffe::Blob<Dtype>();
        cleanup_blobs_.push_back(std::unique_ptr<::caffe::Blob<Dtype>>(blob));
        top_.push_back(blob);
      }
      caffe_data_layer_->SetUp(bottom_, top_);
      const std::vector<int> &shape = top_[DATA]->shape();
      const size_t shapeDimCount = shape.size();
      if (shapeDimCount > 0) {
        batch_size_ = shape[0];
        if (shapeDimCount > 1) {
          channels_ = shape[1];
          if (shapeDimCount > 2) {
            width_ = shape[2];
            if (shapeDimCount > 3) {
              height_ = shape[3];
            }
          }
        }
      }

      if (top_size > DATA) {
        if (param_.flat) {
          batch_data_ = TBlob(nullptr, mshadow::Shape2(batch_size_,
                                                       channels_ * width_ * height_),
                              cpu::kDevCPU, type_flag_);
        } else {
          batch_data_ = TBlob(nullptr, mxnet::TShape(top_[DATA]->shape().begin(),
                                                     top_[DATA]->shape().end()),
                              cpu::kDevCPU, type_flag_);
        }
      }
      out_.data.clear();
      if (top_size > LABEL) {
          batch_label_ = TBlob(nullptr, mxnet::TShape(top_[LABEL]->shape().begin(),
                                                      top_[LABEL]->shape().end()),
                               cpu::kDevCPU, type_flag_);
      }
      out_.batch_size = batch_size_;
    }
  }

  virtual void BeforeFirst(void) {
    loc_ = 0;
  }

  virtual bool Next(void) {
    // MxNet iterator is expected to return CPU-accessible memory
    if (::caffe::Caffe::mode() != ::caffe::Caffe::CPU) {
      ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
      CHECK_EQ(::caffe::Caffe::mode(), ::caffe::Caffe::CPU);
    }
    caffe_data_layer_->Forward(bottom_, top_);
    CHECK_GT(batch_size_, 0) << "batch size must be greater than zero";
    CHECK_EQ(out_.batch_size, batch_size_) << "Internal Error: batch size mismatch";

    if (loc_ + batch_size_ <= param_.num_examples) {
      batch_data_.dptr_ = top_[DATA]->mutable_cpu_data();
      batch_label_.dptr_ = top_[LABEL]->mutable_cpu_data();

      out_.data.clear();
      out_.data.push_back(batch_data_);
      out_.data.push_back(batch_label_);
      loc_ += batch_size_;
      return true;
    }

    return false;
  }

  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 private:
  /*! \brief indexes into top_ */
  enum { DATA = 0, LABEL, NR_SUPPORTED_TOP_ITEMS };

  /*! \brief MNISTCass iter params */
  CaffeDataParam param_;
  /*! \brief Shape scalar values */
  index_t batch_size_, channels_, width_, height_;
  /*! \brief Caffe data layer */
  boost::shared_ptr<caffe::Layer<Dtype> >  caffe_data_layer_;
  /*! \brief batch data blob */
  mxnet::TBlob batch_data_;
  /*! \brief batch label blob */
  mxnet::TBlob batch_label_;
  /*! \brief Output blob data for this iteration */
  TBlobBatch out_;
  /*! \brief Bottom and top connection-point blob data */
  std::vector<::caffe::Blob<Dtype>*> bottom_, top_;
  /*! \brief Cleanup these blobs on exit */
  std::list<std::unique_ptr<::caffe::Blob<Dtype>>> cleanup_blobs_;
  /*! \brief type flag of the tensor blob */
  const int type_flag_;
  /*! \brief Blobs done so far */
  std::atomic<size_t>  loc_;
};  // class CaffeDataIter

class CaffeDataIterWrapper : public PrefetcherIter {
 public:
  CaffeDataIterWrapper() : PrefetcherIter(NULL), next_time_(0) {}
  virtual ~CaffeDataIterWrapper() {
    IF_CHECK_TIMING(
      if (next_time_.load() > 0) {
        LOG(WARNING) << "Caffe data loader was blocked for "
                     << next_time_.load()
                     << " ms waiting for incoming data";
      }
    )
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    // We need to init prefetcher args in order to get dtype
    this->param_.InitAllowUnknown(kwargs);
    if (!this->param_.dtype) this->param_.dtype = mshadow::kFloat32;
    switch (this->param_.dtype.value()) {
      case mshadow::kFloat32:
        this->loader_.reset(new CaffeDataIter<float>(this->param_.dtype.value()));
        break;
      case mshadow::kFloat64:
        this->loader_.reset(new CaffeDataIter<double>(this->param_.dtype.value()));
        break;
      case mshadow::kFloat16:
        LOG(FATAL) << "float16 layer is not supported by caffe";
        return;
      default:
        LOG(FATAL) << "Unsupported type " << this->param_.dtype.value();
        return;
    }
    PrefetcherIter::Init(kwargs);
    this->param_.prefetch_buffer = 1;
  }
  virtual void BeforeFirst(void) {
    return PrefetcherIter::BeforeFirst();
  }
  virtual bool Next(void) {
    IF_CHECK_TIMING(
      const uint64_t start_time = GetTickCountMS();
    )
    const bool rc = PrefetcherIter::Next();
    IF_CHECK_TIMING(
      const uint64_t diff_time  = GetTickCountMS() - start_time;
      next_time_.fetch_add(diff_time);
    )
    return rc;
  }

 protected:
  IF_CHECK_TIMING(
    static uint64_t GetTickCountMS() {
      struct timeval tv;
      gettimeofday(&tv, 0);
      return uint64_t( tv.tv_sec ) * 1000 + tv.tv_usec / 1000;
    }
  )

  /*! \brief milliseconds spent in Next() */
  std::atomic<uint64_t> next_time_;
};  // class CaffeDataIterWrapper

DMLC_REGISTER_PARAMETER(CaffeDataParam);

MXNET_REGISTER_IO_ITER(CaffeDataIter)
.describe("Create MxNet iterator for a Caffe data layer.")
.add_arguments(CaffeDataParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new CaffeDataIterWrapper();
});

}  // namespace io
}  // namespace mxnet

