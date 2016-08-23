/*!
 * Copyright (c) 2015 by Contributors
 * \file caffe_data_iter.cc
 * \brief register mnist iterator
*/
#include <caffe/proto/caffe.pb.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <atomic>

#include "../../src/operator/operator_common.h"
#include "caffe_common.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"
#include "../../src/io/inst_vector.h"
#include "../../src/io/iter_prefetcher.h"
#include "../../src/operator/cast-inl.h"

namespace mxnet {
namespace io {

struct CaffeDataParam : public dmlc::Parameter<CaffeDataParam> {
  /*! \brief protobuf text */
  ::caffe::LayerParameter prototxt;
  /*! \brief maximum iteration count */
  int max_iterations;
  /*! \brief data mode */
  bool flat;

  DMLC_DECLARE_PARAMETER(CaffeDataParam) {
    DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
      .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(flat).set_default(false)
      .describe("Augmentation Param: Whether to flat the data into 1D.");
    DMLC_DECLARE_FIELD(max_iterations).set_lower_bound(1).set_default(10000)
      .describe("Maximum number of iterations per epoch.");
  }
};

struct CaffePrefetchParam : dmlc::Parameter<CaffePrefetchParam> {
    /*! \brief data type */
    int dtype;

    DMLC_DECLARE_PARAMETER(CaffePrefetchParam) {
      DMLC_DECLARE_FIELD(dtype)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("uint8", mshadow::kUint8)
        .add_enum("int32", mshadow::kInt32)
        .set_default(mshadow::default_type_flag)
        .describe("Data type.");
    }
};

template<typename Dtype>
class CaffeDataIter : public IIterator<TBlobBatch> {
 public:
  CaffeDataIter(void) : loc_(0), batch_size_(0), channels_(0), width_(0), height_(0) {}
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
    if (caffe_data_layer_) {
      const size_t top_size = param_.prototxt.top_size();
      if (top_size > 0) {
        top_.reserve(top_size);
        for (size_t x = 0; x < top_size; ++x) {
          ::caffe::Blob<Dtype> *blob = new ::caffe::Blob<Dtype>();
          cleanup_blobs_.push_back(std::unique_ptr<::caffe::Blob<Dtype>>(blob));
          top_.push_back(blob);
        }
        caffe_data_layer_->SetUp(bottom_, top_);
        const std::vector<int> &shape = top_[0]->shape();
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

        if (param_.flat) {
          batch_data_.shape_ = mshadow::Shape4(batch_size_, 1, 1, width_ * height_);
        } else {
          batch_data_.shape_ = mshadow::Shape4(batch_size_, channels_, width_, height_);
        }

        out_.data.clear();
        batch_label_.shape_ = mshadow::Shape2(batch_size_, 1);
        batch_label_.stride_ = 1;
        batch_data_.stride_ = batch_data_.size(3);
        out_.batch_size = batch_size_;
      }
    }
  }

  virtual void BeforeFirst(void) {
    loc_ = 0;
  }

  enum { DATA = 0, LABEL = 1 };

  virtual bool Next(void) {
    if (caffe_data_layer_) {
      // MxNet iterator is expected to return CPU-accessible memory
      assert(::caffe::Caffe::mode() == ::caffe::Caffe::CPU);
      caffe_data_layer_->Forward(bottom_, top_);
      assert(batch_size_ > 0);

      if (loc_ + batch_size_ <= param_.max_iterations) {
        batch_data_.dptr_ = top_[DATA]->mutable_cpu_data();
        batch_label_.dptr_ = top_[LABEL]->mutable_cpu_data();

        out_.data.clear();
        if (param_.flat) {
          out_.data.push_back(TBlob(batch_data_.FlatTo2D()));
        } else {
          out_.data.push_back(TBlob(batch_data_));
        }
        assert(out_.batch_size == batch_size_);
        out_.data.push_back(TBlob(batch_label_));
        loc_ += batch_size_;
        return true;
      }
    }
    return false;
  }

  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 private:
  /*! \brief MNISTCass iter params */
  CaffeDataParam param_;
  /*! Shape scalar values */
  size_t batch_size_, channels_, width_, height_;
  /*! \brief Caffe data layer */
  boost::shared_ptr<caffe::Layer<Dtype> >  caffe_data_layer_;
  /*! \brief batch data tensor */
  mshadow::Tensor<cpu, 4, Dtype> batch_data_;
  /*! \brief batch label tensor  */
  mshadow::Tensor<cpu, 2, Dtype> batch_label_;
  /*! \brief Output blob data for this iteration */
  TBlobBatch out_;
  /*! \brief Bottom and top connection-point blob data */
  std::vector<::caffe::Blob<Dtype>*> bottom_, top_;
  /*! \brief Cleanup these blobs on exit */
  std::list<std::unique_ptr<::caffe::Blob<Dtype>>> cleanup_blobs_;
  /*! \brief Blobs done so far */
  std::atomic<size_t>  loc_;
};  // class CaffeDataIter

class CaffeDataPrefetcherIter : public PrefetcherIter {
 public:
    CaffeDataPrefetcherIter() : PrefetcherIter(NULL) {}
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      switch (param_.dtype) {
        case mshadow::kFloat32:
          this->loader_.reset(new CaffeDataIter<float>());
          break;
        case mshadow::kFloat64:
          this->loader_.reset(new CaffeDataIter<double>());
          break;
        case mshadow::kFloat16:
          LOG(FATAL) << "float16 layer is not supported by caffe";
          return;
        default:
          LOG(FATAL) << "Unsupported type " << param_.dtype;
          return;
      }
      PrefetcherIter::Init(kwargs);
    }

 protected:
    CaffePrefetchParam param_;
};

DMLC_REGISTER_PARAMETER(CaffeDataParam);
DMLC_REGISTER_PARAMETER(CaffePrefetchParam);

MXNET_REGISTER_IO_ITER(CaffeDataIter)
.describe("Create MxNet iterator for a Caffe data layer.")
.add_arguments(CaffeDataParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(CaffePrefetchParam::__FIELDS__())
.set_body([]() {
    return new CaffeDataPrefetcherIter();
});

}  // namespace io
}  // namespace mxnet

