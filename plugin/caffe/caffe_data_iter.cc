/*!
 * Copyright (c) 2015 by Contributors
 * \file caffe_data_iter.cc
 * \brief register mnist iterator
*/
#include <caffe/proto/caffe.pb.h>
#include <atomic>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

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
  /*! \brief number of iterations per epoch */
  int epoch_size;
  /*! \brief data mode */
  bool flat;

  DMLC_DECLARE_PARAMETER(CaffeDataParam) {
    DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
      .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(flat).set_default(false)
      .describe("Augmentation Param: Whether to flat the data into 1D.");
    DMLC_DECLARE_FIELD(epoch_size).set_lower_bound(1).set_default(10000)
      .describe("Number of iterations for each epoch.");
  }
};

struct CaffeDataIterWrapperParam : dmlc::Parameter<CaffeDataIterWrapperParam> {
    /*! \brief data type */
    int dtype;

    DMLC_DECLARE_PARAMETER(CaffeDataIterWrapperParam) {
      DMLC_DECLARE_FIELD(dtype)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
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
    CHECK(caffe_data_layer_ != nullptr) << "Failed creating caffe data layer";
    const size_t top_size = param_.prototxt.top_size();
    if (top_size > 0) {
      if(top_size > NR_SUPPORTED_TOP_ITEMS) {
        LOG(WARNING) << "Too may \"top\" items, only two (one data, one label) are currently supported";
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

      if (param_.flat) {
        batch_data_.shape_ = mshadow::Shape2(batch_size_, width_ * height_);
      } else {
        batch_data_.shape_ = mxnet::TShape(top_[DATA]->shape().begin(), top_[DATA]->shape().end());
      }
      batch_data_.stride_ = batch_data_.size(batch_data_.shape_.ndim() - 1);
      out_.data.clear();
      if(top_.size() > LABEL) {
        batch_label_.shape_ = mxnet::TShape(top_[LABEL]->shape().begin(), top_[LABEL]->shape().end());
        batch_label_.stride_ = 1;
      }
      out_.batch_size = batch_size_;
    }
  }

  virtual void BeforeFirst(void) {
    loc_ = 0;
  }

  virtual bool Next(void) {
    // MxNet iterator is expected to return CPU-accessible memory
    if(::caffe::Caffe::mode() != ::caffe::Caffe::CPU) {
      ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
    }
    CHECK_EQ(::caffe::Caffe::mode(), ::caffe::Caffe::CPU);
    caffe_data_layer_->Forward(bottom_, top_);
    CHECK_GT(batch_size_, 0) << "batch size must be greater than zero";
    CHECK_EQ(out_.batch_size, batch_size_) << "Internal Error: batch size mismatch";

    if (loc_ + batch_size_ <= param_.epoch_size) {
      batch_data_.dptr_ = top_[DATA]->mutable_cpu_data();
      batch_label_.dptr_ = top_[LABEL]->mutable_cpu_data();

      out_.data.clear();
      if (param_.flat) {
        out_.data.push_back(batch_data_.FlatTo2D<cpu, Dtype>());
      } else {
        out_.data.push_back(batch_data_);
      }
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
  /*! \brief Blobs done so far */
  std::atomic<size_t>  loc_;
};  // class CaffeDataIter

class CaffeDataIterWrapper : public IIterator<DataBatch> {
 public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      switch (param_.dtype) {
        case mshadow::kFloat32:
          loader_.reset(new CaffeDataIter<float>());
          break;
        case mshadow::kFloat64:
          loader_.reset(new CaffeDataIter<double>());
          break;
        case mshadow::kFloat16:
          LOG(FATAL) << "float16 layer is not supported by caffe";
          return;
        default:
          LOG(FATAL) << "Unsupported type " << param_.dtype;
          return;
      }
      if(loader_) {
        loader_->Init(kwargs);
      }
    }
    virtual void BeforeFirst(void) {
      CHECK_NOTNULL(loader_.get());
      return loader_->BeforeFirst();
    }
    virtual bool Next(void) {
      CHECK_NOTNULL(loader_.get());

      if (!loader_->Next()) {
        return false;
      }

      const TBlobBatch& batch = loader_->Value();
      if (out_ == nullptr) {
        out_.reset(new DataBatch());
        out_->num_batch_padd = batch.num_batch_padd;
        out_->data.resize(batch.data.size());
        out_->index.resize(batch.batch_size);
        for (size_t i = 0; i < batch.data.size(); ++i) {
          out_->data.at(i) = NDArray(batch.data[i].shape_, Context::CPU());
        }
      }

      // make sure batch size didn't change unexpectedly
      CHECK(batch.data.size() == out_->data.size());

      // copy data over
      for (size_t i = 0, n = batch.data.size(); i < n; ++i) {
        CHECK_EQ(out_->data[i].shape(), batch.data[i].shape_);
        out_->data[i].data() = batch.data[i];
        out_->num_batch_padd = batch.num_batch_padd;
      }

      if (batch.inst_index) {
        std::copy(batch.inst_index, batch.inst_index + batch.batch_size, out_->index.begin());
      }
      return true;
    }
    virtual const DataBatch &Value(void) const {
      return *out_;
    }
 protected:
    /*! \brief Parameters for wrapper */
    CaffeDataIterWrapperParam param_;
    /*! \brief Created data iterator based upon type */
    std::unique_ptr<IIterator<TBlobBatch> > loader_;
    /*! \brief Batch of data object */
    std::unique_ptr<DataBatch> out_;
};

DMLC_REGISTER_PARAMETER(CaffeDataParam);
DMLC_REGISTER_PARAMETER(CaffeDataIterWrapperParam);

MXNET_REGISTER_IO_ITER(CaffeDataIter)
.describe("Create MxNet iterator for a Caffe data layer.")
.add_arguments(CaffeDataParam::__FIELDS__())
.add_arguments(CaffeDataIterWrapperParam::__FIELDS__())
.set_body([]() {
    return new CaffeDataIterWrapper();
});

}  // namespace io
}  // namespace mxnet

