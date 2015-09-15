/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_prefetcher.h
 * \brief define a prefetcher using threaditer to keep k batch fetched 
 */
#ifndef MXNET_IO_ITER_PREFETCHER_H_
#define MXNET_IO_ITER_PREFETCHER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <dmlc/logging.h>
#include <dmlc/threadediter.h>
#include <mshadow/tensor.h>
#include <utility>
#include <string>
#include <vector>
#include <queue>

namespace mxnet {
namespace io {
// Batch parameters
struct BatchParam : public dmlc::Parameter<BatchParam> {
  /*! \brief label width */
  index_t batch_size;
  /*! \brief input shape */
  TShape input_shape;
  /*! \brief label width */
  index_t label_width;
  /*! \brief use round roubin to handle overflow batch */
  bool round_batch;
  /*! \brief skip read */
  bool test_skipread;
  /*! \brief silent */
  bool silent;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BatchParam) {
    DMLC_DECLARE_FIELD(batch_size)
        .describe("Batch size.");
    index_t input_shape_default[] = {3, 224, 224};
    DMLC_DECLARE_FIELD(input_shape)
        .set_default(TShape(input_shape_default, input_shape_default + 3))
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Input shape of the neural net");
    DMLC_DECLARE_FIELD(label_width).set_default(1)
        .describe("Label width.");
    DMLC_DECLARE_FIELD(round_batch).set_default(true)
        .describe("Use round robin to handle overflow batch.");
    DMLC_DECLARE_FIELD(test_skipread).set_default(false)
        .describe("Skip read for testing.");
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Whether to print batch information.");
  }
};

/*! \brief create a batch iterator from single instance iterator */
class BatchLoader {
 public:
  explicit BatchLoader(IIterator<DataInst> *base): base_(base), num_overflow_(0) {}
  virtual ~BatchLoader(void) {
    delete base_;
  }
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init batch param, it could have similar param with
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // init base iterator
    base_->Init(kwargs);
    std::vector<size_t> data_shape_vec;
    data_shape_vec.push_back(param_.batch_size);
    data_shape_vec.push_back(param_.input_shape[0]);
    data_shape_vec.push_back(param_.input_shape[1]);
    data_shape_vec.push_back(param_.input_shape[2]);
    data_shape_ = TShape(data_shape_vec.begin(), data_shape_vec.end());
    std::vector<size_t> label_shape_vec;
    label_shape_vec.push_back(param_.batch_size);
    label_shape_vec.push_back(param_.label_width);
    label_shape_ = TShape(label_shape_vec.begin(), label_shape_vec.end());
  }
  inline void BeforeFirst(void) {
    if (param_.round_batch == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }
  inline bool LoadNext(DataBatch* out) {
    out->num_batch_padd = 0;

    // skip read if in head version
    if (param_.test_skipread != 0 && head_ == 0)
        return true;
    else
        this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      out->inst_index[top] = d.index;
      mshadow::Copy(out->data[1].data().get<mshadow::cpu, 2, float>()[top],
              d.data[1].get<mshadow::cpu, 1, float>());
      mshadow::Copy(out->data[0].data().get<mshadow::cpu, 4, float>()[top],
              d.data[0].get<mshadow::cpu, 3, float>());

      if (++ top >= param_.batch_size) {
        return true;
      }
    }
    if (top != 0) {
      if (param_.round_batch != 0) {
        num_overflow_ = 0;
        base_->BeforeFirst();
        for (; top < param_.batch_size; ++top, ++num_overflow_) {
          CHECK(base_->Next()) << "number of input must be bigger than batch size";
          const DataInst& d = base_->Value();
          out->inst_index[top] = d.index;
          mshadow::Copy(out->data[1].data().get<mshadow::cpu, 2, float>()[top],
                  d.data[1].get<mshadow::cpu, 1, float>());
          mshadow::Copy(out->data[0].data().get<mshadow::cpu, 4, float>()[top],
                  d.data[0].get<mshadow::cpu, 3, float>());
        }
        out->num_batch_padd = num_overflow_;
      } else {
        out->num_batch_padd = param_.batch_size - top;
      }
      return true;
    }
    return false;
  }

 private:
  /*! \brief batch parameters */
  BatchParam param_;
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief on first */
  int head_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief data shape */
  TShape data_shape_;
  /*! \brief label shape */
  TShape label_shape_;
};  // class BatchLoader
    
// Define prefetcher parameters
struct PrefetcherParam : public dmlc::Parameter<PrefetcherParam> {
  /*! \brief number of prefetched batches */
  size_t capacity;
    /*! \brief label width */
  index_t batch_size;
  /*! \brief input shape */
  TShape input_shape;
  /*! \brief label width */
  index_t label_width;
  // declare parameters
  DMLC_DECLARE_PARAMETER(PrefetcherParam) {
    DMLC_DECLARE_FIELD(capacity).set_default(1)
        .describe("Number of prefetched batches");
    index_t input_shape_default[] = {3, 224, 224};
    DMLC_DECLARE_FIELD(input_shape)
        .set_default(TShape(input_shape_default, input_shape_default + 3))
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Input shape of the neural net");
    DMLC_DECLARE_FIELD(label_width).set_default(1)
        .describe("Label width.");
  }
};
  
// iterator on image recordio
class PrefetcherIter : public IIterator<DataBatch> {
 public:
  PrefetcherIter(IIterator<DataInst>* base) : loader_(base){
  }
  virtual ~PrefetcherIter(void) {
    iter_.Destroy();
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init image rec param
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // use the kwarg to init batch loader
    loader_.Init(kwargs);
    std::vector<size_t> data_shape_vec;
    data_shape_vec.push_back(param_.batch_size);
    data_shape_vec.push_back(param_.input_shape[0]);
    data_shape_vec.push_back(param_.input_shape[1]);
    data_shape_vec.push_back(param_.input_shape[2]);
    data_shape_ = TShape(data_shape_vec.begin(),data_shape_vec.end());
    std::vector<size_t> label_shape_vec;
    label_shape_vec.push_back(param_.batch_size);
    label_shape_vec.push_back(param_.label_width);
    label_shape_ = TShape(label_shape_vec.begin(), label_shape_vec.end());
    // init thread iter
    iter_.set_max_capacity(param_.capacity);
    iter_.Init([this](DataBatch **dptr) {
        if (*dptr == NULL) {
          *dptr = new DataBatch();
          // init NDArrays
          Context ctx; 
          (*dptr)->data.push_back(NDArray(data_shape_, ctx, true));
          (*dptr)->data.push_back(NDArray(label_shape_, ctx, true));
        }
        return loader_.LoadNext(*dptr);
      },
      [this]() { loader_.BeforeFirst(); });
  }
  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
  }
  virtual bool Next(void) {
     if (ready_narrays_.size() == param_.capacity) {
         std::vector<NDArray*> old_narrays = ready_narrays_.front();
         for (size_t i = 0; i < old_narrays.size(); i++) {
             old_narrays[i]->WaitToWrite();
         }
         ready_narrays_.pop();
         DataBatch* old_batch = ready_batches_.front();
         ready_batches_.pop();
         iter_.Recycle(&old_batch);
     }
     DataBatch* next_batch = NULL;
     if (!iter_.Next(&next_batch)) return false;
     out_.data.clear();
     // copy the batch
     for (size_t i = 0; i < next_batch->data.size(); i++) {
         out_.data.push_back(next_batch->data[i].Copy(next_batch->data[i].ctx()));
     }
     // push the narrays and batch into the queue
     ready_batches_.push(next_batch);
     std::vector<NDArray*> next_batch_narrays;
     for (size_t i = 0; i < out_.data.size(); i++) {
         next_batch_narrays.push_back(&out_.data[i]);
     }
     ready_narrays_.push(next_batch_narrays);
     return true;
  }
  virtual const DataBatch &Value(void) const {
    return out_;
  }
 private:
  /*! \brief prefetcher parameters */
  PrefetcherParam param_;
  /*! \brief output data */
  DataBatch out_;
  /*! \brief queue to hold the NDArrays for check whether writable */
  std::queue<std::vector<NDArray*> > ready_narrays_;
  /*! \brief queue to hold the NDArrays for check whether writable */
  std::queue<DataBatch*> ready_batches_;
  // internal batch loader
  BatchLoader loader_;
  // backend thread
  dmlc::ThreadedIter<DataBatch> iter_;
  /*! \brief data shape */
  TShape data_shape_;
  /*! \brief label shape */
  TShape label_shape_;
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_PREFETCHER_H_
