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
#include "./inst_vector.h"

namespace mxnet {
namespace io {
// Define prefetcher parameters
struct PrefetcherParam : public dmlc::Parameter<PrefetcherParam> {
  /*! \brief number of prefetched batches */
  size_t prefetch_buffer;
  /*! \brief label width */
  index_t batch_size;
  /*! \brief input shape */
  TShape input_shape;
  /*! \brief label width */
  index_t label_width;
  // declare parameters
  DMLC_DECLARE_PARAMETER(PrefetcherParam) {
    DMLC_DECLARE_FIELD(prefetch_buffer).set_default(1)
        .describe("Backend Param: Number of prefetched batches.");
    DMLC_DECLARE_FIELD(batch_size)
        .describe("Batch Param: Batch size.");
    index_t input_shape_default[] = {3, 224, 224};
    DMLC_DECLARE_FIELD(input_shape)
        .set_default(TShape(input_shape_default, input_shape_default + 3))
        .enforce_nonzero()
        .describe("Dataset Param: Input shape of the neural net.");
    DMLC_DECLARE_FIELD(label_width).set_default(1)
        .describe("Dataset Param: Label width.");
  }
};

// iterator on image recordio
class PrefetcherIter : public IIterator<DataBatch> {
 public:
  explicit PrefetcherIter(IIterator<TBlobBatch>* base) : loader_(base) {
    pdata_vec.clear();
    plabel_vec.clear();
  }
  virtual ~PrefetcherIter(void) {
    iter_.Destroy();
    for (size_t i = 0; i < pdata_vec.size(); i++) {
      delete[] pdata_vec[i];
      delete[] plabel_vec[i];
    }
    delete loader_;
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init image rec param
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // use the kwarg to init batch loader
    loader_->Init(kwargs);
    // create the shape
    std::vector<size_t> data_shape_vec;
    data_shape_vec.push_back(param_.batch_size);
    for (size_t shape_dim = 0; shape_dim < param_.input_shape.ndim(); shape_dim++)
        data_shape_vec.push_back(param_.input_shape[shape_dim]);
    data_shape_ = TShape(data_shape_vec.begin(), data_shape_vec.end());
    std::vector<size_t> label_shape_vec;
    label_shape_vec.push_back(param_.batch_size);
    label_shape_vec.push_back(param_.label_width);
    label_shape_ = TShape(label_shape_vec.begin(), label_shape_vec.end());
    // init thread iter
    iter_.set_max_capacity(param_.prefetch_buffer);
    iter_.Init([this](TBlobBatch **dptr) {
        bool load_success = loader_->Next();
        if (load_success == false)
          return false;
        if (*dptr == NULL) {
          *dptr = new TBlobBatch();
          // create the spaces and record the pointers
          real_t* pdata = new real_t[data_shape_.Size()];
          pdata_vec.push_back(pdata);
          real_t* plabel = new real_t[label_shape_.Size()];
          plabel_vec.push_back(plabel);
          (*dptr)->data.push_back(TBlob(pdata, data_shape_, mshadow::cpu::kDevMask));
          (*dptr)->data.push_back(TBlob(plabel, label_shape_, mshadow::cpu::kDevMask));
        }
        const TBlobBatch& batch = loader_->Value();
        if (data_shape_.ndim() == 4) {
          mshadow::Copy((*dptr)->data[0].get<mshadow::cpu, 4, float>(),
                  batch.data[0].get<mshadow::cpu, 4, float>());
        } else if (data_shape_.ndim() == 2) {
          mshadow::Copy((*dptr)->data[0].get<mshadow::cpu, 2, float>(),
                  batch.data[0].get<mshadow::cpu, 2, float>());
        } else {
          // TODO(tianjun): ?
          LOG(FATAL) << "fail";
        }
        mshadow::Copy((*dptr)->data[1].get<mshadow::cpu, 2, float>(),
                batch.data[1].get<mshadow::cpu, 2, float>());
        return load_success;
      },
      [this]() { loader_->BeforeFirst(); });
  }
  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
  }
  virtual bool Next(void) {
     if (ready_batches_.size() == param_.prefetch_buffer) {
         TBlobBatch* old_batch = ready_batches_.front();
         for (size_t i = 0; i < old_batch->data.size(); i++) {
             NDArray old_ndarray = ready_ndarrays_.front();
             old_ndarray.WaitToWrite();
             ready_ndarrays_.pop();
         }
         iter_.Recycle(&old_batch);
         ready_batches_.pop();
     }
     TBlobBatch* next_batch = NULL;
     if (!iter_.Next(&next_batch)) return false;
     out_.data.clear();
     // copy the batch
     for (size_t i = 0; i < next_batch->data.size(); i++) {
         out_.data.push_back(NDArray(next_batch->data[i], mshadow::cpu::kDevMask));
         ready_ndarrays_.push(out_.data[i]);
     }
     // push the narrays and batch into the queue
     ready_batches_.push(next_batch);
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
  /*! \brief batch holder */
  TBlobBatch out_holder_;
  /*! \brief queue to hold the NDArrays for check whether writable */
  std::queue<TBlobBatch*> ready_batches_;
  /*! \breif ndarrays to wait to write */
  std::queue<NDArray> ready_ndarrays_;
  // internal batch loader
  IIterator<TBlobBatch>* loader_;
  // backend thread
  dmlc::ThreadedIter<TBlobBatch> iter_;
  /*! \brief data shape */
  TShape data_shape_;
  /*! \brief label shape */
  TShape label_shape_;
  /*! \brief log the pointers of the space created for data*/
  std::vector<real_t*> pdata_vec;
  /*! \brief log the pointers of the space created for label*/
  std::vector<real_t*> plabel_vec;
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_PREFETCHER_H_
