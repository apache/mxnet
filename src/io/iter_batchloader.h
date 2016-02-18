/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_batchloader.h
 * \brief define a batch adapter to create tblob batch
 */
#ifndef MXNET_IO_ITER_BATCHLOADER_H_
#define MXNET_IO_ITER_BATCHLOADER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <utility>
#include <vector>
#include <string>
#include "./inst_vector.h"

namespace mxnet {
namespace io {
// Batch parameters
struct BatchParam : public dmlc::Parameter<BatchParam> {
  /*! \brief label width */
  index_t batch_size;
  /*! \brief use round roubin to handle overflow batch */
  bool round_batch;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BatchParam) {
    DMLC_DECLARE_FIELD(batch_size)
        .describe("Batch Param: Batch size.");
    DMLC_DECLARE_FIELD(round_batch).set_default(true)
        .describe("Batch Param: Use round robin to handle overflow batch.");
  }
};

/*! \brief create a batch iterator from single instance iterator */
class BatchLoader : public IIterator<TBlobBatch> {
 public:
  explicit BatchLoader(IIterator<DataInst> *base):
      base_(base), head_(1), num_overflow_(0) {
  }

  virtual ~BatchLoader(void) {
    delete base_;
  }

  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init batch param, it could have similar param with
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // Init space for out_
    out_.inst_index = new unsigned[param_.batch_size];
    out_.batch_size = param_.batch_size;
    out_.data.clear();
    // init base iterator
    base_->Init(kwargs);
  }

  virtual void BeforeFirst(void) {
    if (param_.round_batch == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }
  virtual bool Next(void) {
    out_.num_batch_padd = 0;
    out_.batch_size = param_.batch_size;
    this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      out_.inst_index[top] = d.index;
      if (data_.size() == 0) {
        this->InitData(d);
      }
      for (size_t i = 0; i < d.data.size(); ++i) {
        CHECK_EQ(unit_size_[i], d.data[i].Size());
        mshadow::Copy(data_[i].Slice(top * unit_size_[i], (top + 1) * unit_size_[i]),
                      d.data[i].get_with_shape<cpu, 1, real_t>(mshadow::Shape1(unit_size_[i])));
      }
      if (++top >= param_.batch_size) {
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
          out_.inst_index[top] = d.index;
          // copy data
          for (size_t i = 0; i < d.data.size(); ++i) {
            CHECK_EQ(unit_size_[i], d.data[i].Size());
            mshadow::Copy(data_[i].Slice(top * unit_size_[i], (top + 1) * unit_size_[i]),
                          d.data[i].get_with_shape<cpu, 1, real_t>(mshadow::Shape1(unit_size_[i])));
          }
        }
        out_.num_batch_padd = num_overflow_;
      } else {
        out_.num_batch_padd = param_.batch_size - top;
      }
      return true;
    }
    return false;
  }
  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 private:
  /*! \brief batch parameters */
  BatchParam param_;
  /*! \brief output data */
  TBlobBatch out_;
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief on first */
  int head_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief data shape */
  std::vector<TShape> shape_;
  /*! \brief unit size */
  std::vector<size_t> unit_size_;
  /*! \brief tensor to hold data */
  std::vector<mshadow::TensorContainer<mshadow::cpu, 1, real_t> > data_;
  // initialize the data holder by using from the first batch.
  inline void InitData(const DataInst& first_batch) {
    shape_.resize(first_batch.data.size());
    data_.resize(first_batch.data.size());
    unit_size_.resize(first_batch.data.size());
    for (size_t i = 0; i < first_batch.data.size(); ++i) {
      TShape src_shape = first_batch.data[i].shape_;
      // init object attributes
      std::vector<index_t> shape_vec;
      shape_vec.push_back(param_.batch_size);
      for (index_t dim = 0; dim < src_shape.ndim(); ++dim) {
        shape_vec.push_back(src_shape[dim]);
      }
      TShape dst_shape(shape_vec.begin(), shape_vec.end());
      shape_[i] = dst_shape;
      data_[i].Resize(mshadow::Shape1(dst_shape.Size()));
      unit_size_[i] = src_shape.Size();
      out_.data.push_back(TBlob(data_[i].dptr_, dst_shape, cpu::kDevMask));
    }
  }
};  // class BatchLoader
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_BATCHLOADER_H_
