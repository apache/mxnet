/*!
 * \file iter_batch_proc-inl.hpp
 * \brief definition of preprocessing iterators that takes an iterator and do some preprocessing
 * \author Tianqi Chen
 */
#ifndef MXNET_IO_ITER_BATCH_H_
#define MXNET_IO_ITER_BATCH_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>

namespace mxnet {
namespace io {
// Batch parameters
struct BatchParam : public dmlc::Parameter<BatchParam> {
  /*! \brief label width */
  index_t batch_size_;
  /*! \brief label width */
  index_t label_width_;
  /*! \brief use round roubin to handle overflow batch */
  bool round_batch_;
  /*! \brief skip read */
  bool test_skipread_;
  /*! \brief silent */
  bool silent_;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BatchParam) {
    DMLC_DECLARE_FIELD(batch_size_).set_default(1)
        .describe("Batch size.");
    DMLC_DECLARE_FIELD(label_width_).set_default(1)
        .describe("Label width.");
    DMLC_DECLARE_FIELD(round_batch_).set_default(false)
        .describe("Use round robin to handle overflow batch.");
    DMLC_DECLARE_FIELD(test_skipread_).set_default(false)
        .describe("Skip read for testing.");
    DMLC_DECLARE_FIELD(silent_).set_default(false)
        .describe("Whether to print batch information.");
  }
};
    
/*! \brief create a batch iterator from single instance iterator */
class BatchAdaptIter: public IIterator<DataBatch> {
public:
  BatchAdaptIter(IIterator<DataInst> *base): base_(base) {
    num_overflow_ = 0;
  }
  virtual ~BatchAdaptIter(void) {
    delete base_;
    FreeSpaceDense();
  }
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init batch param, it could have similar param with 
    kwargs_left = param_.InitAllowUnknown(kwargs);
    for (size_t i = 0; i < kwargs_left.size(); i++) {
      if (!strcmp(kwargs_left[i].first.c_str(), "input_shape")) {
        CHECK(sscanf(kwargs_left[i].second.c_str(), "%u,%u,%u", &shape_[1], &shape_[2], &shape_[3]) == 3)
          << "input_shape must be three consecutive integers without space example: 1,1,200 ";
      }
    }
    // init base iterator
    base_->Init(kwargs);
    mshadow::Shape<4> tshape = shape_;
    tshape[0] = param_.batch_size_;
    AllocSpaceDense(false);
  }
  virtual void BeforeFirst(void) {
    if (param_.round_batch_ == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }
  virtual bool Next(void) {
    out_.num_batch_padd = 0;

    // skip read if in head version
    if (param_.test_skipread_ != 0 && head_ == 0) return true;
    else this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      mshadow::Copy(label[top], d.data[1].get<mshadow::cpu, 1, float>());
      out_.inst_index[top] = d.index;
      mshadow::Copy(data[top], d.data[0].get<mshadow::cpu, 3, float>());

      if (++ top >= param_.batch_size_) {
        out_.data[0] = TBlob(data);
        out_.data[1] = TBlob(label);
        return true;
      }
    }
    if (top != 0) {
      if (param_.round_batch_ != 0) {
        num_overflow_ = 0;
        base_->BeforeFirst();
        for (; top < param_.batch_size_; ++top, ++num_overflow_) {
          CHECK(base_->Next()) << "number of input must be bigger than batch size";
          const DataInst& d = base_->Value();
          mshadow::Copy(label[top], d.data[1].get<mshadow::cpu, 1, float>());
          out_.inst_index[top] = d.index;
          mshadow::Copy(data[top], d.data[0].get<mshadow::cpu, 3, float>());
        }
        out_.num_batch_padd = num_overflow_;
      } else {
        out_.num_batch_padd = param_.batch_size_ - top;
      }
      out_.data[0] = TBlob(data);
      out_.data[1] = TBlob(label);
      return true;
    }
    return false;
  }
  virtual const DataBatch &Value(void) const {
    CHECK(head_ == 0) << "must call Next to get value";
    return out_;
  }
private:
  /*! \brief batch parameters */
  BatchParam param_;
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief input shape */
  mshadow::Shape<4> shape_;
  /*! \brief output data */
  DataBatch out_;
  /*! \brief on first */
  int head_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief label information of the data*/
  mshadow::Tensor<mshadow::cpu, 2> label;
  /*! \brief content of dense data, if this DataBatch is dense */
  mshadow::Tensor<mshadow::cpu, 4> data;
  // Functions that allocate and free tensor space
  inline void AllocSpaceDense(bool pad = false) { 
    data = mshadow::NewTensor<mshadow::cpu>(shape_, 0.0f, pad);
    mshadow::Shape<2> lshape = mshadow::Shape2(param_.batch_size_, param_.label_width_);
    label = mshadow::NewTensor<mshadow::cpu>(lshape, 0.0f, pad);
    out_.inst_index = new unsigned[param_.batch_size_];
    out_.batch_size = param_.batch_size_;
    out_.data.resize(2);
  }
  /*! \brief auxiliary function to free space, if needed, dense only */
  inline void FreeSpaceDense(void) {
    if (label.dptr_ != NULL) {
      delete [] out_.inst_index;
      mshadow::FreeSpace(&label);
      mshadow::FreeSpace(&data);
      label.dptr_ = NULL;
    }
  }
}; // class BatchAdaptIter
}  // namespace io
}  // namespace cxxnet
#endif  // MXNET_IO_ITER_BATCH_H_
