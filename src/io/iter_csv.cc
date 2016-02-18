/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_csv.cc
 * \brief define a CSV Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_prefetcher.h"
#include "./iter_batchloader.h"

namespace mxnet {
namespace io {
// CSV parameters
struct CSVIterParam : public dmlc::Parameter<CSVIterParam> {
  /*! \brief path to data csv file */
  std::string data_csv;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label csv file */
  std::string label_csv;
  /*! \brief label shape */
  TShape label_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CSVIterParam) {
    DMLC_DECLARE_FIELD(data_csv)
        .describe("Dataset Param: Data csv path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("Dataset Param: Shape of the data.");
    DMLC_DECLARE_FIELD(label_csv).set_default("NULL")
        .describe("Dataset Param: Label csv path. If is NULL, all labels will be returned as 0");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("Dataset Param: Shape of the label.");
  }
};

class CSVIter: public IIterator<DataInst> {
 public:
  CSVIter() {
    out_.data.resize(2);
  }
  virtual ~CSVIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    data_parser_.reset(dmlc::Parser<uint32_t>::Create(param_.data_csv.c_str(), 0, 1, "csv"));
    if (param_.label_csv != "NULL") {
      label_parser_.reset(dmlc::Parser<uint32_t>::Create(param_.label_csv.c_str(), 0, 1, "csv"));
    } else {
      dummy_label.set_pad(false);
      dummy_label.Resize(mshadow::Shape1(1));
      dummy_label = 0.0f;
    }
  }

  virtual void BeforeFirst() {
    data_parser_->BeforeFirst();
    if (label_parser_.get() != nullptr) {
      label_parser_->BeforeFirst();
    }
    data_ptr_ = label_ptr_ = 0;
    data_size_ = label_size_ = 0;
    inst_counter_ = 0;
    end_ = false;
  }

  virtual bool Next() {
    if (end_) return false;
    while (data_ptr_ >= data_size_) {
      if (!data_parser_->Next()) {
        end_ = true; return false;
      }
      data_ptr_ = 0;
      data_size_ = data_parser_->Value().size;
    }
    out_.index = inst_counter_++;
    CHECK_LT(data_ptr_, data_size_);
    out_.data[0] = AsTBlob(data_parser_->Value()[data_ptr_++], param_.data_shape);

    if (label_parser_.get() != nullptr) {
      while (label_ptr_ >= label_size_) {
        CHECK(label_parser_->Next())
            << "Data CSV's row is smaller than the number of rows in label_csv";
        label_ptr_ = 0;
        label_size_ = label_parser_->Value().size;
      }
      CHECK_LT(label_ptr_, label_size_);
      out_.data[1] = AsTBlob(label_parser_->Value()[label_ptr_++], param_.label_shape);
    } else {
      out_.data[1] = dummy_label;
    }
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  inline TBlob AsTBlob(const dmlc::Row<uint32_t>& row, const TShape& shape) {
    CHECK_EQ(row.length, shape.Size())
        << "The data size in CSV do not match size of shape: "
        << "specified shape=" << shape << ", the csv row-length=" << row.length;
    const real_t* ptr = row.value;
    return TBlob((real_t*)ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  CSVIterParam param_;
  // output instance
  DataInst out_;
  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};
  // dummy label
  mshadow::TensorContainer<cpu, 1, real_t> dummy_label;
  // label parser
  size_t label_ptr_{0}, label_size_{0};
  size_t data_ptr_{0}, data_size_{0};
  std::unique_ptr<dmlc::Parser<uint32_t> > label_parser_;
  std::unique_ptr<dmlc::Parser<uint32_t> > data_parser_;
};


DMLC_REGISTER_PARAMETER(CSVIterParam);

MXNET_REGISTER_IO_ITER(CSVIter)
.describe("Create iterator for dataset in csv.")
.add_arguments(CSVIterParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new CSVIter()));
  });

}  // namespace io
}  // namespace mxnet
