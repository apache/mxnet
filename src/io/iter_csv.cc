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
        .describe("The input CSV file or a directory path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("The shape of one example.");
    DMLC_DECLARE_FIELD(label_csv).set_default("NULL")
        .describe("The input CSV file or a directory path. "
                  "If NULL, all labels will be returned as 0.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
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
.describe(R"code(Returns the CSV file iterator.

In this function, the `data_shape` parameter is used to set the shape of each line of the input data.
If a row in an input file is `1,2,3,4,5,6`` and `data_shape` is (3,2), that row
will be reshaped, yielding the array [[1,2],[3,4],[5,6]] of shape (3,2).

By default, the `CSVIter` has `round_batch` parameter set to ``True``. So, if `batch_size`
is 3 and there are 4 total rows in CSV file, 2 more examples
are consumed at the first round. If `reset` function is called after first round,
the call is ignored and remaining examples are returned in the second round.

If one wants all the instances in the second round after calling `reset`, make sure
to set `round_batch` to False.

If ``data_csv = 'data/'`` is set, then all the files in this directory will be read.

Examples::

  // Contents of CSV file ``data/data.csv``.
  1,2,3
  2,3,4
  3,4,5
  4,5,6

  // Creates a `CSVIter` with `batch_size`=2 and default `round_batch`=True.
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 2)

  // Two batches read from the above iterator are as follows:
  [[ 1.  2.  3.]
  [ 2.  3.  4.]]
  [[ 3.  4.  5.]
  [ 4.  5.  6.]]

  // Creates a `CSVIter` with `round_batch` set to False.
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 3)

  // Two batches read from the above iterator in the first pass are as follows:
  [[1.  2.  3.]
  [2.  3.  4.]
  [3.  4.  5.]]

  [[4.  5.  6.]
  [2.  3.  4.]
  [3.  4.  5.]]

  // Now, `reset` method is called.
  CSVIter.reset()

  // Batch read from the above iterator in the second pass is as follows:
  [[ 3.  4.  5.]
  [ 4.  5.  6.]
  [ 1.  2.  3.]]

  // Creates a `CSVIter` with `round_batch`=False.
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 3, round_batch=True)

  // Contents of two batches read from the above iterator in both passes after calling
  // `reset` method before second pass is as follows:
  [[1.  2.  3.]
  [2.  3.  4.]
  [3.  4.  5.]]

  [[4.  5.  6.]
  [2.  3.  4.]
  [3.  4.  5.]]

)code" ADD_FILELINE)
.add_arguments(CSVIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new CSVIter()));
  });

}  // namespace io
}  // namespace mxnet
