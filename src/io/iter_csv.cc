/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
  mxnet::TShape data_shape;
  /*! \brief path to label csv file */
  std::string label_csv;
  /*! \brief label shape */
  mxnet::TShape label_shape;
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
    DMLC_DECLARE_FIELD(label_shape).set_default(mxnet::TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
  }
};

class CSVIterBase: public IIterator<DataInst> {
 public:
  CSVIterBase() {
    out_.data.resize(2);
  }
  ~CSVIterBase() override = default;

  // initialize iterator loads data in
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override = 0;
  /*! \brief reset the iterator */
  void BeforeFirst() override = 0;
  /*! \brief move to next item */
  bool Next() override = 0;
  /*! \brief get current data */
  const DataInst &Value() const override {
    return out_;
  }

 protected:
  CSVIterParam param_;

  DataInst out_;

  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};

  // label parser
  size_t label_ptr_{0}, label_size_{0};
  size_t data_ptr_{0}, data_size_{0};
};

template <typename DType>
class CSVIterTyped: public CSVIterBase {
 public:
  ~CSVIterTyped() override = default;
  // intialize iterator loads data in
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.InitAllowUnknown(kwargs);
    data_parser_.reset(dmlc::Parser<uint32_t, DType>::Create(param_.data_csv.c_str(), 0, 1, "csv"));
    if (param_.label_csv != "NULL") {
      label_parser_.reset(
        dmlc::Parser<uint32_t, DType>::Create(param_.label_csv.c_str(), 0, 1, "csv"));
    } else {
      dummy_label.set_pad(false);
      dummy_label.Resize(mshadow::Shape1(1));
      dummy_label = 0;
    }
  }

  void BeforeFirst() override {
    data_parser_->BeforeFirst();
    if (label_parser_.get() != nullptr) {
      label_parser_->BeforeFirst();
    }
    data_ptr_ = label_ptr_ = 0;
    data_size_ = label_size_ = 0;
    inst_counter_ = 0;
    end_ = false;
  }

  bool Next() override {
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

 private:
  inline TBlob AsTBlob(const dmlc::Row<uint32_t, DType>& row, const mxnet::TShape& shape) {
    CHECK_EQ(row.length, shape.Size())
        << "The data size in CSV do not match size of shape: "
        << "specified shape=" << shape << ", the csv row-length=" << row.length;
    const DType* ptr = row.value;
    return TBlob((DType*)ptr, shape, cpu::kDevMask, 0);  // NOLINT(*)
  }
  // dummy label
  mshadow::TensorContainer<cpu, 1, DType> dummy_label;
  std::unique_ptr<dmlc::Parser<uint32_t, DType> > label_parser_;
  std::unique_ptr<dmlc::Parser<uint32_t, DType> > data_parser_;
};

class CSVIter: public IIterator<DataInst> {
 public:
  CSVIter() = default;
  ~CSVIter() override = default;

  // intialize iterator loads data in
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.InitAllowUnknown(kwargs);
    bool dtype_has_value = false;
    int target_dtype = -1;
    for (const auto& arg : kwargs) {
      if (arg.first == "dtype") {
        dtype_has_value = true;
        if (arg.second == "int32") {
          target_dtype = mshadow::kInt32;
        } else if (arg.second == "int64") {
          target_dtype = mshadow::kInt64;
        } else if (arg.second == "float32") {
          target_dtype = mshadow::kFloat32;
        } else {
          CHECK(false) << arg.second << " is not supported for CSVIter";
        }
      }
    }
    if (dtype_has_value && target_dtype == mshadow::kInt32) {
      iterator_.reset(reinterpret_cast<CSVIterBase*>(new CSVIterTyped<int32_t>()));
    } else if (dtype_has_value && target_dtype == mshadow::kInt64) {
      iterator_.reset(reinterpret_cast<CSVIterBase*>(new CSVIterTyped<int64_t>()));
    } else if (!dtype_has_value || target_dtype == mshadow::kFloat32) {
      iterator_.reset(reinterpret_cast<CSVIterBase*>(new CSVIterTyped<float>()));
    }
    iterator_->Init(kwargs);
  }

  void BeforeFirst() override {
    iterator_->BeforeFirst();
  }

  bool Next() override {
    return iterator_->Next();
  }

  const DataInst &Value() const override {
    return iterator_->Value();
  }

 private:
  CSVIterParam param_;
  std::unique_ptr<CSVIterBase> iterator_;
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

``reset()`` is expected to be called only after a complete pass of data.

By default, the CSVIter parses all entries in the data file as float32 data type,
if `dtype` argument is set to be 'int32' or 'int64' then CSVIter will parse all entries in the file
as int32 or int64 data type accordingly.

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

  // Creates a `CSVIter` with default `round_batch` set to True.
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 3)

  // Two batches read from the above iterator in the first pass are as follows:
  [[1.  2.  3.]
  [2.  3.  4.]
  [3.  4.  5.]]

  [[4.  5.  6.]
  [1.  2.  3.]
  [2.  3.  4.]]

  // Now, `reset` method is called.
  CSVIter.reset()

  // Batch read from the above iterator in the second pass is as follows:
  [[ 3.  4.  5.]
  [ 4.  5.  6.]
  [ 1.  2.  3.]]

  // Creates a `CSVIter` with `round_batch`=False.
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 3, round_batch=False)

  // Contents of two batches read from the above iterator in both passes, after calling
  // `reset` method before second pass, is as follows:
  [[1.  2.  3.]
  [2.  3.  4.]
  [3.  4.  5.]]

  [[4.  5.  6.]
  [2.  3.  4.]
  [3.  4.  5.]]

  // Creates a 'CSVIter' with `dtype`='int32'
  CSVIter = mx.io.CSVIter(data_csv = 'data/data.csv', data_shape = (3,),
  batch_size = 3, round_batch=False, dtype='int32')

  // Contents of two batches read from the above iterator in both passes, after calling
  // `reset` method before second pass, is as follows:
  [[1  2  3]
  [2  3  4]
  [3  4  5]]

  [[4  5  6]
  [2  3  4]
  [3  4  5]]

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
