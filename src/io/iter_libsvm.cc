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
 * \file iter_libsvm.cc
 * \brief define a LibSVM Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_sparse_prefetcher.h"
#include "./iter_sparse_batchloader.h"

namespace mxnet {
namespace io {
// LibSVM parameters
struct LibSVMIterParam : public dmlc::Parameter<LibSVMIterParam> {
  /*! \brief path to data libsvm file */
  std::string data_libsvm;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label libsvm file */
  std::string label_libsvm;
  /*! \brief label shape */
  TShape label_shape;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LibSVMIterParam) {
    DMLC_DECLARE_FIELD(data_libsvm)
        .describe("The input zero-base indexed LibSVM data file or a directory path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("The shape of one example.");
    DMLC_DECLARE_FIELD(label_libsvm).set_default("NULL")
        .describe("The input LibSVM label file or a directory path. "
                  "If NULL, all labels will be read from ``data_libsvm``.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
  }
};

class LibSVMIter: public SparseIIterator<DataInst> {
 public:
  LibSVMIter() {}
  virtual ~LibSVMIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    CHECK_EQ(param_.data_shape.ndim(), 1) << "dimension of data_shape is expected to be 1";
    CHECK_GT(param_.num_parts, 0) << "number of parts should be positive";
    CHECK_GE(param_.part_index, 0) << "part index should be non-negative";
    data_parser_.reset(dmlc::Parser<uint64_t>::Create(param_.data_libsvm.c_str(),
                                                      param_.part_index,
                                                      param_.num_parts, "libsvm"));
    if (param_.label_libsvm != "NULL") {
      label_parser_.reset(dmlc::Parser<uint64_t>::Create(param_.label_libsvm.c_str(),
                                                         param_.part_index,
                                                         param_.num_parts, "libsvm"));
      CHECK_GT(param_.label_shape.Size(), 1)
        << "label_shape is not expected to be (1,) when param_.label_libsvm is set.";
    } else {
      CHECK_EQ(param_.label_shape.Size(), 1)
        << "label_shape is expected to be (1,) when param_.label_libsvm is NULL";
    }
    // both data and label are of CSRStorage in libsvm format
    if (param_.label_shape.Size() > 1) {
      out_.data.resize(6);
    } else {
      // only data is of CSRStorage in libsvm format.
      out_.data.resize(4);
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
    const auto data_row = data_parser_->Value()[data_ptr_++];
    // data, indices and indptr
    out_.data[0] = AsDataBlob(data_row);
    out_.data[1] = AsIdxBlob(data_row);
    out_.data[2] = AsIndPtrPlaceholder(data_row);

    if (label_parser_.get() != nullptr) {
      while (label_ptr_ >= label_size_) {
        CHECK(label_parser_->Next())
            << "Data LibSVM's row is smaller than the number of rows in label_libsvm";
        label_ptr_ = 0;
        label_size_ = label_parser_->Value().size;
      }
      CHECK_LT(label_ptr_, label_size_);
      const auto label_row = label_parser_->Value()[label_ptr_++];
      // data, indices and indptr
      out_.data[3] = AsDataBlob(label_row);
      out_.data[4] = AsIdxBlob(label_row);
      out_.data[5] = AsIndPtrPlaceholder(label_row);
    } else {
      out_.data[3] = AsScalarLabelBlob(data_row);
    }
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    if (is_data) return kCSRStorage;
    return param_.label_shape.Size() > 1 ? kCSRStorage : kDefaultStorage;
  }

  virtual const TShape GetShape(bool is_data) const {
    if (is_data) return param_.data_shape;
    return param_.label_shape;
  }

 private:
  inline TBlob AsDataBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.value;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((real_t*) ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  inline TBlob AsIdxBlob(const dmlc::Row<uint64_t>& row) {
    const uint64_t* ptr = row.index;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((int64_t*) ptr, shape, cpu::kDevMask, mshadow::kInt64);  // NOLINT(*)
  }

  inline TBlob AsIndPtrPlaceholder(const dmlc::Row<uint64_t>& row) {
    return TBlob(nullptr, mshadow::Shape1(0), cpu::kDevMask, mshadow::kInt64);
  }

  inline TBlob AsScalarLabelBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.label;
    return TBlob((real_t*) ptr, mshadow::Shape1(1), cpu::kDevMask);  // NOLINT(*)
  }

  LibSVMIterParam param_;
  // output instance
  DataInst out_;
  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};
  // label parser
  size_t label_ptr_{0}, label_size_{0};
  size_t data_ptr_{0}, data_size_{0};
  std::unique_ptr<dmlc::Parser<uint64_t> > label_parser_;
  std::unique_ptr<dmlc::Parser<uint64_t> > data_parser_;
};


DMLC_REGISTER_PARAMETER(LibSVMIterParam);

MXNET_REGISTER_IO_ITER(LibSVMIter)
.describe(R"code(Returns the LibSVM iterator which returns data with `csr`
storage type. This iterator is experimental and should be used with care.

The input data is stored in a format similar to LibSVM file format, except that the **indices
are expected to be zero-based instead of one-based, and the column indices for each row are
expected to be sorted in ascending order**. Details of the LibSVM format are available
`here. <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>`_


The `data_shape` parameter is used to set the shape of each line of the data.
The dimension of both `data_shape` and `label_shape` are expected to be 1.

The `data_libsvm` parameter is used to set the path input LibSVM file.
When it is set to a directory, all the files in the directory will be read.

When `label_libsvm` is set to ``NULL``, both data and label are read from the file specified
by `data_libsvm`. In this case, the data is stored in `csr` storage type, while the label is a 1D
dense array.

The `LibSVMIter` only support `round_batch` parameter set to ``True``. Therefore, if `batch_size`
is 3 and there are 4 total rows in libsvm file, 2 more examples are consumed at the first round.

When `num_parts` and `part_index` are provided, the data is split into `num_parts` partitions,
and the iterator only reads the `part_index`-th partition. However, the partitions are not
guaranteed to be even.

``reset()`` is expected to be called only after a complete pass of data.

Example::

  # Contents of libsvm file ``data.t``.
  1.0 0:0.5 2:1.2
  -2.0
  -3.0 0:0.6 1:2.4 2:1.2
  4 2:-1.2

  # Creates a `LibSVMIter` with `batch_size`=3.
  >>> data_iter = mx.io.LibSVMIter(data_libsvm = 'data.t', data_shape = (3,), batch_size = 3)
  # The data of the first batch is stored in csr storage type
  >>> batch = data_iter.next()
  >>> csr = batch.data[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr.asnumpy()
  [[ 0.5        0.          1.2 ]
  [ 0.          0.          0.  ]
  [ 0.6         2.4         1.2]]
  # The label of first batch
  >>> label = batch.label[0]
  >>> label
  [ 1. -2. -3.]
  <NDArray 3 @cpu(0)>

  >>> second_batch = data_iter.next()
  # The data of the second batch
  >>> second_batch.data[0].asnumpy()
  [[ 0.          0.         -1.2 ]
   [ 0.5         0.          1.2 ]
   [ 0.          0.          0. ]]
  # The label of the second batch
  >>> second_batch.label[0].asnumpy()
  [ 4.  1. -2.]

  >>> data_iter.reset()
  # To restart the iterator for the second pass of the data

When `label_libsvm` is set to the path to another LibSVM file,
data is read from `data_libsvm` and label from `label_libsvm`.
In this case, both data and label are stored in the csr format.
If the label column in the `data_libsvm` file is ignored.

Example::

  # Contents of libsvm file ``label.t``
  1.0
  -2.0 0:0.125
  -3.0 2:1.2
  4 1:1.0 2:-1.2

  # Creates a `LibSVMIter` with specified label file
  >>> data_iter = mx.io.LibSVMIter(data_libsvm = 'data.t', data_shape = (3,),
                   label_libsvm = 'label.t', label_shape = (3,), batch_size = 3)

  # Both data and label are in csr storage type
  >>> batch = data_iter.next()
  >>> csr_data = batch.data[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr_data.asnumpy()
  [[ 0.5         0.          1.2  ]
   [ 0.          0.          0.   ]
   [ 0.6         2.4         1.2 ]]
  >>> csr_label = batch.label[0]
  <CSRNDArray 3x3 @cpu(0)>
  >>> csr_label.asnumpy()
  [[ 0.          0.          0.   ]
   [ 0.125       0.          0.   ]
   [ 0.          0.          1.2 ]]

)code" ADD_FILELINE)
.add_arguments(LibSVMIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new SparsePrefetcherIter(
        new SparseBatchLoader(
            new LibSVMIter()));
  });

}  // namespace io
}  // namespace mxnet
