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
 * \file iter_rmf.cc
 * \brief define a Recommendation Multi-Format Feature Sample Reader to read in arrays
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
// RMF parameters
struct RMFIterParam : public dmlc::Parameter<RMFIterParam> {
  /*! \brief path to data RMF file */
  std::string data;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief dense shape */
  TShape dense_shape;
  /*! \brief cate shape */
  TShape cate_shape;
  /*! \brief sparse shape */
  TShape sparse_shape;
  /*! \brief label shape */
  TShape label_shape;
  /*! \brief sparse shape */
  std::vector<TShape> multi_field_shape;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  /*! \brief the number of multiple field feature*/
  size_t multi_field_num;
  /*! \brief the number of label*/
  int label_width;
  std::string multi_field_shape_str;
  // declare parameters 
  DMLC_DECLARE_PARAMETER(RMFIterParam) {
    DMLC_DECLARE_FIELD(label_width).set_default(1)
        .describe("the number of label");
    DMLC_DECLARE_FIELD(data)
        .describe("The input zero-base indexed RMF data file or a directory path.");
    index_t shape4[] = {6};
    DMLC_DECLARE_FIELD(data_shape).set_default(TShape(shape4, shape4 + 1))
        .describe("The shape of data feature for one example.");
    index_t shape3[] = {12};
    DMLC_DECLARE_FIELD(dense_shape).set_default(TShape(shape3, shape3 + 1))
        .describe("The shape of dense feature for one example.");
    index_t shape2[] = {6};
    DMLC_DECLARE_FIELD(cate_shape).set_default(TShape(shape2, shape2 + 1))
        .describe("The shape of cate feature for one example.");
    index_t shape0[] = {10000000};
    DMLC_DECLARE_FIELD(sparse_shape).set_default(TShape(shape0, shape0 + 1))
        .describe("The shape of sparse feature for one example.");
    DMLC_DECLARE_FIELD(multi_field_shape_str).set_default("10000,10000,10000,10000,10000")
        .describe("The shape of multiple field feature for one example.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
    DMLC_DECLARE_FIELD(multi_field_num).set_default(0)
        .describe("the number of multiple field for one feature");
  }
};

std::vector<std::string> split(std::string str, char delimiter) {
  std::vector<std::string> internal;
  std::stringstream ss(str);
  std::string tok;

  while (std::getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  return internal;
}

class RMFIter: public SparseIIterator<DataInst> {
 public:
  RMFIter() {}
  virtual ~RMFIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    CHECK_EQ(param_.dense_shape.ndim(), 1) << "dimension of dense_shape is expected to be 1";
    CHECK_EQ(param_.cate_shape.ndim(), 1) << "dimension of cate_shape is expected to be 1";
    CHECK_EQ(param_.sparse_shape.ndim(), 1) << "dimension of sparse_shape is expected to be 1";
    CHECK_GT(param_.num_parts, 0) << "number of parts should be positive";
    CHECK_GE(param_.part_index, 0) << "part index should be non-negative";
    data_parser_.reset(dmlc::Parser<uint64_t>::Create(param_.data.c_str(),
                                                      param_.part_index,
                                                      param_.num_parts, "rmf"));
    CHECK_EQ(param_.label_shape.Size(), 1)
      << "label_shape is expected to be (1,) when param_.label_RMF is NULL";
    std::vector<std::string> multi_shape_str = split(param_.multi_field_shape_str, ',');
    CHECK_EQ(multi_shape_str.size(), param_.multi_field_num) 
      << "length of multi_field_shape_str is expected to be equal to multi_field_num";
    //TODO to fill multi_shape_str in multi_shape
    for (size_t i = 0; i < param_.multi_field_num; ++i) {
      index_t shape0[] = {std::stoi(multi_shape_str[i])};
      param_.multi_field_shape.push_back(TShape(shape0, shape0 + 1));
    }
    // both data and label are of multi Storage in RMF format
    if (param_.label_shape.Size() > 1) {
      out_.data.resize(5 + 3 * param_.multi_field_num + 3);
    } else {
      // only data is of CSRStorage in RMF format.
      // dense cate sparse multi_field
      // csv 1
      // libsvm 3
      out_.data.resize(5 + 3 * param_.multi_field_num + 1);
    }
    if (param_.label_width > 1) {
      index_t label_shape[] = {param_.label_width};
      param_.label_shape = TShape(label_shape, label_shape + 1);
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
    // dense   
    out_.data[0] = AsCSVTBlob(data_row, param_.dense_shape, 0);
    // cate
    out_.data[1] = AsCSVTBlob(data_row, param_.cate_shape, 1);
    //sparse data, indices and indptr
    out_.data[2] = AsCSRDataBlob(data_row, 2);
    out_.data[3] = AsCSRIdxBlob(data_row, 2);
    out_.data[4] = AsCSRIndPtrPlaceholder(data_row, 2);
    //multi_fields
    size_t ind = 5;
    for(size_t i = 0; i < param_.multi_field_num; i++) {
      out_.data[ind] = AsCSRDataBlob(data_row, i + 3);
      ++ind;
      out_.data[ind] = AsCSRIdxBlob(data_row, i + 3);
      ++ind;
      out_.data[ind] = AsCSRIndPtrPlaceholder(data_row, i + 3);
      ++ind;
    }
    if (label_parser_.get() != nullptr) {
      while (label_ptr_ >= label_size_) {
        CHECK(label_parser_->Next())
            << "Data RMF's row is smaller than the number of rows in label_RMF";
        label_ptr_ = 0;
        label_size_ = label_parser_->Value().size;
      }
      CHECK_LT(label_ptr_, label_size_);
      const auto label_row = label_parser_->Value()[label_ptr_++];
      // data, indices and indptr
      out_.data[ind++] = AsCSRDataBlob(label_row, 0);
      out_.data[ind++] = AsCSRIdxBlob(label_row, 0);
      out_.data[ind] = AsCSRIndPtrPlaceholder(label_row, 0);
    } else {
      if (param_.label_width > 1)
        out_.data[ind] = AsMultiLabelBlob(data_row);
      else
        out_.data[ind] = AsScalarLabelBlob(data_row);
    }
    //LOG(INFO) << "total ind = " << ind;
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

  virtual const NDArrayStorageType GetStorageType(size_t ind, bool is_data) const {
    if (ind == 0 || ind == 1) return kDefaultStorage;
    if (ind == 2) return kCSRStorage;
    size_t index = ind - 3;
    if (index < param_.multi_field_num) return kCSRStorage;
    return param_.label_shape.Size() > 1 ? (param_.label_width > 1? kDefaultStorage : kCSRStorage ) : kDefaultStorage;
  }

  virtual const TShape GetShape(size_t ind, bool is_data) const {
    if (ind == 0) return param_.dense_shape;
    if (ind == 1) return param_.cate_shape;
    if (ind == 2) return param_.sparse_shape;
    size_t index = ind - 3;
    if (index < param_.multi_field_num) {
      return param_.multi_field_shape[index];
    }
    return param_.label_shape;
  }

  bool IsIndPtr(size_t ind) {
    if (ind > 1 && (ind % 3 == 1)) return true; 
    return false;
  }

 private:
  // csv TODO process csv with different datatype
  inline TBlob AsCSVTBlob(const dmlc::Row<uint64_t>& row, const TShape& shape, size_t ind) {
    CHECK_LT(ind, 2)
        << "Index of csv data is less than size of Row's extra";
    CHECK_EQ(row.extra[ind].length, shape.Size())
        << "The data size in CSV do not match size of shape: "
        << "specified shape=" << shape << ", the csv row-length=" << row.length;
    const real_t* ptr = row.extra[ind].value;
    return TBlob((real_t*)ptr, shape, cpu::kDevMask, 0);  // NOLINT(*)
  }

  //libsvm
  inline TBlob AsCSRDataBlob(const dmlc::Row<uint64_t>& row, size_t ind) {
    //TODO check
    CHECK_LT(ind, param_.multi_field_num + 3)
        << "Index of csr data is less than size of Row's extra";
    const real_t* ptr = row.extra[ind].value;
    if (ptr == nullptr)
      return TBlob((real_t*) ptr, mshadow::Shape1(0), cpu::kDevMask);  // NOLINT(*)
    TShape shape(mshadow::Shape1(row.extra[ind].length));
    return TBlob((real_t*) ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  inline TBlob AsCSRIdxBlob(const dmlc::Row<uint64_t>& row, size_t ind) {
    const uint64_t* ptr = row.extra[ind].index;
    TShape shape(mshadow::Shape1(row.extra[ind].length));
    return TBlob((int64_t*) ptr, shape, cpu::kDevMask, mshadow::kInt64);  // NOLINT(*)
  }

  inline TBlob AsCSRIndPtrPlaceholder(const dmlc::Row<uint64_t>& row, size_t ind) {
    return TBlob(nullptr, mshadow::Shape1(0), cpu::kDevMask, mshadow::kInt64);
  }

  inline TBlob AsScalarLabelBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.label;
    return TBlob((real_t*) ptr, mshadow::Shape1(1), cpu::kDevMask);  // NOLINT(*)
  }
 
  inline TBlob AsMultiLabelBlob(const dmlc::Row<uint64_t>& row) {
    const real_t* ptr = row.label;
    return TBlob((real_t*) ptr, param_.label_shape, cpu::kDevMask);  // NOLINT(*)
  }

  RMFIterParam param_;
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


DMLC_REGISTER_PARAMETER(RMFIterParam);

MXNET_REGISTER_IO_ITER(RMFIter)
.describe(R"code(Returns the RMF iterator which returns data with `csr`
storage type. This iterator is experimental and should be used with care.

The input data is stored in a format similar to RMF file format, except that the **indices
are expected to be zero-based instead of one-based, and the column indices for each row are
expected to be sorted in ascending order**. Details of the RMF format are available
`here. <https://www.csie.ntu.edu.tw/~cjlin/RMFtools/datasets/>`_


The `data_shape` parameter is used to set the shape of each line of the data.
The dimension of both `data_shape` and `label_shape` are expected to be 1.

The `data` parameter is used to set the path input RMF file.
When it is set to a directory, all the files in the directory will be read.

When `label_RMF` is set to ``NULL``, both data and label are read from the file specified
by `data`. In this case, the data is stored in `csr` storage type, while the label is a 1D
dense array.

The `RMFIter` only support `round_batch` parameter set to ``True``. Therefore, if `batch_size`
is 3 and there are 4 total rows in RMF file, 2 more examples are consumed at the first round.

When `num_parts` and `part_index` are provided, the data is split into `num_parts` partitions,
and the iterator only reads the `part_index`-th partition. However, the partitions are not
guaranteed to be even.

``reset()`` is expected to be called only after a complete pass of data.

Example::

  # Contents of RMF file ``data.t``.
  1.0 0:0.5 2:1.2
  -2.0
  -3.0 0:0.6 1:2.4 2:1.2
  4 2:-1.2

  # Creates a `RMFIter` with `batch_size`=3.
  >>> data_iter = mx.io.RMFIter(data = 'data.t', data_shape = (3,), batch_size = 3)
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

When `label_RMF` is set to the path to another RMF file,
data is read from `data` and label from `label_RMF`.
In this case, both data and label are stored in the csr format.
If the label column in the `data` file is ignored.

Example::

  # Contents of RMF file ``label.t``
  1.0
  -2.0 0:0.125
  -3.0 2:1.2
  4 1:1.0 2:-1.2

  # Creates a `RMFIter` with specified label file
  >>> data_iter = mx.io.RMFIter(data = 'data.t', data_shape = (3,),
                   label_RMF = 'label.t', label_shape = (3,), batch_size = 3)

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
.add_arguments(RMFIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new SparsePrefetcherIter(
        new SparseBatchLoader(
            new RMFIter()));
  });

}  // namespace io
}  // namespace mxnet
