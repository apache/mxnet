/*!
 *  Copyright (c) 2015 by Contributors
 * \inst_vector.h
 * \brief holder of a sequence of DataInst in CPU
 *        that are not necessarily of same shape
 */
#ifndef MXNET_IO_INST_VECTOR_H_
#define MXNET_IO_INST_VECTOR_H_
#include <dmlc/base.h>
#include <mshadow/tensor.h>
#include <vector>
#include <string>
#include "./data.h"
namespace mxnet {
/*!
 * \brief tensor vector that can store sequence of tensor
 *  in a memory compact way, tensors do not have to be of same shape
 */
template<int dim, typename DType>
class TensorVector {
 public:
  TensorVector(void) {
    this->Clear();
  }
  // get i-th tensor
  inline mshadow::Tensor<cpu, dim, DType>
  operator[](size_t i) const {
    CHECK(i + 1 < offset_.size());
    CHECK(shape_[i].Size() == offset_[i + 1] - offset_[i]);
    return mshadow::Tensor<cpu, dim, DType>
        (reinterpret_cast<DType*>(BeginPtr(content_)) + offset_[i], shape_[i]);
  }
  inline mshadow::Tensor<cpu, dim, DType> Back() const {
    return (*this)[Size() - 1];
  }
  inline size_t Size(void) const {
    return shape_.size();
  }
  // push a tensor of certain shape
  // return the reference of the pushed tensor
  inline void Push(mshadow::Shape<dim> shape) {
    shape_.push_back(shape);
    offset_.push_back(offset_.back() + shape.Size());
    content_.resize(offset_.back());
  }
  inline void Clear(void) {
    offset_.clear();
    offset_.push_back(0);
    content_.clear();
    shape_.clear();
  }

 private:
  // offset of the data content
  std::vector<size_t> offset_;
  // data content
  std::vector<DType> content_;
  // shape of data
  std::vector<mshadow::Shape<dim> > shape_;
};

/*!
 * \brief tblob vector that can store sequence of tblob
 *  in a memory compact way, tblobs do not have to be of same shape
 */
template<typename DType>
class TBlobVector {
 public:
  TBlobVector(void) {
    this->Clear();
  }
  // get i-th tblob
  inline TBlob operator[](size_t i) const;
  // get the last tblob
  inline TBlob Back();
  // return the size of the vector
  inline size_t Size(void) const;
  // push a tensor of certain shape
  // return the reference of the pushed tensor
  inline void Push(TShape shape_);
  inline void Clear(void);
 private:
  // offset of the data content
  std::vector<size_t> offset_;
  // data content
  std::vector<DType> content_;
  // shape of data
  std::vector<TShape > shape_;
};

/*!
 * \brief instance vector that can holds
 * non-uniform shape data instance in a shape efficient way
 */
class InstVector {
 public:
  inline size_t Size(void) const {
    return index_.size();
  }
  // instance
  inline DataInst operator[](size_t i) const;
  // get back of instance vector
  inline DataInst Back() const;
  // clear the container
  inline void Clear(void);
  // push the newly coming instance
  inline void Push(unsigned index, TBlob data_);

 private:
  /*! \brief index of the data */
  std::vector<unsigned> index_;
  // data
  std::vector<TensorVector<real_t> > data_;
  // extra data
  std::vector<std::string> extra_data_;
};
#endif  // MXNET_IO_INST_VECTOR_H_
