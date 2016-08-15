/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MXNET_TENSOR_BLOB_H_
#define MXNET_TENSOR_BLOB_H_

#include <dmlc/logging.h>
#include <dmlc/json.h>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include "./base.h"

namespace mxnet {

/*!
 * \brief A dynamic sized array data strcuture
 *  that is optimized for storing small number of elements with same type.
 *  Data will be stored in stack when number of elements is small.
 *
 *  It is suitable to hold Shape of Tensor.
 *
 * \tparam ValueType The type of data stored inside tuple.
 * \sa TShape
 */
template<typename ValueType>
class Tuple {
 public:
  // Tuple requires the content to be simple data type.
  static_assert(std::is_pod<ValueType>::value,
                "Tuple only support simple data type like int");
  /*! \brief default constructor */
  Tuple() = default;
  /*! \brief destructor */
  inline ~Tuple() {
    delete [] data_heap_;
  }
  /*!
   * \brief copy constructor from another tuple
   * \param s the source tuple
   */
  inline Tuple(const Tuple<ValueType>& s) {
    this->assign(s.begin(), s.end());
  }
  /*!
   * \brief constructor from initializer list
   * \param init the initializer_list
   */
  inline Tuple(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
  }
  /*!
   * \brief move constructor from Tuple
   * \param src the source shape
   */

  inline Tuple(Tuple<ValueType>&& src) { // NOLINT(*)
    this->swap(src);
  }
  /*!
   * \brief construct the Tuple from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline Tuple(RandomAccessIterator begin,
               RandomAccessIterator end) {
    this->assign(begin, end);
  }
  /*!
   * \brief Assign content to tuple from iterator.
   * \param begin the beginning of iteratro
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline void assign(RandomAccessIterator begin,
                     RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, this->begin());
  }
  /*!
   * \brief Swap current object with other
   * \param other another object to be swapped.
   */
  inline void swap(Tuple<ValueType>& other) {  // NOLINT(*)
    std::swap(ndim_, other.ndim_);
    std::swap(num_heap_allocated_, other.num_heap_allocated_);
    std::swap(data_stack_, other.data_stack_);
    std::swap(data_heap_, other.data_heap_);
  }
  /*!
   * \brief assignment from another tuple.
   * \param src source tuple
   * \return reference of self
   */
  inline Tuple<ValueType>& operator=(const Tuple<ValueType>& src) {
    this->assign(src.begin(), src.end());
    return *this;
  }
  /*!
   * \brief assignment from rvalue of another tuple.
   * \param src source tuple
   * \return reference of self
   */
  inline Tuple<ValueType>& operator=(Tuple<ValueType>&& src) {
    Tuple<ValueType>(std::move(src)).swap(*this);
    return *this;
  }
  /*!
   * \brief assignment from initializer list
   * \param init the source initializer list
   * \return reference of self
   */
  inline Tuple<ValueType> &operator=(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
    return *this;
  }
  /*!
   * \return whether two tuple equals
   * \param s the tuple to compare against
   */
  inline bool operator==(const Tuple<ValueType> &s) const {
    if (ndim_ != s.ndim_) return false;
    return std::equal(begin(), end(), s.begin());
  }
  /*!
   * \return whether two tuple not equal
   * \param s the tuple to compare against
   */
  inline bool operator!=(const Tuple<ValueType> &s) const {
    return !(*this == s);
  }
  /*! \return the begin data pointer to content of the tuple */
  inline const ValueType *begin() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the begin data pointer to content of the tuple */
  inline ValueType *begin() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data pointer to end of the tuple */
  inline const ValueType* end() const {
    return ndim_ <= kStackCache ? (data_stack_ + ndim_): (data_heap_ + ndim_);
  }
  /*! \return the data pointer to end the tuple */
  inline ValueType* end() {
    return ndim_ <= kStackCache ? (data_stack_ + ndim_): (data_heap_ + ndim_);
  }
  /*! \return number of dimension of the tuple */
  inline index_t ndim() const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline ValueType& operator[](index_t i) {
    return begin()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const ValueType& operator[](index_t i) const {
    return begin()[i];
  }
  /*!
   * \brief Save Tuple to JSON.
   * \param writer JSONWriter
   */
  inline void Save(dmlc::JSONWriter* writer) const {
    std::vector<ValueType> tmp(begin(), end());
    writer->Write(tmp);
  }
  /*!
   * \brief Load Tuple from JSON.
   * \param reader JSONReader
   */
  inline void Load(dmlc::JSONReader* reader) {
    std::vector<ValueType> tmp;
    reader->Read(&tmp);
    this->assign(tmp.begin(), tmp.end());
  }
  /*!
   * \brief allow output string of tuple to ostream
   * \param os the output stream
   * \param t the tuple
   * \return the ostream
   */
  friend std::ostream &operator<<(std::ostream &os, const Tuple<ValueType> &t) {
    os << '(';
    const ValueType* begin = t.begin();
    const ValueType* end = t.end();
    for (const ValueType* it = begin; it != end; ++it) {
      if (it != begin) os << ',';
      os << *it;
    }
    // python style tuple
    if (t.ndim() == 1) os << ',';
    os << ')';
    return os;
  }
  /*!
   * \brief read tuple from the istream
   * \param is the input stream
   * \param t The tuple
   * \return the istream
   */
  friend std::istream &operator>>(std::istream &is, Tuple<ValueType> &t) {
    // get (
    while (true) {
      char ch = is.peek();
      if (isdigit(ch)) {
        ValueType idx;
        if (is >> idx) {
          t.assign(&idx, &idx + 1);
        }
        return is;
      }
      is.get();
      if (ch == '(') break;
      if (!isspace(ch)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    }
    index_t idx;
    std::vector<ValueType> tmp;
    while (is >> idx) {
      tmp.push_back(idx);
      char ch;
      do {
        ch = is.get();
      } while (isspace(ch));
      if (std::is_integral<ValueType>::value && ch == 'L') {
        ch = is.get();
      }
      if (ch == ',') {
        while (true) {
          ch = is.peek();
          if (isspace(ch)) {
            is.get(); continue;
          }
          if (ch == ')') {
            is.get(); break;
          }
          break;
        }
        if (ch == ')') break;
      } else if (ch == ')') {
        break;
      } else {
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    t.assign(tmp.begin(), tmp.end());
    return is;
  }

 protected:
  // stack cache size
  static const uint32_t kStackCache = 4;
  /*! \brief number of dimension of the tuple */
  index_t ndim_{0};
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_{0};
  /*! \brief in stack space used to store shape when it is small */
  ValueType data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  ValueType* data_heap_{nullptr};
  // internal function to change the dimension
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
        dim > num_heap_allocated_) {
      delete [] data_heap_;
      data_heap_ = new ValueType[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*!
 * \brief A Shape class that is used to represent shape of each tensor.
 */
class TShape : public Tuple<index_t> {
 public:
  /*! \brief default constructor */
  TShape() = default;
  /*!
   * constructor to construct a shape with all 1.
   * \param ndim the number of dimension
   */
  inline TShape(index_t ndim) {  // NOLINT(*)
    this->SetDim(ndim);
    std::fill_n(begin(), ndim, 1);
  }
  /*!
   * \brief copy constructor of TShape
   * \param s source shape.
   */
  inline TShape(const Tuple<index_t>& s) { // NOLINT(*)
    this->assign(s.begin(), s.end());
  }
  /*!
   * \brief constructor from initializer list
   * \param init the initializer_list
   */
  inline TShape(std::initializer_list<index_t> init) {
    this->assign(init.begin(), init.end());
  }
  /*!
   * \brief move constructor.
   * \param s source shape.
   */
  inline TShape(Tuple<index_t>&& s) {  // NOLINT(*)
    this->swap(s);
  }
  /*!
   * \brief construct the Tuple from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline TShape(RandomAccessIterator begin,
                RandomAccessIterator end) {
    this->assign(begin, end);
  }
  /*!
   * \brief assignment function from tshape
   * \param src source shape.
   * \return self.
   */
  inline TShape& operator=(const Tuple<index_t>& src) {
    this->assign(src.begin(), src.end());
    return *this;
  }
  /*!
   * \brief move assignment function from tshape
   * \param src source shape.
   * \return self.
   */
  inline TShape& operator=(Tuple<index_t>&& src) {  // NOLINT(*)
    TShape(std::move(src)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \return total number of elements in the shape */
  inline size_t Size() const {
    size_t size = 1;
    const index_t* start = begin(), *fin = end();
    for (const index_t* it = start; it != fin; ++it) {
      size *= *it;
    }
    return size;
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  inline index_t ProdShape(int dimstart, int dimend) const {
    index_t num = 1;
    const index_t *d = this->data();
    for (int i = dimstart; i < dimend; ++i) {
      num *= d[i];
    }
    return num;
  }
  /*! \return the begin data pointer to content of the tuple */
  inline const index_t *data() const {
    return begin();
  }
  /*! \return the begin data pointer to content of the tuple */
  inline index_t *data() {
    return begin();
  }
#ifdef MSHADOW_XINLINE
  template<int dim>
  inline TShape(mshadow::Shape<dim> &&s) {// NOLINT(*)
    this->assign(s.shape_, s.shape_ + dim);
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \tparam dim shape dimension
   * \return reference of self
   */
  template<int dim>
  inline TShape &operator=(const mshadow::Shape<dim> &shape) {
    this->assign(shape.shape_, shape.shape_ + dim);
    return *this;
  }
  /*!
   * \brief get the shape of tensor specifying dim
   * \return the shape requested
   * \tparam dim dimension of the tensor
   */
  template<int dim>
  inline mshadow::Shape<dim> get() const {
    CHECK_EQ(dim, ndim())
        << "dimension do not match target dimension " << dim << " vs " << ndim();
    const index_t *d = this->data();
    mshadow::Shape<dim> s;
    for (int i = 0; i < dim; ++i) {
      s[i] = d[i];
    }
    return s;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  inline mshadow::Shape<2> FlatTo2D(void) const {
    mshadow::Shape<2> s;
    if (ndim() == 0) return mshadow::Shape2(0, 0);
    const index_t *d = this->data();
    s.shape_[1] = d[ndim() - 1];
    index_t ymax = 1;
    for (index_t i = 1; i < ndim(); ++i) {
      ymax *= d[i - 1];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*!
  * flatten the shape into three parts: [0, axis_begin), [axis_begin, axis_end], (axis_end, ndim)
  * \param axis_begin The beginning axis specified.
  * \param axis_end The ending axis specified.
  * \return the flat 3d shape
  */
  inline mshadow::Shape<3> FlatTo3D(index_t axis_begin, index_t axis_end) const {
    CHECK(axis_end >= axis_begin);
    mshadow::Shape<3> s;
    if (ndim() == 0) return mshadow::Shape3(0, 0, 0);
    const index_t *d = this->data();
    s.shape_[0] = 1;
    s.shape_[1] = 1;
    s.shape_[2] = 1;

    for (index_t i = 0; i < axis_begin; ++i) {
      s.shape_[0] *= d[i];
    }
    for (index_t i = axis_begin; i <= axis_end; ++i) {
      s.shape_[1] *= d[i];
    }
    for (index_t i = axis_end + 1; i < ndim(); ++i) {
      s.shape_[2] *= d[i];
    }
    return s;
  }
  /*!
   * flatten the axis before and after the specified axis, so it becomes 3D tensor
   * \param axis The axis specified.
   * \return the flat 3d shape
   */
  inline mshadow::Shape<3> FlatTo3D(index_t axis) const {
    return FlatTo3D(axis, axis);
  }
  inline bool operator==(const TShape &s) const {
    if (ndim() != s.ndim()) return false;
    return std::equal(begin(), end(), s.begin());
  }
  inline bool operator!=(const TShape &s) const {
    return !(*this == s);
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator==(const mshadow::Shape<dim> &s) const {
    if (ndim_ != dim) return false;
    const index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (index_t i = 0; i < dim; ++i) {
      if (d[i] != s.shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator!=(const mshadow::Shape<dim> &s) const {
    return !(*this == s);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   */
  template<typename TStream>
  inline void Save(TStream *strm) const {
    strm->Write(&ndim_, sizeof(ndim_));
    strm->Write(data(), sizeof(index_t) * ndim_);
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   * \return whether the load is successful
   */
  template<typename TStream>
  inline bool Load(TStream *strm) {
    if (strm->Read(&ndim_, sizeof(ndim_)) != sizeof(ndim_)) return false;
    this->SetDim(ndim_);
    size_t nread = sizeof(index_t) * ndim_;
    if (strm->Read(data(), nread) != nread) return false;
    return true;
  }
#endif
};

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself do not involve any arithmentic operations,
 *  but it can be converted to tensor of fixed dimension for further operations
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 */
class TBlob {
 public:
  /*! \brief pointer to the data */
  void *dptr_;
  /*! \brief shape of the tensor */
  TShape shape_;
  /*!
   * \brief storing the stride information in x dimension
   */
  index_t stride_;
  /*! \brief device mask of the corresponding device */
  int dev_mask_;
  /*! \brief type flag of the tensor blob */
  int type_flag_;
  /*! \brief default constructor, default copy assign will work */
  TBlob(void)
      : dptr_(NULL), dev_mask_(cpu::kDevMask),
        type_flag_(mshadow::DataType<real_t>::kFlag) {}
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   */
  template<typename DType>
  TBlob(DType *dptr,
        const TShape &shape,
        int dev_mask)
      : dptr_(dptr), shape_(shape),
        stride_(shape[shape.ndim() - 1]),
        dev_mask_(dev_mask),
        type_flag_(mshadow::DataType<DType>::kFlag) {}
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   * \param type_flag the type flag. Can be one of enum mshadow::dtype
   */
  TBlob(void *dptr,
        const TShape &shape,
        int dev_mask,
        int type_flag)
      : dptr_(dptr), shape_(shape),
        stride_(shape[shape.ndim() - 1]),
        dev_mask_(dev_mask),
        type_flag_(type_flag) {}
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  TBlob(const mshadow::Tensor<Device, dim, DType> &src) {  // NOLINT(*)
    *this = src;
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   * \return reference of self
   */
  template<typename Device, int dim, typename DType>
  inline TBlob
  &operator=(const mshadow::Tensor<Device, dim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    stride_ = src.stride_;
    dev_mask_ = Device::kDevMask;
    type_flag_ = mshadow::DataType<DType>::kFlag;
    return *this;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return shape_[shape_.ndim() - 1] == stride_;
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 2, DType> FlatTo2D(
      mshadow::Stream<Device> *stream = NULL) const {
    CHECK(Device::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << mshadow::DataType<DType>::kFlag;
    return mshadow::Tensor<Device, 2, DType>(static_cast<DType*>(dptr_),
                                             shape_.FlatTo2D(), stride_, stream);
  }
  /*! \brief return number of dimension of the tensor inside */
  inline int ndim(void) const {
    return shape_.ndim();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param idx the dimension count from the highest dimensin
   * \return the size
   */
  inline index_t size(index_t idx) const {
    return shape_[idx];
  }
  /*! \brief total number of elements in the tensor */
  inline index_t Size(void) const {
    return shape_.Size();
  }
  /*!
   * \brief fetch the tensor, with respect to specific dimension
   * if dim do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline mshadow::Tensor<Device, dim, DType> get(mshadow::Stream<Device> *stream = NULL) const {
    CHECK(Device::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << mshadow::DataType<DType>::kFlag;
    return mshadow::Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                               shape_.get<dim>(),
                                               stride_, stream);
  }
  /*!
   * \brief fetch a tensor in given shape
   *  If size do not match the stored size, an error will be issued
   * \return the tensor requested
   * \param shape the shape required
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline mshadow::Tensor<Device, dim, DType> get_with_shape(
      const mshadow::Shape<dim> &shape,
      mshadow::Stream<Device> *stream = NULL) const {
    CHECK(Device ::kDevMask == dev_mask_)
      << "TBlob.get: device type do not match specified type";
    CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << type_flag_ << " v.s. given " << mshadow::DataType<DType>::kFlag;
    CHECK_EQ(this->CheckContiguous(), true) << "TBlob.get_reshape: must be contiguous";
    CHECK_EQ(this->shape_.Size(), shape.Size())
      << "TBlob.get_with_shape: new and old shape do not match total elements";
    return mshadow::Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                               shape,
                                               shape[dim - 1],
                                               stream);
  }
  /*!
   * \brief flatten the tensor to 3 dimension,
   *  collapse the dimension before and after specified axis.
   * \param axis The axis specified.
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
      int axis, mshadow::Stream<Device> *stream = NULL) const {
    return this->get_with_shape<Device, 3, DType>(
        this->shape_.FlatTo3D(axis), stream);
  }
  /*!
   * \brief flatten the tensor to 3 dimension,
   *  collapse the dimension: [0, axis_begin), [axis_begin, axis_end], (axis_end, ndim).
   * \param axis_begin The beginning axis specified.
   * \param axis_end The ending axis specified.
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
      int axis_begin, int axis_end,
      mshadow::Stream<Device> *stream = NULL) const {
    return this->get_with_shape<Device, 3, DType>(
        this->shape_.FlatTo3D(axis_begin, axis_end), stream);
  }
};
}  // namespace mxnet

namespace dmlc {
// Add a few patches to support TShape in dmlc/parameter.
DMLC_DECLARE_TYPE_NAME(mxnet::TShape, "Shape(tuple)");

namespace parameter {

template<>
class FieldEntry<mxnet::TShape>
    : public FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> {
 public:
  FieldEntry() : enforce_nonzero_(false), expect_ndim_(0) {}
  // parent class
  typedef FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> Parent;

  virtual void Check(void *head) const {
    Parent::Check(head);
    mxnet::TShape &v = this->Get(head);
    if (expect_ndim_ != 0 && v.ndim() != expect_ndim_) {
      std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " has wrong dimensions, expected dimension=" << expect_ndim_;
        throw dmlc::ParamError(os.str());
    }
    if (enforce_nonzero_) {
      for (mxnet::index_t i = 0; i < v.ndim(); ++i) {
        if (v[i] == 0U) {
          std::ostringstream os;
          os << "value " << v << "for Parameter " << this->key_
             << " is invalid, the input shape must be nonzero in all dimensions";
          throw dmlc::ParamError(os.str());
        }
      }
    }
  }
  inline FieldEntry<mxnet::TShape> &enforce_nonzero() {
    this->enforce_nonzero_ = true;
    return this->self();
  }
  inline FieldEntry<mxnet::TShape> &set_expect_ndim(mxnet::index_t ndim) {
    expect_ndim_ = ndim;
    return this->self();
  }

 private:
  // whether all the entries need to be nonzero
  bool enforce_nonzero_;
  // expected number of dimension, default = 0 means no restriction.
  mxnet::index_t expect_ndim_;
};

}  // namespace parameter
}  // namespace dmlc

#endif  // MXNET_TENSOR_BLOB_H_
