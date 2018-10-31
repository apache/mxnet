/*!
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/tuple.h
 * \brief Data structure Tuple and TShape to store dynamic sized shapes.
 */
#ifndef NNVM_TUPLE_H_
#define NNVM_TUPLE_H_

#include <vector>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <iostream>
#include <string>
#include "base.h"

namespace nnvm {

/*! \brief data type to store dim size */
typedef int64_t dim_t;

/*!
 * \brief A dynamic sized array data structure that is optimized for storing
 *        small number of elements with same type.
 *
 *  Data will be stored in stack when number of elements is small.
 *  It is suitable to hold shape of Tensor.
 *
 * \tparam ValueType The type of data stored inside tuple.
 * \sa TShape
 */
template<typename ValueType>
class Tuple {
 public:
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
   * \brief constructor from vector
   * \param init the vector
   */
  inline Tuple(std::vector<ValueType> init) {  // NOLINT(runtime/explicit)
    this->assign(init.begin(), init.end());
  }
  /*!
   * \brief move constructor from Tuple
   * \param src the source shape
   */

  inline Tuple(Tuple<ValueType>&& src) {   // NOLINT(runtime/explicit)
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
   * \param begin the beginning of iterator
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
  inline uint32_t ndim() const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline ValueType& operator[](size_t i) {
    return begin()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const ValueType& operator[](size_t i) const {
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
    os << '[';
    const ValueType* begin = t.begin();
    const ValueType* end = t.end();
    for (const ValueType* it = begin; it != end; ++it) {
      if (it != begin) os << ',';
      os << *it;
    }
    os << ']';
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
      if (isdigit(ch) || ch == '-') {
        ValueType idx;
        if (is >> idx) {
          t.assign(&idx, &idx + 1);
        }
        return is;
      }
      is.get();
      if (ch == '(' || ch == '[') break;
      if (!isspace(ch)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    }
    // Handle empty tuple
    while (isspace(is.peek())) {
      is.get();
    }
    if (is.peek() == ')' || is.peek() == ']') {
      is.get();
      return is;
    }
    // Handle non-empty tuple
    ValueType idx;
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
          if (ch == ')' || ch == ']') {
            is.get(); break;
          }
          break;
        }
        if (ch == ')' || ch == ']') break;
      } else if (ch == ')' || ch == ']') {
        break;
      } else {
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    t.assign(tmp.begin(), tmp.end());
    return is;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   * \tparam DType data type that save to
   * \tparam TStream any stream type that have write
   */
  template<typename DType = ValueType, typename TStream>
  inline void Save(TStream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \tparam DType data type that load from
   * \tparam TStream any stream type that have write
   * \return whether the load is successful
   */
  template<typename DType = ValueType, typename TStream>
  inline bool Load(TStream *strm);

 protected:
  // stack cache size
  static const uint32_t kStackCache = 4;
  /*! \brief number of dimension of the tuple */
  uint32_t ndim_{0};
  /*! \brief number of cells allocated in data_heap_ */
  uint32_t num_heap_allocated_{0};
  /*! \brief in stack space used to store shape when it is small */
  ValueType data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  ValueType* data_heap_{nullptr};
  // internal function to change the dimension
  inline void SetDim(uint32_t ndim) {
    if (ndim > kStackCache &&
        ndim > num_heap_allocated_) {
      delete [] data_heap_;
      data_heap_ = new ValueType[ndim];
      num_heap_allocated_ = ndim;
    }
    ndim_ = ndim;
  }
};

/*!
 * \brief A Shape class that is used to represent shape of each tensor.
 */
class TShape : public Tuple<dim_t> {
 public:
  /*! \brief default constructor */
  TShape() = default;
  /*!
   * constructor to construct a shape with all 1.
   * \param ndim the number of dimension
   */
  inline TShape(uint32_t ndim) {  // NOLINT(*)
    this->SetDim(ndim);
    std::fill_n(begin(), ndim, 1);
  }
  /*!
   * \brief copy constructor of TShape
   * \param s source shape.
   */
  inline TShape(const Tuple<dim_t>& s) { // NOLINT(*)
    this->assign(s.begin(), s.end());
  }
  /*!
   * \brief constructor from initializer list
   * \param init the initializer_list
   */
  inline TShape(std::initializer_list<dim_t> init) {
    this->assign(init.begin(), init.end());
  }
  /*!
   * \brief move constructor.
   * \param s source shape.
   */
  inline TShape(Tuple<dim_t>&& s) {  // NOLINT(*)
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
  inline TShape& operator=(const Tuple<dim_t>& src) {
    this->assign(src.begin(), src.end());
    return *this;
  }
  /*!
   * \brief move assignment function from tshape
   * \param src source shape.
   * \return self.
   */
  inline TShape& operator=(Tuple<dim_t>&& src) {  // NOLINT(*)
    TShape(std::move(src)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \return total number of elements in the shape */
  inline size_t Size() const {
    dim_t size = 1;
    const dim_t* start = begin(), *fin = end();
    for (const dim_t* it = start; it != fin; ++it) {
      size *= *it;
    }
    return size;
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  inline size_t ProdShape(int dimstart, int dimend) const {
    dim_t num = 1;
    const dim_t *d = this->data();
    for (int i = dimstart; i < dimend; ++i) {
      num *= d[i];
    }
    return num;
  }
  /*! \return the begin data pointer to content of the tuple */
  inline const dim_t *data() const {
    return begin();
  }
  /*! \return the begin data pointer to content of the tuple */
  inline dim_t *data() {
    return begin();
  }
#ifdef MSHADOW_XINLINE
  template<int dim>
  inline TShape(const mshadow::Shape<dim> &s) {// NOLINT(*)
    this->assign(s.shape_, s.shape_ + dim);
  }

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
    CHECK_EQ(dim, static_cast<int>(ndim()))
        << "dimension do not match target dimension " << dim << " vs " << ndim();
    const dim_t *d = this->data();
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
    const dim_t *d = this->data();
    s.shape_[1] = d[ndim() - 1];
    dim_t ymax = 1;
    for (size_t i = 1; i < ndim(); ++i) {
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
  inline mshadow::Shape<3> FlatTo3D(size_t axis_begin, size_t axis_end) const {
    CHECK(axis_end >= axis_begin);
    mshadow::Shape<3> s;
    if (ndim() == 0) return mshadow::Shape3(0, 0, 0);
    const dim_t *d = this->data();
    s.shape_[0] = 1;
    s.shape_[1] = 1;
    s.shape_[2] = 1;

    for (size_t i = 0; i < axis_begin; ++i) {
      s.shape_[0] *= d[i];
    }
    for (size_t i = axis_begin; i <= axis_end; ++i) {
      s.shape_[1] *= d[i];
    }
    for (size_t i = axis_end + 1; i < ndim(); ++i) {
      s.shape_[2] *= d[i];
    }
    return s;
  }
  /*!
   * flatten the axis before and after the specified axis, so it becomes 3D tensor
   * \param axis The axis specified.
   * \return the flat 3d shape
   */
  inline mshadow::Shape<3> FlatTo3D(size_t axis) const {
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
    const dim_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (size_t i = 0; i < dim; ++i) {
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
#endif
};

/*! \brief helper function to cast type of container elements */
template<typename SrcIter, typename DstIter>
inline DstIter ShapeTypeCast(const SrcIter begin,
                             const SrcIter end,
                             DstIter dst_begin) {
  typedef typename std::iterator_traits<SrcIter>::value_type SrcDType;
  typedef typename std::iterator_traits<DstIter>::value_type DstDType;
  auto cast = [](const SrcDType& dim) { return static_cast<DstDType>(dim); };
  return std::transform(begin, end, dst_begin, cast);
}

/*! \brief helper function to transform a container to TShape with type cast */
template<typename SrcIter>
inline TShape ShapeTypeCast(const SrcIter begin, const SrcIter end) {
  size_t ndim = std::distance(begin, end);
  TShape res(ndim);
  ShapeTypeCast(begin, end, res.begin());
  return res;
}

/*! \tparam ValueType The type of data stored inside tuple. */
template<typename ValueType>
template<typename DType, typename TStream>
inline void Tuple<ValueType>::Save(TStream *strm) const {
  strm->Write(&ndim_, sizeof(ndim_));
  if (typeid(DType) == typeid(ValueType)) {
    strm->Write(begin(), sizeof(ValueType) * ndim_);
  } else {
    std::vector<DType> buffer(ndim_);
    ShapeTypeCast(begin(), end(), buffer.data());
    strm->Write(buffer.data(), sizeof(DType) * ndim_);
  }
}

/*! \tparam ValueType The type of data stored inside tuple. */
template<typename ValueType>
template<typename DType, typename TStream>
inline bool Tuple<ValueType>::Load(TStream *strm) {
  if (strm->Read(&ndim_, sizeof(ndim_)) != sizeof(ndim_)) return false;
  this->SetDim(ndim_);
  size_t nread = sizeof(DType) * ndim_;
  if (typeid(DType) == typeid(ValueType)) {
    if (strm->Read(begin(), nread) != nread) return false;
  } else {
    std::vector<DType> buffer(ndim_);
    if (strm->Read(buffer.data(), nread) != nread) return false;
    ShapeTypeCast(buffer.begin(), buffer.end(), begin());
  }
  return true;
}

}  // namespace nnvm

namespace std {
/*! \brief hash function for Tuple. */
template<typename T>
struct hash<nnvm::Tuple<T> > {
  /*! \brief hash a Tuple into unsigned int */
  size_t operator()(const nnvm::Tuple<T>& val) const {
    std::hash<uint32_t> hash_uint;
    size_t res = hash_uint(val.ndim());
    for (uint32_t i = 0; i < val.ndim(); ++i) {
      res = dmlc::HashCombine(res, val[i]);
    }
    return res;
  }
};

/*! \brief hash function for TShape. */
template<>
struct hash<nnvm::TShape> {
  /*! \brief hash a TShape into unsigned int */
  size_t operator()(const nnvm::TShape& val) const {
    std::hash<uint32_t> hash_uint;
    size_t res = hash_uint(val.ndim());
    for (uint32_t i = 0; i < val.ndim(); ++i) {
      res = dmlc::HashCombine(res, val[i]);
    }
    return res;
  }
};
}  // namespace std

namespace dmlc {
/*! \brief description for optional TShape */
DMLC_DECLARE_TYPE_NAME(optional<nnvm::TShape>, "Shape or None");
// avoid low version of MSVC
#if !defined(_MSC_VER)
template<typename T>
struct type_name_helper<nnvm::Tuple<T> > {
  static inline std::string value() {
    return "tuple of <" + type_name<T>() + ">";
  }
};
#endif
}  // namespace dmlc
#endif  // NNVM_TUPLE_H_
