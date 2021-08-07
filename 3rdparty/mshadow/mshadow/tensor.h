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
 * \file tensor.h
 * \brief header file of tensor data structure and functions
 *  This lib requires explicit memory allocation and de-allocation
 *  all the data structure Tensor<cpu,1>, Tensor<gpu,1> are like handles(pointers),
 *  no memory allocation is happening during calculation
 *
 *  For STL style tensor, see tensor_container.h
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_H_
#define MSHADOW_TENSOR_H_
#include <string>
#include <iostream>
#include "./base.h"
#include "./expression.h"

namespace mshadow {
/*! \brief device name CPU */
struct cpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = true;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 0;
};
/*! \brief device name GPU */
struct gpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = false;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 1;
};
template<int ndim>
struct Shape;

/*!
 * \brief allow string printing of the shape
 * \param os the output stream
 * \param shape the shape
 * \return the ostream
 */
template<int ndim>
inline std::ostream &operator<<(std::ostream &os, const Shape<ndim> &shape); // NOLINT(*)

/*!
 * \brief shape of a tensor
 * \tparam dimension dimension of tensor
 */
template<int dimension>
struct Shape {
  /*! \brief dimension of current shape */
  static const int kDimension = dimension;
  /*! \brief dimension of current shape minus one */
  static const int kSubdim = dimension - 1;
  /*! \brief storing the dimension information */
  index_t shape_[kDimension];
  /*! \brief default constructor, do nothing */
  MSHADOW_XINLINE Shape(void) {}
  /*! \brief constuctor */
  MSHADOW_XINLINE Shape(const Shape<kDimension> &s) {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      this->shape_[i] = s[i];
    }
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE index_t &operator[](int idx) {
    return shape_[idx];
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE const index_t &operator[](int idx) const {
    return shape_[idx];
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   */
  MSHADOW_XINLINE bool operator==(const Shape<kDimension> &s) const {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      if (s.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equal
   * \param s the shape to compare against
   */
  MSHADOW_XINLINE bool operator!=(const Shape<kDimension> &s) const {
    return !(*this == s);
  }
  /*!
   * flatten the tensor, return a 1D shape
   * \return the flat 1d shape
   */
  MSHADOW_XINLINE Shape<1> FlatTo1D(void) const {
    Shape<1> s;
    s[0] = this->Size();
    return s;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  MSHADOW_XINLINE Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    s.shape_[1] = this->shape_[kDimension - 1];
    index_t ymax = 1;
    #pragma unroll
    for (int i = 0; i < kDimension - 1; ++i) {
      ymax *= this->shape_[i];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*! \return number of valid elements */
  MSHADOW_XINLINE index_t Size(void) const {
    index_t size = this->shape_[0];
    #pragma unroll
    for (int i = 1; i < kDimension; ++i) {
      size *= this->shape_[i];
    }
    return size;
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  MSHADOW_XINLINE index_t ProdShape(int dimstart, int dimend) const {
    index_t num = 1;
    #pragma unroll
    for (int i = dimstart; i < dimend; ++i) {
      num *= this->shape_[i];
    }
    return num;
  }
  /*!
   * \brief get subshape that takes off largest dimension
v   * \return subshape
   */
  MSHADOW_XINLINE Shape<kSubdim> SubShape(void) const {
    Shape<kSubdim> s;
    // for cuda
    #pragma unroll
    for (int i = 0; i < kSubdim; ++i) {
      s.shape_[i] = this->shape_[i + 1];
    }
    return s;
  }
  /*!
   * \brief slice the shape from start to end
   * \tparam dimstart start dimension
   * \tparam dimend end dimension
   * \return the sliced shape
   */
  template<int dimstart, int dimend>
  MSHADOW_XINLINE Shape<dimend - dimstart> Slice(void) const {
    Shape<dimend - dimstart> s;
    #pragma unroll
    for (int i = dimstart; i < dimend; ++i) {
      s[i - dimstart] = this->shape_[i];
    }
    return s;
  }
  //! \cond Doxygen_Suppress
  template<int dim>
  friend std::ostream &operator<<(std::ostream &os, const Shape<dim> &shape); // NOLINT(*)
  //! \endcond
};  // Shape
//------------------------------------------------
// useful construction functions to generate shape
//-------------------------------------------------
/*!
 * \brief construct a one dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<1> Shape1(index_t s0) {
  Shape<1> s; s[0] = s0;
  return s;
}
/*!
 * \brief construct a two dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
  Shape<2> s; s[0] = s0; s[1] = s1;
  return s;
}
/*!
 * \brief construct a three dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \param s2 size of dimension 2
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<3> Shape3(index_t s0, index_t s1, index_t s2) {
  Shape<3> s;
  s[0] = s0; s[1] = s1; s[2] = s2;
  return s;
}
/*!
 * \brief construct a four dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \param s2 size of dimension 2
 * \param s3 size of dimension 3
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<4> Shape4(index_t s0, index_t s1,
                                index_t s2, index_t s3) {
  Shape<4> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
  return s;
}
/*!
* \brief construct a five dimension shape, stride will equal s0
* \param s0 size of dimension 0
* \param s1 size of dimension 1
* \param s2 size of dimension 2
* \param s3 size of dimension 3
* \param s4 size of dimension 4
* \return the shape construction
*/
MSHADOW_XINLINE Shape<5> Shape5(index_t s0, index_t s1, index_t s2,
                                index_t s3, index_t s4) {
  Shape<5> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; s[4] = s4;
  return s;
}

/*!
* \brief Convert shape in src_layout to shape in dst_layout
* \param src original shape
* \param src_layout layout of original shape
* \param dst_layout target layout
* \return shape in target layout
*/
inline Shape<3> ConvertLayout(const Shape<3>& src, int src_layout, int dst_layout) {
  Shape<3> dst;
  switch (src_layout) {
  case kNCW:
    dst = src;
    break;
  case kNWC:
    dst[0] = src[0];
    dst[1] = src[2];
    dst[2] = src[1];
    break;
  default:
    LOG(FATAL) << "Invalid layout for 3d shape " << src_layout;
  }
  switch (dst_layout) {
  case kNCW:
    return dst;
  case kNWC:
    {
      index_t tmp = dst[1];
      dst[1] = dst[2];
      dst[2] = tmp;
    }
    break;
  default:
    LOG(FATAL) << "Invalid layout for 3d shape " << src_layout;
  }
  return dst;
}

/*!
* \brief Convert shape in src_layout to shape in dst_layout
* \param src original shape
* \param src_layout layout of original shape
* \param dst_layout target layout
* \return shape in target layout
*/
inline Shape<4> ConvertLayout(const Shape<4>& src, int src_layout, int dst_layout) {
  Shape<4> dst;
  switch (src_layout) {
  case kNCHW:
    dst = src;
    break;
  case kNHWC:
    dst[0] = src[0];
    dst[2] = src[1];
    dst[3] = src[2];
    dst[1] = src[3];
    break;
  default:
    LOG(FATAL) << "Invalid layout for 4d shape " << src_layout;
    dst = src;  // fixes compiler warning
  }
  Shape<4> dst2;
  switch (dst_layout) {
  case kNCHW:
    return dst;
  case kNHWC:
    dst2[0] = dst[0];
    dst2[1] = dst[2];
    dst2[2] = dst[3];
    dst2[3] = dst[1];
    break;
  default:
    LOG(FATAL) << "Invalid layout for 4d shape " << src_layout;
    dst2 = src;  // fixes compiler warning
  }
  return dst2;
}

/*!
* \brief Convert shape in src_layout to shape in dst_layout
* \param src original shape
* \param src_layout layout of original shape
* \param dst_layout target layout
* \return shape in target layout
*/
inline Shape<5> ConvertLayout(const Shape<5>& src, int src_layout, int dst_layout) {
  Shape<5> dst;
  switch (src_layout) {
  case kNCDHW:
    dst = src;
    break;
  case kNDHWC:
    dst[0] = src[0];
    dst[2] = src[1];
    dst[3] = src[2];
    dst[4] = src[3];
    dst[1] = src[4];
    break;
  default:
    LOG(FATAL) << "Invalid layout for 5d shape " << src_layout;
  }
  Shape<5> dst2;
  switch (dst_layout) {
  case kNCDHW:
    return dst;
  case kNDHWC:
    dst2[0] = dst[0];
    dst2[1] = dst[2];
    dst2[2] = dst[3];
    dst2[3] = dst[4];
    dst2[4] = dst[1];
    break;
  default:
    LOG(FATAL) << "Invalid layout for 5d shape " << src_layout;
  }
  return dst2;
}

/*!
 * \brief computaion stream structure, used for asynchronous computations
 */
template<typename Device>
struct Stream {
  // this is only a dummy implementation for CPU
  // for GPU, the actual implementation will be specialized in tensor_gpu-inl.h
  /*!
   * \brief wait for all the computations associated
   *  with this stream to complete
   */
  inline void Wait(void) {}
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the jobs have been completed
   */
  inline bool CheckIdle(void) {
    return true;
  }
  /*! \brief create a blas handle */
  inline void CreateBlasHandle() {}
};
/*!
 * \brief Tensor RValue, this is the super type of all kinds of possible tensors
 * \tparam Container the tensor type
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Container, typename Device, int dimension, typename DType>
struct TRValue: public expr::RValueExp<Container, DType> {
};
// more compact template
/*!
 * \brief general tensor
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Device, int dimension,
         typename DType MSHADOW_DEFAULT_DTYPE>
struct Tensor: public TRValue<Tensor<Device, dimension, DType>,
                              Device, dimension, DType> {
 public:
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief whether current type lies in cpu */
  static const bool kDevCPU = Device::kDevCPU;
  /*! \brief dimension of subtype */
  static const int  kSubdim = dimension - 1;
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief pointer to the data */
  DType *dptr_ = nullptr;
  /*! \brief shape of the tensor */
  Shape<dimension> shape_;
  /*!
   * \brief storing the stride information in x dimension
   *    this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
   */
  index_t stride_;
  /*!
   * \brief stream where the computation lies
   * stream is a device dependency concept where each computation
   */
  Stream<Device> *stream_;
  //--------------------------------
  // functions
  //--------------------------------
  /*! \brief default constructor */
  MSHADOW_XINLINE Tensor(void) : stream_(NULL) {}
  /*! \brief constructor from shape  */
  MSHADOW_XINLINE Tensor(const Shape<dimension> &shape)
      : shape_(shape), stream_(NULL) {}
  /*! \brief constructor from data pointer and shape, without stride */
  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape)
      : dptr_(dptr), shape_(shape), stride_(shape[kSubdim]), stream_(NULL) {}
  /*! \brief constructor from data pointer and shape, without stride */
  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape,
                         Stream<Device> *stream)
    : dptr_(dptr), shape_(shape), stride_(shape[kSubdim]), stream_(stream) {}
  /*! \brief constructor from data pointer and shape  */
  MSHADOW_XINLINE Tensor(DType *dptr,
                         const Shape<dimension> &shape,
                         index_t stride, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}
  /*!
   * \brief set the stream to do computation of current tensor
   * \param stream the computation stream
   */
  inline void set_stream(Stream<Device> *stream) {
    this->stream_ = stream;
  }
  /*!
   * \return memory cost of the tensor, including the aligned x dimension
   * \tparam startdim the starting dimension
   */
  template<int startdim>
  MSHADOW_XINLINE index_t MemSize(void) const {
    index_t memsz = this->stride_;
    #pragma unroll
    for (int i = startdim; i < kSubdim; ++i) {
      memsz *= this->shape_[i];
    }
    return memsz;
  }
  /*!
   * \return whether the tensor's memory is continuous
   * x dimension same as stride
   */
  MSHADOW_XINLINE bool CheckContiguous(void) const {
    return this->shape_[dimension - 1] == stride_;
  }
  /*!
   * \return memory cost of the tensor, including the aligned x dimension
   */
  MSHADOW_XINLINE index_t MSize(void) const {
    return this->MemSize<0>();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param idx the dimension count from the highest dimensin
   * \return the size
   */
  MSHADOW_XINLINE index_t size(int idx) const {
    return shape_[idx];
  }
  /*!
   * \brief flatten the tensor to 1 dimension
   * \return tensor after flatten
   */
  MSHADOW_XINLINE Tensor<Device, 1, DType> FlatTo1D(void) const {
    return Tensor<Device, 1, DType>(dptr_, shape_.FlatTo1D(), stride_, stream_);
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \return tensor after flatten
   */
  MSHADOW_XINLINE Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }
  /*!
   * \brief get a element of dimension - 1
   * \param idx index
   * \return the result tensor
   */
  MSHADOW_XINLINE Tensor<Device, kSubdim, DType> operator[](index_t idx) const {
    return Tensor<Device, kSubdim, DType>(dptr_ + this->MemSize<1>() * idx,
                                          shape_.SubShape(), stride_, stream_);
  }
  /*!
   * \brief slice the tensor in highest dimension [begin,end)
   * \param begin begin position of slice
   * \param end end position of slice
   * \return tensor after slice
   */
  MSHADOW_XINLINE Tensor<Device, dimension, DType>
  Slice(index_t begin, index_t end) const {
    Shape<dimension> s = this->shape_;
    s[0] = end - begin;
    return Tensor<Device, dimension, DType>(dptr_ + this->MemSize<1>() * begin,
                                            s, stride_, stream_);
  }
  /*!\brief implement the assignment of same type */
  inline Tensor<Device, dimension, DType> &
  operator=(const Tensor<Device, dimension, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }
  /*!\brief functions to fit expression template */
  template<typename E, int etype>
  inline Tensor<Device, dimension, DType> &
  operator=(const expr::Exp<E, DType, etype> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  inline Tensor<Device, dimension, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};
/*
 *  respecialized class Tensor1D, thei is due to different implementation in operator[]
 */
template<typename Device, typename DType>
struct Tensor<Device, 1, DType>:
      public TRValue<Tensor<Device, 1, DType>, Device, 1, DType> {
 public:
  DType *dptr_;
  Shape<1> shape_;
  index_t stride_;
  Stream<Device> *stream_;
  // constructor
  MSHADOW_XINLINE Tensor(void) : stream_(NULL) {}
  MSHADOW_XINLINE Tensor(const Shape<1> &shape)
      : shape_(shape), stream_(NULL) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape)
      : dptr_(dptr), shape_(shape), stride_(shape[0]), stream_(NULL) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(shape[0]), stream_(stream) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape,
                         index_t stride, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}
  inline void set_stream(Stream<Device> *stream) {
    this->stream_ = stream;
  }
  MSHADOW_XINLINE Tensor<Device, 1, DType> FlatTo1D(void) const {
    return *this;
  }
  MSHADOW_XINLINE Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }
  MSHADOW_XINLINE Tensor<Device, 1, DType> Slice(index_t begin, index_t end) const {
    Shape<1> s;
    s[0] = end  - begin;
    return Tensor<Device, 1, DType>(dptr_ + begin, s, s[0], stream_);
  }
  MSHADOW_XINLINE bool CheckContiguous(void) const {
    return true;
  }
  MSHADOW_XINLINE index_t MSize(void) const {
    return shape_[0];
  }
  MSHADOW_XINLINE index_t size(index_t i) const {
    return shape_[0];
  }
  MSHADOW_XINLINE DType &operator[](index_t idx) {
    return dptr_[idx];
  }
  MSHADOW_XINLINE const DType &operator[](index_t idx) const {
    return dptr_[idx];
  }
  /*!\brief implement the assignment of same type */
  inline Tensor<Device, 1, DType> &
  operator=(const Tensor<Device, 1, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }
  template<typename E, int etype>
  inline Tensor<Device, 1, DType> &
  operator=(const expr::Exp<E, DType, etype> &exp) {
    return this->__assign(exp);
  }
  inline Tensor<Device, 1, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};
//------------------------
// Function Declarations
//-----------------------
/*!
 * \brief initialize tensor engine, used to call intialization functions of dependent libs
 *        this function should be called before all GPU tensor operations,
 *        for using tensors in CPU, this call is actually not needed
 * \param device_id GPU device id to be choosed
 * \tparam Device the device type
 */
template<typename Device>
inline void InitTensorEngine(int device_id = 0);
/*!
 * \brief Shutdown tensor engine on current device
 *     this function should be called after all GPU tensor operations,
 *     for using tensors in CPU, this call is actually not needed
 * \tparam Device the device type
 */
template<typename Device>
inline void ShutdownTensorEngine(void);
/*!
 * \brief set the device of current thread to work on
 * \param devid the device id
 * \tparam Device the device type
 */
template<typename Device>
inline void SetDevice(int devid);
/*!
 * \brief create a new stream from system
 * \param create_blas_handle whether create blas & cusolver handle in stream
 * \param create_dnn_handle whether create cudnn handle in stream
 * \param dev_id device id
 * \return a pointer to the created stream
 * \tparam Device the device type
 */
template<typename Device>
inline Stream<Device> *NewStream(bool create_blas_handle,
                                 bool create_dnn_handle,
                                 int dev_id = -1);
/*! \brief default behavior: create cublas handle
 *  \param dev_id device id
 *  \return a pointer to the created stream
 */
template<typename Device>
inline Stream<Device> *NewStream(int dev_id) {
  return NewStream<Device>(true, false, dev_id);
}
/*!
 * \brief delete the computing stream
 * \param stream the stream parameter to be deleted
 */
template<typename Device>
inline void DeleteStream(Stream<Device> *stream);
/*!
 * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
 *        this function is responsible to set the stride_ in each obj.shape
 * \param obj the tensor object, with shape specified
 * \param pad whether padding dimension 0, to make last dimension aligned,
 *            padding may help improve efficiency of matrix multiplications
 *            if true, will allocate space with stride_ that may not equals shape[0]
 *            if false, will allocate continuous space
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void AllocSpace(Tensor<cpu, dim, DType> *obj,
                       bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
 *        this function is responsible to set the stride_ in each obj.shape
 * \param obj the tensor object, with shape specified
 * \param pad whether padding dimension 0, to make last dimension aligned,
 *            padding may help improve efficiency of matrix multiplications
 *            if true, will allocate space with stride_ that may not equals shape[0]
 *            if false, will allocate continuous space
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void AllocSpace(Tensor<gpu, dim, DType> *obj,
                       bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
 * \param obj the tensor object
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> *obj);
/*!
 * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
 * \param obj the tensor object
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> *obj);
/*!
 * \brief CPU/GPU: short cut to allocate and initialize a Tensor
 * \param shape: shape of tensor
 * \param initv: initialization value
 * \param pad : padding option
 * \param stream : stream of tensor
 * \tparam Device device of tensor
 * \tparam DType type of element in tensor
 * \tparam dim dimention of tensor
 * \return a new allocated tensor
 * \sa AllocSpace
 */
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType> NewTensor(const Shape<dim> &shape,
                                            DType initv,
                                            bool pad = MSHADOW_ALLOC_PAD,
                                            Stream<Device> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<cpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \param stream the stream, when specified, the copy can exhibit asynchronize behavior
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream = NULL);
/*!
 * \brief CPU/GPU: normalize softmax: dst[i][j] = exp(energy[i][j]) /(sum_j exp(energy[i][j]))
 * \param dst destination
 * \param energy input energy
 */
template<typename DType>
inline void Softmax(Tensor<cpu, 2, DType> dst, const Tensor<cpu, 2, DType> &energy);
/*!
 * \brief CPU/GPU: normalize softmax: dst[i][j] = exp(energy[i][j]) /(sum_j exp(energy[i][j]))
 * \param dst destination
 * \param energy input energy
 */
template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> dst, const Tensor<gpu, 2, DType> &energy);

/*!
 * \brief CPU/GPU: softmax gradient
 * \param dst destination
 * \param src source output
 * \param label label info
 */
template<typename DType>
inline void SoftmaxGrad(Tensor<cpu, 2, DType> dst,
                        const Tensor<cpu, 2, DType> &src,
                        const Tensor<cpu, 1, DType> &label);
/*!
 * \brief CPU/GPU: softmax gradient
 * \param dst destination
 * \param src source output
 * \param label label info
 */
template<typename DType>
inline void SoftmaxGrad(const Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label);
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[index[i]] += src[i]
                   Called when the featuredim of src is much larger than the batchsize
 * \param dst destination
 * \param index index to take
 * \param src source output
 */
template<bool clip = true, typename IndexType, typename DType>
inline void AddTakeGrad(Tensor<cpu, 2, DType> dst,
                        const Tensor<cpu, 1, IndexType>& index,
                        const Tensor<cpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[index[i]] += src[i]
                   Called when the featuredim of src is much larger than the batchsize
 * \param dst destination
 * \param index index to take
 * \param src source output
 */
template<bool clip = true, typename IndexType, typename DType>
inline void AddTakeGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 1, IndexType>& index,
                        const Tensor<gpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[sorted[i]] += src[index[i]]
                   Called when the batchsize of src is larger than the featuredim
 * \param dst destination
 * \param sorted the sorted indices
 * \param index original index of the sorted indices
 * \param src source output
 */
template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(Tensor<cpu, 2, DType> dst,
                                  const Tensor<cpu, 1, IndexType>& sorted,
                                  const Tensor<cpu, 1, IndexType>& index,
                                  const Tensor<cpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[sorted[i]] += src[index[i]]
                   Called when the batchsize of src is larger than the featuredim
 * \param dst destination
 * \param sorted the sorted indices
 * \param index original index of the sorted indices
 * \param src source output
 */
template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(Tensor<gpu, 2, DType> dst,
                                  const Tensor<gpu, 1, IndexType>& sorted,
                                  const Tensor<gpu, 1, IndexType>& index,
                                  const Tensor<gpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Fill the values of the destination matrix to specific rows in the source matrix.
                   dst[index[i]] = src[i]
                   Will use atomicAdd in the inner implementation and the result may not be deterministic.
 * \param dst destination
 * \param index the index to accumulate value
 * \param src source output
 */
template<typename IndexType, typename DType>
inline void IndexFill(Tensor<cpu, 2, DType> dst,
                      const Tensor<cpu, 1, IndexType>& index,
                      const Tensor<cpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Fill the values of the destination matrix to specific rows in the source matrix.
                   dst[index[i]] = src[i]
                   Will use atomicAdd in the inner implementation and the result may not be deterministic.
 * \param dst destination
 * \param index the index to accumulate value
 * \param src source output
 */
template<typename IndexType, typename DType>
inline void IndexFill(Tensor<gpu, 2, DType> dst,
                      const Tensor<gpu, 1, IndexType>& index,
                      const Tensor<gpu, 2, DType> &src);
/*!
 * \brief CPU/GPU: Sort key-value pairs stored in separate places. (Stable sort is performed!)
 * \param keys the keys to sort
 * \param values the values that sorts w.r.t the key
 * \param is_ascend whether to sort key in ascending order
 */
template<typename KDType, typename VDType>
inline void SortByKey(Tensor<cpu, 1, KDType> keys, Tensor<cpu, 1, VDType> values,
                      bool is_ascend = true);
/*!
 * \brief CPU/GPU: Sort key-value pairs stored in separate places. (Stable sort is performed!)
 * \param keys the keys to sort
 * \param values the values that sorts w.r.t the key
 * \param is_ascend whether to sort key in ascending order
 */
template<typename KDType, typename VDType>
inline void SortByKey(Tensor<gpu, 1, KDType> keys, Tensor<gpu, 1, VDType> values,
                      bool is_ascend = true);
/*!
 * \brief CPU/GPU: Sort the keys within each segment. (Stable sort is performed!)
                   Segments is defined as an ascending ordered vector like [0, 0, 0, 1, 1, 2, 3, 3, 3,...]
                   We sort separately the keys labeled by 0 and 1, 2, 3, etc.
                   Currently only supports sorting in ascending order !!
 * \param values the data to sort
 * \param segments segment indicator
 */
template<typename Device, typename VDType, typename SDType>
inline void VectorizedSort(Tensor<Device, 1, VDType> values, Tensor<Device, 1, SDType> segments);

// function declarations to support expression, no need to understand them
// these functions do not need to be directly used
/*!
 * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
 * \tparam Saver specify storage method
 * \tparam R specifies the storage type of the tensor
 * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
 */
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, cpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp);
/*!
 * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
 * \tparam Saver specify storage method
 * \tparam R specifies the storage type of the tensor
 * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
 */
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, gpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, cpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, typename R,
         typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, gpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, cpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, gpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale = 1);
/*!
 * \brief CPU/GPU: 1 dimension vector dot
 * \param dst Length 1 vector, used to hold the result.
 * \param lhs Left operand vector
 * \param rhs Right operand vector
 */
template<typename Device, typename DType>
inline void VectorDot(Tensor<Device, 1, DType> dst,
                      const Tensor<Device, 1, DType> &lhs,
                      const Tensor<Device, 1, DType> &rhs);
/*!
 * \brief CPU/GPU: dst = alpha * op(lhs) op(rhs) + beta * dst
 * \param dst Length 3 tensor, used to hold the result
 * \param lhs Left operand vector
 * \param rhs Right operand vector
 * \param alpha multiplier of op(lhs)op(rhs)
 * \param beta multiplier of dst
 * \param workspace Workspace for casting DType* to DType** (batched-view), must have size >= 3 * batch_size
 */
template<bool transpose_left, bool transpose_right, typename Device, typename DType>
inline void BatchGEMM(Tensor<Device, 3, DType> dst,
                      const Tensor<Device, 3, DType> &lhs,
                      const Tensor<Device, 3, DType> &rhs,
                      DType alpha,
                      DType beta,
                      Tensor<Device, 1, DType*> workspace);
}  // namespace mshadow
// include headers
#include "./stream_gpu-inl.h"
#include "./extension.h"
#include "./expr_engine-inl.h"
#include "./tensor_cpu-inl.h"
#include "./tensor_gpu-inl.h"
#include "./io.h"
#include "./tensor_container.h"
#include "./random.h"
// add definition of scalar related operators
#ifdef MSHADOW_SCALAR_
  #error "MSHADOW_SCALAR_ must not be defined"
#endif
// enumerate all the scalar data type we aim to be good at
#define MSHADOW_SCALAR_ float
#include "./expr_scalar-inl.h"
#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ double
#include "./expr_scalar-inl.h"
#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ int32_t
#include "./expr_scalar-inl.h"
#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ int64_t
#include "./expr_scalar-inl.h"
#undef MSHADOW_SCALAR_
#define MSHADOW_SCALAR_ mshadow::half::half_t
#include "./expr_scalar-inl.h"
#undef MSHADOW_SCALAR_
#endif  // MSHADOW_TENSOR_H_
