/*!
 *  Copyright (c) 2014 by Contributors
 * \file io.h
 * \brief definitions of I/O functions for mshadow tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_IO_H_
#define MSHADOW_IO_H_
#include "./tensor.h"

namespace mshadow {
namespace utils {
/*!
 * \brief interface of stream I/O, used to serialize data,
 *   mshadow does not restricted to only this interface in SaveBinary/LoadBinary
 *   mshadow accept all class that implements Read and Write
 */
class IStream {
 public:
  /*!
   * \brief read data from stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   * \return usually is the size of data readed
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief write data to stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}
};
}  // namespace utils
/*!
 * \brief CPU/GPU: save a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 * \param fo output binary stream
 * \param src source data file
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<cpu, dim, DType> &src);  // NOLINT(*)
/*!
 * \brief CPU/GPU: save a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 * \param fo output binary stream
 * \param src source data file
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<gpu, dim, DType> &src); // NOLINT(*)
/*!
 * \brief CPU/GPU: load a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 *       if pre_alloc is true , then space in dst is preallocated, and must have same shape of the tensor loaded
 *       if pre_alloc is false, then dst originally does not have space allocated, LoadBinary will allocate space for dst
 * \param fi output binary stream
 * \param dst destination file
 * \param pre_alloc whether space is pre-allocated, if false, space allocation will happen
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi,  // NOLINT(*)
                       Tensor<cpu, dim, DType> *dst, bool pre_alloc);
/*!
 * \brief CPU/GPU: load a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
 *       if pre_alloc is true , then space in dst is preallocated, and must have same shape of the tensor loaded
 *       if pre_alloc is false, then dst originally does not have space allocated, LoadBinary will allocate space for dst
 * \param fi output binary stream
 * \param dst destination file
 * \param pre_alloc whether space is pre-allocated, if false, space allocation will happen
 * \tparam dim dimension of tensor
 * \tparam DType type of element in tensor
 * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
 */

template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<gpu, dim, DType> *dst, bool pre_alloc);

// implementations
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<cpu, dim, DType> &src_) { // NOLINT(*)
  fo.Write(&src_.shape_, sizeof(src_.shape_));
  Tensor<cpu, 2, DType> src = src_.FlatTo2D();
  for (index_t i = 0; i < src.size(0); ++i) {
    fo.Write(src[i].dptr_, sizeof(DType) * src.size(1));
  }
}
template<int dim, typename DType, typename TStream>
inline void SaveBinary(TStream &fo, const Tensor<gpu, dim, DType> &src) { // NOLINT(*)
  // copy to CPU, then save
  Tensor<cpu, dim, DType> tmp(src.shape_);
  AllocSpace(&tmp);
  Stream<gpu> stream;
  Copy(tmp, src, &stream);
  SaveBinary(fo, tmp);
  FreeSpace(&tmp);
}
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<cpu, dim, DType> *dst_, bool pre_alloc) {
  Shape<dim> shape;
  CHECK_NE(fi.Read(&shape, sizeof(shape)), 0) << "mshadow::LoadBinary";
  if (pre_alloc) {
    CHECK_EQ(shape, dst_->shape_) << "LoadBinary, shape do not match pre-allocated shape";
  } else {
    dst_->shape_ = shape; AllocSpace(dst_);
  }
  Tensor<cpu, 2, DType> dst = dst_->FlatTo2D();
  if (dst.size(0) == 0) return;
  for (index_t i = 0; i < dst.size(0); ++i) {
    CHECK_NE(fi.Read(dst[i].dptr_, sizeof(DType) * dst.size(1)), 0) << "mshadow::LoadBinary";
  }
}
template<int dim, typename DType, typename TStream>
inline void LoadBinary(TStream &fi, // NOLINT(*)
                       Tensor<gpu, dim, DType> *dst, bool pre_alloc) {
  Tensor<cpu, dim, DType> tmp;
  LoadBinary(fi, &tmp, false);
  if (pre_alloc) {
    CHECK_EQ(tmp.shape, dst->shape_) << "LoadBinary, shape do not match pre-allocated shape";
  } else {
    dst->shape = tmp.shape; AllocSpace(dst);
  }
  Stream<gpu> stream;
  Copy(*dst, tmp, &stream);
  FreeSpace(&tmp);
}
}  // namespace mshadow
#endif  // MSHADOW_IO_H_
