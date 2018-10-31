/*!
 *  Copyright (c) 2014 by Contributors
 * \file plain-inl.h
 * \brief support of plain packet that use the plain datatype.
 */
#ifndef MSHADOW_PACKET_PLAIN_INL_H_
#define MSHADOW_PACKET_PLAIN_INL_H_

#include "../base.h"
#include "../packet-inl.h"

namespace mshadow {
namespace packet {
template<typename DType>
struct Packet<DType, kPlain> {
 public:
  /*! \brief number of float in vector */
  static constexpr index_t size = 1;
  /*! \brief The internal data */
  DType data_;
  // enable default copy constructor
  Packet(void) {}
  // constructor from the intrinsic type
  explicit Packet(DType data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<DType, kPlain> Fill(DType s) {
    return Packet<DType, kPlain>(s);
  }
  // load from address
  MSHADOW_CINLINE static Packet<DType, kPlain> Load(const DType* src) {
    return Packet<DType, kPlain>(*src);
  }
  // load from address
  MSHADOW_CINLINE static Packet<DType, kPlain> LoadUnAligned(const DType* src) {
    return Packet<DType, kPlain>(*src);
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<DType, kPlain>& operator=(DType s) {
    data_ = s;
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(DType* dst) const {
    *dst = data_;
  }
  // get the sum of all contents
  MSHADOW_CINLINE DType Sum() const {
    return data_;
  }
};

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator+(const Packet<DType, kPlain>& lhs,
                                                const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ + rhs.data_);
}

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator-(const Packet<DType, kPlain>& lhs,
                                                const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ - rhs.data_);
}
template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator*(const Packet<DType, kPlain>& lhs,
                                                    const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ * rhs.data_);
}

template<typename DType>
MSHADOW_CINLINE Packet<DType, kPlain> operator/(const Packet<DType, kPlain>& lhs,
                                                    const Packet<DType, kPlain>& rhs) {
  return Packet<DType, kPlain>(lhs.data_ / rhs.data_);
}
}  // namespace packet
}  // namespace mshadow
#endif  // MSHADOW_PACKET_PLAIN_INL_H_
