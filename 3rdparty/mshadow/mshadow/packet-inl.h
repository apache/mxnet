/*!
 *  Copyright (c) 2014 by Contributors
 * \file packet-inl.h
 * \brief Generic packet vectorization code
 */
#ifndef MSHADOW_PACKET_INL_H_
#define MSHADOW_PACKET_INL_H_

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include "./base.h"
#include "./tensor.h"
#include "./expression.h"


namespace mshadow {
/*! \brief namespace of packet math*/
namespace packet {

enum PacketArch {
  kPlain,
  kSSE2,
};

#if MSHADOW_USE_SSE
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kSSE2
#else
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kPlain
#endif

// whether packet operator is enabled.
/*!
 * \brief Generic packet type
 * \tparam DType The data type of the packet.
 * \tparam Arch the Arch of the packet.
 */
template<typename DType, PacketArch Arch = MSHADOW_DEFAULT_PACKET>
struct Packet;

template<PacketArch Arch>
struct AlignBytes {
  static const index_t value = 4;
};

}  // namespace packet
}  // namespace mshadow

namespace mshadow {
namespace packet {
/*!
 * \brief analog to cudaMallocPitch, allocate a aligned space with num_line * lspace cells
 * \param out_pitch output parameter, the actuall space allocated for each line
 * \param lspace number of cells required for each line
 * \param num_line number of lines to be allocated
 */
inline void* AlignedMallocPitch(size_t *out_pitch,
                                size_t lspace,
                                size_t num_line) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t mask = (1 << bits) - 1;

  size_t pitch = ((lspace + mask) >> bits) << bits;
  *out_pitch = pitch;
#ifdef _MSC_VER
  void *res = _aligned_malloc(pitch * num_line, 1 << bits);
#else
  void *res;
  int ret = posix_memalign(&res, 1 << bits, pitch * num_line);
  CHECK_EQ(ret, 0) << "AlignedMallocPitch failed";
#endif
  if (res == NULL) {
    LOG(FATAL) << "AlignedMallocPitch failed";
  }
  return res;
}

/*!
 * \brief free aligned space
 * \param ptr pointer to space to be freed
 */
inline void AlignedFree(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/*! \brief check if a pointer is aligned */
template<PacketArch Arch>
inline bool CheckAlign(size_t pitch) {
  const index_t bits = AlignBytes<Arch>::value;
  return !(pitch & ((1 << bits) - 1));
}

/*! \brief check if a pointer is aligned */
template<PacketArch Arch>
inline bool CheckAlign(void *ptr) {
  return CheckAlign<Arch>(reinterpret_cast<size_t>(ptr));
}

/*!
 * \brief get upper bound of aligned index of size
 * \param size size of the array
 * \param fsize size of float
 */
template<typename DType, PacketArch Arch>
inline index_t UpperAlign(index_t size) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t mask = (1 << bits) - 1;
  const index_t fsize = sizeof(DType);
  return (((size * fsize + mask) >> bits) << bits) / fsize;
}

/*!
 * \brief get lower bound of aligned index of size
 * \param size size of the array
 * \param fsize size of float
 */
template<typename DType, PacketArch Arch>
inline index_t LowerAlign(index_t size) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t fsize = sizeof(DType);
  return (((size * fsize) >> bits) << bits) / fsize;
}

/*!
 * \brief generic Packet operator
 * \tparam OP The operator
 * \tparam DType The data type
 * \tparam Arch The architecture.
 */
template<typename OP, typename DType, PacketArch Arch>
struct PacketOp {
  static const bool kEnabled = false;
};
// specialization of operators
template<typename DType, PacketArch Arch>
struct PacketOp<op::plus, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                   const Packet<DType, Arch>& rhs) {
    return lhs + rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::minus, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs - rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::mul, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs * rhs;
  }
};
template<typename DType, PacketArch Arch>
struct PacketOp<op::div, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& lhs,
                                                  const Packet<DType, Arch>& rhs) {
    return lhs / rhs;
  }
};

template<typename DType, PacketArch Arch>
struct PacketOp<op::identity, DType, Arch> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static Packet<DType, Arch> Map(const Packet<DType, Arch>& src) {
    return src;
  }
};


// savers to do storage
template<typename SV, typename TFloat, PacketArch Arch>
struct Saver{
  MSHADOW_CINLINE static void Save(TFloat *dst, const Packet<TFloat, Arch>& src) {
    Packet<TFloat, Arch> lhs = Packet<TFloat, Arch>::Load(dst);
    Packet<TFloat, Arch> ans = PacketOp<typename SV::OPType, TFloat, Arch>::Map(lhs, src);
    ans.Store(dst);
  }
};
template<typename TFloat, PacketArch Arch>
struct Saver<sv::saveto, TFloat, Arch> {
  MSHADOW_CINLINE static void Save(TFloat *dst, const Packet<TFloat, Arch>& src) {
    src.Store(dst);
  }
};
}  // namespace packet
}  // namespace mshadow

#include "packet/plain-inl.h"
#if MSHADOW_USE_SSE && !defined(__CUDACC__)
#include "packet/sse-inl.h"
#endif

namespace mshadow {
namespace expr {

typedef packet::PacketArch PacketArch;

// same as plan, but use packet
template<typename ExpType, typename DType, PacketArch Arch>
class PacketPlan {
 public:
  /*!
   * \brief evaluate the expression at index [y][x],
   * x will be aligned to Packet<DType, Arch>::Size()
   */
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const;
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const;
};

template <typename Device, int dim, typename DType, PacketArch Arch>
class PacketPlan<Tensor<Device, dim, DType>, DType, Arch> {
 public:
  explicit PacketPlan(const Tensor<Device, dim, DType> &t)
      :dptr_(t.dptr_), stride_(t.stride_) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::Packet<DType, Arch>::Load(&dptr_[y * stride_ + x]);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

 private:
  const DType  *dptr_;
  index_t stride_;
};

template<typename DType, PacketArch Arch>
class PacketPlan<ScalarExp<DType>, DType, Arch> {
 public:
  explicit PacketPlan(DType scalar) : scalar_(scalar) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::Packet<DType, Arch>::Fill(scalar_);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return scalar_;
  }

 private:
  DType scalar_;
};

template<typename OP, typename TA, typename TB, int etype, typename DType, PacketArch Arch>
class PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch> {
 public:
  PacketPlan(const PacketPlan<TA, DType, Arch> &lhs, const PacketPlan<TB, DType, Arch> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_CINLINE packet::Packet<DType, Arch> EvalPacket(index_t y, index_t x) const {
    return packet::PacketOp<OP, DType, Arch>::Map(lhs_.EvalPacket(y, x), rhs_.EvalPacket(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(lhs_.Eval(y, x), rhs_.Eval(y, x));
  }

 private:
  PacketPlan<TA, DType, Arch> lhs_;
  PacketPlan<TB, DType, Arch> rhs_;
};

template<typename OP, typename TA, int etype, typename DType, PacketArch Arch>
class PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch> {
 public:
  PacketPlan(const PacketPlan<TA, DType, Arch> &src) : src_(src) {}
  MSHADOW_CINLINE packet::Packet<DType> EvalPacket(index_t y, index_t x) const {
    return packet::PacketOp<OP, DType, Arch>::Map(src_.EvalPacket(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

 private:
  PacketPlan<TA, DType, Arch> src_;
};

template<PacketArch Arch, typename OP, typename TA, typename TB, typename DType, int etype>
inline PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch>
MakePacketPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template<PacketArch Arch, typename DType>
inline PacketPlan<ScalarExp<DType>, DType, Arch> MakePacketPlan(const ScalarExp<DType> &e) {
  return PacketPlan<ScalarExp<DType>, DType, Arch>(e.scalar_);
}
template<PacketArch Arch, typename T, typename DType>
inline PacketPlan<T, DType, Arch> MakePacketPlan(const RValueExp<T, DType> &e) {
  return PacketPlan<T, DType, Arch>(e.self());
}
template<PacketArch Arch, typename T, int dim, typename DType>
inline PacketPlan<T, DType, Arch>
MakePacketPlan(const MakeTensorExp<T, cpu, dim, DType> &e) {
  return PacketPlan<T, DType, Arch>(e.real_self());
}
template<PacketArch Arch, typename OP, typename TA, typename DType, int etype>
inline PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch>
MakePacketPlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return PacketPlan<UnaryMapExp<OP, TA, DType, etype>, DType, Arch>(MakePacketPlan<Arch>(e.src_));
}
template<PacketArch Arch, typename OP, typename TA, typename TB, typename DType, int etype>
inline PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType, Arch>
MakePacketPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return PacketPlan<BinaryMapExp<OP, TA, TB, DType, etype>,
                    DType, Arch>(MakePacketPlan<Arch>(e.lhs_), MakePacketPlan<Arch>(e.rhs_));
}

/*!
 * \brief static check packet enable
 *
 * \tparam Device the type of Device
 * \tparam dim dimension of the tensor
 * \tparam E expression
 */
template<typename E, PacketArch Arch>
struct PacketCheck{
  static const bool kPass = false;
};
template<PacketArch Arch>
struct PacketCheck<float, Arch> {
  static const bool kPass = true;
};
template<PacketArch Arch>
struct PacketCheck<double, Arch> {
  static const bool kPass = true;
};
template<typename DType, PacketArch Arch>
struct PacketCheck<ScalarExp<DType>, Arch> {
  static const bool kPass = PacketCheck<DType, Arch>::kPass;
};
template<int dim, typename DType, PacketArch Arch>
struct PacketCheck<Tensor<cpu, dim, DType>, Arch> {
  static const bool kPass = PacketCheck<DType, Arch>::kPass;
};
template<typename OP, typename TA, typename DType, int etype, PacketArch Arch>
struct PacketCheck<UnaryMapExp<OP, TA, DType, etype>, Arch> {
  static const bool kPass = PacketCheck<TA, Arch>::kPass &&
      packet::PacketOp<OP, DType, Arch>::kEnabled;
};
template<typename OP, typename TA, typename TB, typename DType, int etype, PacketArch Arch>
struct PacketCheck< BinaryMapExp<OP, TA, TB, DType, etype>, Arch> {
  static const bool kPass = packet::PacketOp<OP, DType, Arch>::kEnabled &&
      PacketCheck<TA, Arch>::kPass && PacketCheck<TB, Arch>::kPass;
};
//----------------------------------------------------
// Check if data is aligned and allow packet operation
//----------------------------------------------------
template<int dim, typename E, PacketArch Arch>
struct PacketAlignCheck {
  inline static bool Check(const E &exp) {
    return false;
  }
};
template<int dim, typename DType, PacketArch Arch>
struct PacketAlignCheck<dim, ScalarExp<DType>, Arch> {
  inline static bool Check(const ScalarExp<DType> &exp) {
    return true;
  }
};
template<int dim, typename DType, PacketArch Arch>
struct PacketAlignCheck<dim, Tensor<cpu, dim, DType>, Arch> {
  inline static bool Check(const Tensor<cpu, dim, DType> &t) {
    return packet::CheckAlign<Arch>(t.dptr_) &&
        packet::CheckAlign<Arch>(t.stride_ * sizeof(DType));
  }
};
template<int dim, typename OP, typename TA, typename DType, int etype, PacketArch Arch>
struct PacketAlignCheck<dim, UnaryMapExp<OP, TA, DType, etype>, Arch> {
  inline static bool Check(const UnaryMapExp<OP, TA, DType, etype> &t) {
    return PacketAlignCheck<dim, TA, Arch>::Check(t.src_);
  }
};
template<int dim, typename OP, typename TA, typename TB,
         typename DType, int etype, PacketArch Arch>
struct PacketAlignCheck<dim, BinaryMapExp<OP, TA, TB, DType, etype>, Arch> {
  inline static bool Check(const BinaryMapExp<OP, TA, TB, DType, etype> &t) {
    return PacketAlignCheck<dim, TA, Arch>::Check(t.lhs_) &&
        PacketAlignCheck<dim, TB, Arch>::Check(t.rhs_);
  }
};

/*!
 * \brief use PacketPlan to compute result
 */
template<typename SV, typename E, int dim, typename DType, PacketArch Arch>
inline void MapPacketPlan(Tensor<cpu, dim, DType> _dst,
                          const expr::PacketPlan<E, DType, Arch>& plan) {
  Tensor<cpu, 2, DType> dst = _dst.FlatTo2D();
  const index_t xlen = packet::LowerAlign<DType, Arch>(dst.size(1));
  const size_t packetSize = packet::Packet<DType, Arch>::size;
#ifndef __CUDACC__
  #pragma omp parallel for
#endif
  for (openmp_index_t y = 0; y < dst.size(0); ++y) {
    for (index_t x = 0; x < xlen; x += packetSize) {
      packet::Saver<SV, DType, Arch>::Save(&dst[y][x], plan.EvalPacket(y, x));
    }
    for (index_t x = xlen; x < dst.size(1); ++x) {
      SV::Save(dst[y][x], plan.Eval(y, x));
    }
  }
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_PACKET_INL_H_
