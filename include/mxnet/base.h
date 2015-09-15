/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuation of mxnet as well as basic data structure.
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/parameter.h>
#include <mshadow/tensor.h>
#include <string>

/*!
 *\brief whether to use opencv support
 */
#ifndef MXNET_USE_OPENCV
#define MXNET_USE_OPENCV 1
#endif

/*!
 *\brief whether to use cuda support
 */
#ifndef MXNET_USE_CUDA
#define MXNET_USE_CUDA MSHADOW_USE_CUDA
#endif

/*!
 *\brief whether to use cudnn library for convolution
 */
#ifndef MXNET_USE_CUDNN
#define MXNET_USE_CUDNN MSHADOW_USE_CUDNN
#endif

/*! \brief Error message for using gpu when MXNET_USE_CUDA==0 */
#define MXNET_GPU_NOT_ENABLED_ERROR  "GPU is not enabled"

/*!
 * \brief define compatible keywords in g++
 *  Used to support g++-4.6 and g++4.7
 */
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ == 6
#define override
#define final
#endif
#endif

/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief mxnet cpu */
typedef mshadow::cpu cpu;
/*! \brief mxnet gpu */
typedef mshadow::gpu gpu;
/*! \brief index type usually use unsigned */
typedef mshadow::index_t index_t;
/*! \brief data type that will be used to store ndarray */
typedef mshadow::default_real_t real_t;

/*! \brief dynamic shape type */
typedef mshadow::TShape TShape;
/*! \brief storage container type */
typedef mshadow::TBlob TBlob;

/*! \brief Context information about the execution enviroment */
struct Context {
  /*! \brief the device type we run the op can be cpu::kDevMask or gpu::kDevMask */
  int32_t dev_mask;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
  /*! \brief constructor */
  Context() : dev_mask(cpu::kDevMask), dev_id(0) {}
  /*!
   * \brief constructor of context
   * \param dev_mask the device mask
   * \param dev_id the device id
   */
  Context(int dev_mask, int dev_id)
      : dev_mask(dev_mask), dev_id(dev_id) {}
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_mask == b.dev_mask && dev_id == b.dev_id;
  }
  /*!
   * \brief check if current context not equals another one
   * \param b another context to compare
   * \return whether they are not the same
   */
  inline bool operator!=(const Context &b) const {
    return !(*this == b);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const {
    strm->Write(&dev_mask, sizeof(dev_mask));
    strm->Write(&dev_id, sizeof(dev_id));
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm) {
    if (strm->Read(&dev_mask, sizeof(int32_t)) != sizeof(int32_t)) return false;
    if (strm->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) return false;
    return true;
  }
  /*! \brief the maximal device mask, cpu = 1, gpu = 2 */
  static const int32_t kMaxDevMask = 2;
  /*!
   * \brief A dedicate ID for pinned cpu memory.
   *  Any normal CPU ID should be less than this number.
   */
  static const int32_t kPinnedMemoryID = 16;
};

/*!
 * \brief execution time context.
 *  The information needed in runtime for actual execution.
 */
struct RunContext {
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return static_cast<mshadow::Stream<xpu>*>(stream);
  }
};
}  // namespace mxnet

//! \cond Doxygen_Suppress
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
  inline FieldEntry<mxnet::TShape> &set_expect_ndim(mshadow::index_t ndim) {
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
//! \endcond
#endif  // MXNET_BASE_H_
