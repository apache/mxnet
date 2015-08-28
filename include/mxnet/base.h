/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuation of mxnet
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_

#include <dmlc/base.h>
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
#define MXNET_USE_CUDNN 0
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
}  // namespace mxnet


//! \cond Doxygen_Suppress
namespace dmlc {
// Add a few patches to support TShape in dmlc/parameter.
DMLC_DECLARE_TYPE_NAME(mxnet::TShape, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(uint32_t, "unsigned int");


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
