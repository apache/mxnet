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
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#error "Currently we need g++ 4.8 or higher to fully support c++11 features"
#define override
#define final
#endif
#endif

/*!
 * \brief define dllexport for Visual Studio
 */
#ifdef _MSC_VER
#ifdef MXNET_EXPORTS
#define MXNET_API __declspec(dllexport)
#else
#define MXNET_API __declspec(dllimport)
#endif
#else
#define MXNET_API
#endif

/*!
 * \brief define prediction only
 */
#ifndef MXNET_PREDICT_ONLY
#define MXNET_PREDICT_ONLY 0
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

/*! \brief Context information about the execution enviroment */
struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = cpu::kDevMask,
    kGPU = gpu::kDevMask,
    kCPUPinned = 3
  };
  /*! \brief the device type we run the op on */
  DeviceType dev_type;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
  /*! \brief default constructor */
  Context() : dev_type(kCPU), dev_id(0) {}
  /*!
   * \brief Get corresponding device mask
   * \return cpu::kDevMask or gpu::kDevMask
   */
  inline int dev_mask() const {
    if (dev_type == kCPUPinned) return cpu::kDevMask;
    return dev_type;
  }
  /*!
   * \brief Comparator, used to enable Context as std::map key.
   * \param b another context to compare
   * \return compared result
   */
  inline bool operator<(const Context &b) const;
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_type == b.dev_type && dev_id == b.dev_id;
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
  inline void Save(dmlc::Stream *strm) const {
    strm->Write(&dev_type, sizeof(dev_type));
    strm->Write(&dev_id, sizeof(dev_id));
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  inline bool Load(dmlc::Stream *strm) {
    if (strm->Read(&dev_type, sizeof(dev_type)) != sizeof(dev_type)) return false;
    if (strm->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) return false;
    return true;
  }
  /*! \brief the maximal device type */
  static const int32_t kMaxDevType = 4;
  /*! \brief the maximal device index */
  static const int32_t kMaxDevID = 16;
  /*!
   * \brief Create a new context.
   * \param dev_type device type.
   * \param dev_id device id.
   */
  inline static Context Create(DeviceType dev_type, int32_t dev_id);
  /*! \return CPU Context */
  inline static Context CPU();
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context.
   */
  inline static Context GPU(int32_t dev_id);
  /*!
   * Create a pinned CPU context.
   * \param dev_id the device id for corresponding GPU.
   * \return Pinned CPU context.
   */
  inline static Context CPUPinned(int32_t dev_id);
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
namespace mxnet {
// implementing Context
inline bool Context::operator<(const Context &b) const {
  if (dev_type == b.dev_type) {
    return dev_id < b.dev_id;
  } else {
    return dev_type < b.dev_type;
  }
}
inline Context Context::Create(DeviceType dev_type, int32_t dev_id) {
  Context ctx;
  ctx.dev_type = dev_type;
  ctx.dev_id = dev_id;
  return ctx;
}
inline Context Context::CPU() {
  return Create(kCPU, 0);
}

inline Context Context::CPUPinned(int32_t dev_id) {
  return Create(kCPUPinned, dev_id);
}

inline Context Context::GPU(int32_t dev_id) {
  return Create(kGPU, dev_id);
}
}  // namespace mxnet

#include "./tensor_blob.h"
//! \endcond
#endif  // MXNET_BASE_H_
