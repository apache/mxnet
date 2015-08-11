/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuation of mxnet
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_
#include <dmlc/base.h>
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

/*! \brief option to pass into the forward function */
struct Option {
  /*! \brief whether it is training phase*/
  int is_train;
};
/*! \brief gradient request type the request can have */
enum GradReqType {
  /*! \brief no operation, do not write gradient */
  kNullOp = 0,
  /*! \brief write gradient to provided space */
  kWriteTo = 1,
  /*! \brief same as kWriteTo, but provided space is same as space of input-data */
  kWriteInplace = 2,
  /*! \brief add to the provided space */
  kAddTo = 3
};
/*! \brief input argument type of the operator have */
enum ArgType {
  /*! \brief data argument */
  kDataArg = 0,
  /*! \brief weight argument */
  kWeightArg = 1,
  /*! \brief bias argument */
  kBiasArg = 2
};
/*! \brief Property for engine schedule */
enum Property {
  /*! \brief Op contains interanl state, won't influence engine schedule */
  kContainInteralState = 1,
  /*! \brief Op forward require random number, will influence engine schedule */
  kForwardRequireRnd = 2,
};

/*! \brief context information about the execution enviroment */
struct Context {
  /*! \brief the device type we run the op can be cpu::kDevMask or gpu::kDevMask */
  int dev_mask;
  /*! \brief device id we are going to run it on */
  int dev_id;
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
};


/*!
 * \brief execution context provides the information needed
 *  in runtime to actually execute the operation
 */
struct RunContext {
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
};

/*! \brief dynamic shape type */
typedef mshadow::TShape TShape;
/*! \brief storage container type */
typedef mshadow::TBlob TBlob;

}  // namespace mxnet
#endif  // MXNET_BASE_H_
