/*!
 *  Copyright (c) 2015 by Contributors
 * \file tensor_blob.h
 * \brief tensor blob used to hold static memory used by
 */
#ifndef MXNET_TENSOR_BLOB_H_
#define MXNET_TENSOR_BLOB_H_
#include <mshadow/tensor.h>

namespace mxnet {
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
#endif  // MXNET_TENSOR_BLOB_H_
