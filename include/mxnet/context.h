/*!
 *  Copyright (c) 2015 by Contributors
 * \file context.h
 * \brief Context information and resources in mxnet.
 */
#ifndef MXNET_CONTEXT_H_
#define MXNET_CONTEXT_H_

#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include "./base.h"

namespace mxnet {

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
};

/*!
 * \brief Additional resources
 */
struct Resource {
  /*! \brief Resource type, indicating what the pointer type is */
  enum Type {
    /*! \brief mshadow::Random<xpu> object */
    kRandom,
    /*! \brief Temporal space */
    kTempSpace
  };
  /*! \brief pointer to the resource */
  void *ptr;
};

/*!
 * \brief The resources that can be requested by Operator
 */
struct ResourceRequest {
  /*! \brief type of resources */
  Resource::Type type;
  /*! \brief size requirment if it is an temp space request */
  size_t space_size;
  /*! \brief default constructor */
  ResourceRequest() {}
  /*!
   * \brief default constructor, allow implicit conversion
   * \param type type of resources
   */
  ResourceRequest(Resource::Type type) : type(type) {}  // NOLINT(*)
};

}  // namespace mxnet
#endif  // MXNET_CONTEXT_H_
