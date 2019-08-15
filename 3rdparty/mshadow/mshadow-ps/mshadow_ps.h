/*!
 *  Copyright (c) 2014 by Contributors
 * \file mshadow_ps.h
 * \brief parameter server abstraction for mshadow tensor
 *  this is a plugin of mshadow that can be used to syncrhonize
 *  parameters across device and machines
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_H_  // NOLINT(*)
#define MSHADOW_PS_H_  // NOLINT(*)
#include <vector>
// optionally support of lambda function in C++11, if available
#if __cplusplus >= 201103L
#include <functional>
#endif  // C++11
#include "../mshadow/tensor.h"

/*! \brief whether to adapt distributed PS from parameter-server */
#ifndef MSHADOW_DIST_PS
#define MSHADOW_DIST_PS 1
#endif

/*! \brief whether to support BSP rabit API of PS*/
#ifndef MSHADOW_RABIT_PS
#define MSHADOW_RABIT_PS 1
#endif

namespace mshadow {
/*! \brief namespace of mshadow-ps */
namespace ps {
/*!
 * \brief interface of parameter server
 * \tparam xpu the device of the data lies
 * \tparam DType the type of element in the tensor
 */
template<typename xpu,
         typename DType MSHADOW_DEFAULT_DTYPE>
class ISharedModel {
 public:
  /*!
   * \brief callback function that will be executed when pull request finishes
   *        before calling the callback, the thread context is already switched
   *        to the device of pullrequest
   * \param stream the stream of callback thread, it is recommended to operate using this stream
   * \param arg the argument of callback function
   */
  typedef void (CallbackFunction) (Stream<xpu> *stream, void *arg);
  /*! \brief virtual destructor */
  virtual ~ISharedModel(void) {}
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief initialize the paramerver server client
   * \param devices specifies the possible device id
   *   to be input from Push and Pull,
   */
  virtual void Init(const std::vector<int> &devices) {}
  /*!
   * \brief initialize the paramerver server client
   * without specifying the devices, only device 0 is allowed
   */
  inline void Init(void) {
    std::vector<int> dev;
    dev.push_back(0);
    this->Init(dev);
  }
  /*!
   * \brief initialize a key with certain shape
   *  must be called before using Push/PullReq/PullWait
   *  on the corresponding key
   * \param shape the shape content of the key
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  template<int dim>
  inline void InitKey(Shape<dim> shape,
                      int key, int devid) {
    this->InitKey_(shape.FlatTo2D(), key, devid);
  }
  /*!
   * \brief wait until the pull event finishes
   * if there was no pull request, wait will directly returns
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void PullWait(int key, int devid) = 0;
  /*!
   * \brief check if the weight was correct on the current device
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  template<int dim>
  inline void CheckWeight(Tensor<xpu, dim, DType> data,
                          int key,
                          int devid) {
    this->CheckWeight_(data.FlatTo2D(), key, devid);
  }
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   */
  template<int dim>
  inline void Push(Tensor<xpu, dim, DType> data,
                   int key,
                   int devid,
                   int priority = 0) {
    this->Push_(data.FlatTo2D(), key, devid, priority);
  }
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   * \param callback the callback function that will
   *                 be invoked when the request finishes
   * \param callback_arg the argument to pass to callback
   */
  template<int dim>
  inline void PullReq(Tensor<xpu, dim, DType> data,
                      int key,
                      int devid,
                      int priority = 0,
                      CallbackFunction callback = NULL,
                      void *callback_arg = NULL) {
    this->PullReq_(data.FlatTo2D(), key,
                   devid, priority, callback, callback_arg);
  }
#if __cplusplus >= 201103L
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *  this is the c++11 version that allows lambda function as callback
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   * \param callback the callback function
   */
  template<int dim>
  inline void PullReq(Tensor<xpu, dim, DType> data,
                      int key,
                      int devid,
                      int priority,
                      std::function<void(Stream<xpu> *stream)> callback) {
    // need to allocate space, because callback can happen latter..
    auto calbk = new std::function<void(Stream<xpu> *stream)>();
    *calbk = callback;
    this->PullReq(data, key, devid, priority, InvokeLambda_, calbk);
  }
#endif  // C++11

  /*!
   * \brief set weight of corresponding key in server
   *   this is a debug function that was not necessarily
   *   implemented by the server
   * \param data the data to set
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void SetWeight_(Tensor<xpu, 2, DType> data,
                          int key,
                          int devid) = 0;
  /*!
   * \brief check if the weight matches the server side
   *   this is a debug function that was not necessarily
   *   implemented by the server
   * \param data the data to set
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void CheckWeight_(Tensor<xpu, 2, DType> data,
                            int key,
                            int devid) = 0;

 protected:
  /*!
   * \brief initialize a key with certain shape
   * \param shape the shape content of the key
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void InitKey_(Shape<2> shape,
                        int key, int devid) = 0;
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   */
  virtual void Push_(Tensor<xpu, 2, DType> data,
                     int key,
                     int devid,
                     int priority = 0) = 0;
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   * \param callback the callback function that will
   *                 be invoked when the request finishes
   * \param callback_arg the argument to pass to callback
   */
  virtual void PullReq_(Tensor<xpu, 2, DType> data,
                        int key,
                        int devid,
                        int priority,
                        CallbackFunction callback,
                        void *callback_arg) = 0;

 private:
// C++11 support for lambda prepare function
#if __cplusplus >= 201103L
  /*! \brief hack function to convert lambda to callback function */
  inline static void InvokeLambda_(Stream<xpu> *stream, void *fun) {
    auto *fp = static_cast<std::function<void(Stream<xpu> *stream)>*>(fun);
    (*fp)(stream);
    delete fp;
  }
#endif  // C++11
};
/*! \brief interface for customized mshadow server */
template<typename DType>
class IModelUpdater {
 public:
  virtual ~IModelUpdater(void) {}
  /*!
   * \brief set parameters from outside
   * \param name name of parameter
   * \param val value of parameter
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief init the model updater
   * \param rank the rank of the node
   * \param argc number of arguments
   * \param argv arguments
   */
  virtual void InitUpdater(int rank, int argc, char *argv[]) {}
  /*!
   * \brief initialize the model
   * \param key the key of data we point to
   * \param dptr the data pointer
   * \param size size of the parameter key
   */
  virtual void InitModel(int key, DType *dptr, size_t size) {
    this->InitModel_(key, Tensor<cpu, 1, DType>(dptr, Shape1(size)));
  }
  /*!
   * update the model
   * \param key the key of data we point to
   * \param dptr the data pointer
   * \param size size of the parameter key
   */
  virtual void Update(int key, DType *dptr, size_t size) {
    this->Update_(key, Tensor<cpu, 1, DType>(dptr, Shape1(size)));
  }

 protected:
  /*!
   * \brief initialize the model, user can implement this one
   *   to take advantage of tensor operations
   * \param key the key of data we point to
   * \param data the tensor data corresponding to the data we want to initialize
   */
  virtual void InitModel_(int key, Tensor<cpu, 1, DType> data) {
    LOG(FATAL) << "InitModel: not implemented";
  }
  /*!
   * \brief update the model, user can implement this one
   *    to take advantage of tensor operations
   * \param key the key of data we point to
   * \param data the tensor data corresponding to the data we want to initialize
   */
  virtual void Update_(int key, Tensor<cpu, 1, DType> data) {
    LOG(FATAL) << "InitModel: not implemented";
  }
};
/*!
 * \brief create customized server
 * this is a server defined by user
 * \return new server
 */
template<typename DType>
IModelUpdater<DType> *CreateModelUpdater(void);
}  // namespace ps
}  // namespace mshadow

#include "./ps_local-inl.h"
#include "./ps_dist-inl.h"
#include "./ps_rabit-inl.h"
namespace mshadow {
namespace ps {
/*!
 * \brief create a parameter server implementation
 * \param type the type of paramerver server
 *     can either be "local" or "dist"
 * \return the ISharedModel that can be used to synchronize weights
 */
template<typename xpu, typename DType>
inline ISharedModel<xpu, DType> *CreateSharedModel(const char *type) {
  if (!strcmp("local", type)) {
#if MSHADOW_RABIT_PS
    // allreduce on one machine pays no cost
    if (rabit::IsDistributed()) {
      return new RabitModel<xpu, DType>();
    }
#endif
    return new LocalModel<xpu, DType>();
  }
#if MSHADOW_DIST_PS
  if (!strcmp("dist", type)) return new DistModel<xpu, DType>();
#endif
  LOG(FATAL) << "unknown server type " << type;
  return NULL;
}
}  // namespace ps
}  // namespace mshadow
#endif  // MSHADOW_PS_H_  NOLINT(*)
