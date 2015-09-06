/*!
 * Copyright (c) 2015 by Contributors
 * \file ps.h
 * \brief parameter server interface for mxnet
 */
#ifndef MXNET_PS_H_
#define MXNET_PS_H_
#include "dmlc/io.h"
#include "narray.h"

#if DMLC_USE_CXX11 == 0
#error "C++11 was required for ps module."
#endif

namespace mxnet {
namespace ps {

/*!
 * \brief A PS worker node
 *
 * a worker node can push data (gradient) to the servers and also pull data
 * (weight) back
 */
class Worker {
 public:
  /*!
   * \brief push \a data to the server nodes
   *
   * This function returns after adding a push operator to the engine. Any
   * following operator requiring writing \a data will be blocked until the
   * actual push is finished.
   *
   * \param data data for pushing
   */
  void Push(const NArray& data);

  /*!
   * \brief pull data from the server nodes
   *
   * This function returns after adding a pull operator to the engine. Any
   * following operator requiring reading \a data will be blocked until the
   * actual pull is finished.
   *
   * \param data data for pulling, should be pre-allocated
   */
  void Pull(NArray& data);

  /**
   * \brief wait until a push/pull finished
   *
   * Wait until data has already pushed to the servers or pulled back from the
   * servers
   *
   * \param data data for waiting
   */
  void Wait(const NArray& data);
};



/**
 * \brief A PS server node
 *
 * a server node maintains data (weight), and allows user-defined handle to
 * modify the data
 */
class Server {
 public:
  /**
   * \brief constructor
   *
   * The server node triggers the user-defined handle in two ways:
   * - online: the handle is called every time when data received from a
   * worker. often used for asynchronous optimization
   * - batch: the handle is called after data have been aggregated over all
   * workers. often used for synchronous optimization
   *
   * \param batch true for batch, false for online
   */
  explicit Server(bool batch = true);

  /**
   * \brief Load from disk
   */
  void Load(dmlc::Stream *fi);

  /**
   * \brief Save to disk
   */
  void Save(dmlc::Stream *fo);
};

/**
 * \brief user-defined handle
 * \param recv_data data (gradient) received from users
 * \param my_data data (weight) maintained on the server
 */
void ServerHandle(const NArray& recv_data, NArray my_data);


}  // namespace ps
}  // namespace mxnet
#endif  // MXNET_PS_H_
