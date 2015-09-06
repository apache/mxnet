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
 * Worker node can push data (gradient) to the servers and pull data (aggregated
 * gradient or weight) back. A worker is bind to a particular device, namely a
 * worker can only push and pull data with the same \a device_id
 */
class Worker {
 public:
  /*!
   * \brief push data to the server nodes
   *
   * Push the key-value pair (\a key, \a value) to the server nodes.  This
   * function returns after adding a push operator to the engine. Any following
   * operator requiring writing \a value will be blocked until the actual push is
   * finished.
   *
   * One can wait the push is finished via `data.Wait()`
   *
   * \param key the key for pushing
   * \param value the value for pushing
   */
  void Push(int key, const NArray& value);

  /*!
   * \brief pull data from the server nodes
   *
   * Pull the \a value associated with the \a key from the servers.  This
   * function returns after adding a pull operator to the engine. Any following
   * operator requiring reading \a data will be blocked until the actual pull is
   * finished.
   *
   * One can wait the pull is finished via `data.Wait()`
   *
   * \param key the key for pulling
   * \param value data for pulling, should be pre-allocated
   */
  void Pull(int key, NArray* value);
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
