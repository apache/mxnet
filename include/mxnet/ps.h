/*!
 * Copyright (c) 2015 by Contributors
 * \file ps.h
 * \brief parameter server interface for mxnet
 */
#ifndef MXNET_PS_H_
#define MXNET_PS_H_
#include "dmlc/io.h"
#include "narray.h"

#if DMLC_USE_CXX11
#include <functional>
#endif  // DMLC_USE_CXX11

namespace mxnet {
namespace ps {

/*! \brief A PS node */
class Node {
 public:
  Node() {}
  virtual ~Node() {}

  /*! \brief Gets rank of this node in its group, which is in [0, GroupSize) */
  int Rank();

  /*! \brief Get the size of this node group. */
  int GroupSize() { return IsWorker() ? NumWorkers() : NumServer(); }

  /*! \brief Returns the number of worker nodes */
  static int NumWorkers();

  /*! \brief Returns the number of server nodes */
  static int NumServers();

  /*! \brief Returns true if this process runs workers */
  static bool IsWorker();

  /*!\brief Returns true if this process only run servers */
  static bool IsServer();
};

/*!
 * \brief A PS worker node
 *
 * Worker node can push data (gradient) to the servers and pull data (aggregated
 * gradient or weight) back. A worker is bind to a particular device, namely a
 * worker can only push and pull data with the same \a device_id
 *
 * Example to implement allreduce
 * \code
 *   // on worker node:
 *   NArray data;
 *   // init data...
 *   Worker comm;
 *   comm.Push(0, data);
 *   comm.Pull(0, &data);
 *   data.Wait();
 *
 *   // on server node:
 *   Server store;
 * \endcode
 *
 * Example to implement asynchronous SGD
 * \code
 *   // on worker node:
 *   NArray weight, grad;
 *   Worker comm;
 *   if (comm.Rank() == 0) {
 *     // init weight ...
 *     comm.Push(0, weight);
 *   }
 *   comm.Pull(0, &weight);
 *   // compute grad
 *   comm.Push(0, grad);
 *
 *   // on server node:
 *   auto updater = [](const NArray& recv, NArray* weight) {
 *     if (weight->Empty()) {
 *        *weight = recv; // recv is the init weight
 *     } else {
 *        *weight += 0.1 * recv; // recv is grad
 *     }
 *   }
 *   Server store(false, updater);
 * \endcode
 */
class Worker : public Node {
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
   * For each push, each server node will apply a user-defined server handle to merge
   * the value sent to the one maintained by itself. See \ref Server for more
   * details.
   *
   * For a given \a key, the \a value should be always has the same size over
   * all workers.
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
   * System will guarantee that the all pushes issued by this worker have been
   * applied, namely the server handle has been triggered.
   *
   * \param key the key for pulling
   * \param value data for pulling, should be pre-allocated
   */
  void Pull(int key, NArray* value);
};


#if DMLC_USE_CXX11
/**
 * \brief A PS server node
 *
 * A server node maintains data (weight or aggregated gradient), and allows
 * user-defined handle to modify the data
 */
class Server : public Node {
 public:
  /**
   * \brief user-defined handle
   */
  using Handle = std::function<void(const NArray&, NArray*)>;

  /**
   * \brief constructor
   *
   * Given a key, assume \a x is the received value and \a y is the value stored
   * on the server node. The server updates \a y by `h(x, &y)`. The default \a h
   * is ASSIGN, namely `*y = x`.
   *
   * The handle is triggered in two ways:
   *
   * - online: \a h is called every time when \a x is received from a worker. It
   * is often used for asynchronous optimization.
   *
   * - batch: \a h is called after data have been aggregated over all
   * workers. Assume \f$ x_i \f$ is received from worker i. Then the server
   * first computes \f$\sum_{i=0}^n x = x_i\f$, and then applies \a h. It is often
   * used for synchronous optimization
   *
   * \param batch true for batch, false for online
   * \param h user-defined handle, default is assign
   */
  explicit Server(bool batch = true, const Handle& h = Handle());

  /**
   * \brief Load from disk
   */
  void Load(dmlc::Stream *fi);

  /**
   * \brief Save to disk
   */
  void Save(dmlc::Stream *fo);
};
#endif  // DMLC_USE_CXX11


}  // namespace ps
}  // namespace mxnet
#endif  // MXNET_PS_H_
