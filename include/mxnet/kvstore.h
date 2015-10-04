/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.h
 * \brief key-value store interface for mxnet
 */
#ifndef MXNET_KVSTORE_H_
#define MXNET_KVSTORE_H_
#include <dmlc/io.h>
#include <vector>
#if DMLC_USE_CXX11
#include <functional>
#endif  // DMLC_USE_CXX11
#include "./ndarray.h"

namespace mxnet {
/*!
 * \brief distributed key-value store
 *
 * A distributed key-value store for data synchronization over multiple
 * devices/machines. It support user-defined updater.
 */
class KVStore {
 public:
  /*! \brief virtual destructor */
  virtual ~KVStore() {}

  /*!
   * \brief Initialize a list of key-value pair to the store.
   *
   * One should initalize the key before \ref Push and \ref Pull, and a key
   * should be only initialized once
   *
   * It returns after data initialized successfully
   *
   * \param keys a list of unique keys
   * \param values a list of values
   */
  virtual void Init(const std::vector<int>& keys,
                    const std::vector<NDArray>& values) = 0;
  /*!
   * \brief push a list of key-value pairs into the store
   *
   * If a key appears mulitple times in \a keys, then the according values will
   * be aggregated (summed) before pushing.
   *
   * The (aggregated) values are merged into the store one by one
   *
   * \code
   * updater(key, value, &value_in_store);
   * \endcode
   *
   * One can set a user-defined updater by \ref set_updater. The default updater
   * is Assign.
   *
   * This function returns after adding a push operator to the engine. Any
   * following operator requiring writing value will be blocked until the
   * actual push is finished. One can wait the push is finished by
   *
   * \code
   * for (auto& v : values) v.WaitToWrite()
   * \endcode
   *
   * One must call Init() on every key before. And the value NDArray should be
   * always has the same shape as being inited.
   *
   * \param keys the list of keys
   * \param value the list of values
   * \param priority Priority of the action.
   */
  virtual void Push(const std::vector<int>& keys,
                    const std::vector<NDArray>& values,
                    int priority = 0)  = 0;
  /*!
   * \brief pull a list of key-value pairs from the store
   *
   * One must call Init() on \a key before. And \a value should be pre-allocated
   *
   * This function returns after adding a pull operator to the engine. Any
   * following operator requiring reading value will be blocked until the
   * actual pull is finished. One can wait the pull is finished by
   *
   * \code
   * for (auto& v : values) v.WaitToRead()
   * \endcode
   *
   * \param keys the list of keys
   * \param values the list of buffers for the pulled data, they should be preallocated
   * \param priority Priority of the action.
   */
  virtual void Pull(const std::vector<int>& keys,
                    const std::vector<NDArray*>& values,
                    int priority = 0) = 0;


  /**
   * \brief the prototype of user-defined updater
   */
  typedef std::function<void(int, const NDArray&, NDArray*)> Updater;
  /*!
   * \brief returns the default updater, which is ASSIGN
   * \return The default updater
   */
  inline static Updater DefaultUpdater() {
    return [](int key, const NDArray& a, NDArray* b) { CopyFromTo(a, b); };
  }

  /*!
   * \brief set an updater
   *
   * Given a key, assume \a x is the received (pushed) value and \a y is the
   * value stored on the store node. The store updates \a y by `h(x, &y)`. The
   * default \a h is ASSIGN, namely `*y = x`.
   *
   * The updater is applied in two ways depends on whether there is an aggregator
   *
   * - yes: \a h is called after data have been aggregated over all
   * workers. Assume \f$ x_i \f$ is received from worker i. Then the server
   * first computes \f$\sum_{i=0}^n x = x_i\f$, and then applies \a h. It is often
   * used for synchronous optimization
1   *
   * - no: \a h is called every time when \a x is received from a worker. It
   * is often used for asynchronous optimization.
   *
   * \param batch true for batch, false for online
   * \param updt user-defined updater, default is assign
   */
  virtual void set_updater(Updater updater) = 0;

  /*!
   * \brief Create a new KVStore.
   * \param type The type of the kvstore.
   * \return a new created KVStore.
   */
  static KVStore *Create(const char *type = "local");

  /******************************************************
   * the following are used for multi-machines
   ******************************************************/

  /**
   * \return whether or not is in distributed computing
   */
  virtual bool IsDistributed() const {
    return false;
  }

  /**
   * \return whether or not this process is a server node
   */
  static bool IsServerNode() {
    char* role_str = getenv("DMLC_ROLE");
    return (role_str != NULL) && (!strcmp(role_str, "server"));
  }

  /**
   * \return whether or not this process is a worker node
   */
  static bool IsWorkerNode() {
    char* role_str = getenv("DMLC_ROLE");
    return (role_str != NULL) && (!strcmp(role_str, "worker"));
  }

  /*!
   * \return The rank of this node in its group, which is in [0, GroupSize).
   */
  virtual int get_rank() const {
    return 0;
  }

  /*!
   * \return The number of nodes in this group.
   */
  virtual int get_group_size() const {
    return 1;
  }

  /*!
   * \brief global barrier among all worker machines
   *
   * For example, assume there are n machines, we want to let machine 0 first
   * init the values, and then pull the inited value to all machines. Before
   * pulling, we can place a barrier to guarantee that the initialization is
   * finished.
   *
   * \code
   * // this codes run on n machines in parallel
   * if (get_rank() == 0) {
   *   Init(keys, values);
   * }
   * Barrier();
   * Pull(keys, values);
   * \endcode
   */
  virtual void Barrier() { }

  /**
   * \brief send a command to all server nodes
   *
   * Send a command to all server nodes, which will make each server node run
   * \a controller
   *
   * This function returns after the command has been executed in all server nodes
   *
   * \param head the head of the command
   * \param body the body of the command
   */
  virtual void PushCommandToServer(int head, const char* body) { }

  /**
   * \brief the prototype of a server controller
   */
  typedef std::function<void(int, const char*)> Controller;

  /*!
   * \brief return an empty controller
   */
  inline static Controller EmptyController() {
    return [](int head, const char* body) { };
  }

  /**
   * \brief Run as server
   *
   * The behavior of a server:
   * \code
   * while(receive(x)) {
   *   if (IsCommand(x)) controller(x)
   *   else if (IsKeyValue(x)) updater(x)
   * }
   * \endcode
   *
   * \param controller controller which process
   */
  virtual void RunServer(const Controller& controller = EmptyController()) { }

};
}  // namespace mxnet
#endif  // MXNET_KVSTORE_H_
