/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.h
 * \brief key-value store interface for mxnet
 */
#ifndef MXNET_KVSTORE_H_
#define MXNET_KVSTORE_H_
#include <dmlc/io.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <atomic>
#include "./ndarray.h"
#if MXNET_USE_DIST_KVSTORE
#include "ps/ps.h"
#endif  // MXNET_USE_DIST_KVSTORE

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
   * \brief Factory function to create a new KVStore.
   * \param type The type of the kvstore,
   *   - 'local' or 'local_update_cpu' or 'local_allreduce_cpu'
   *       multi-devices on a single machine. can be also
   *   - 'device' or 'local_allreduce_device' : same to local but use gpus for kv
   *       allreduce
   *   - 'dist_*' : multi-machines
   * \return a new created KVStore.
   */
  static KVStore *Create(const char *type = "local");

  /**
   * \brief return the type
   */
  inline const std::string& type() { return type_; }

  /*!
   * \brief Initialize a list of key-value pair to the store.
   *
   * One must initalize the key before \ref Push and \ref Pull, and a key
   * should be only initialized once
   *
   * It returns after data have been initialized successfully.
   *
   * For multiple workers, all workers must call \ref Init. But only worker 0
   * (get_rank() == 0)'s values are used for initialization. So others' values
   * can be empty (but not keys). This function blocks until all workers are
   * finished. That means, any worker can push and pull on the keys now.
   *
   * \param keys a list of unique keys
   * \param values a list of values
   */
  virtual void Init(const std::vector<int>& keys,
                    const std::vector<NDArray>& values) = 0;
  /*!
   * \brief Initialize a list of key-value pair to the store.
   * \param keys a list of unique keys in string format
   * \param values a list of values
   */
  virtual void Init(const std::vector<std::string>& str_keys,
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
   * - when type == "local"
   * \code
   * for (auto& v : values) v.WaitToWrite()
   * \endcode
   *
   * - when type == "dist"
   * \code
   * Wait(keys);
   * \endcode
   *
   * One must call Init() on every key before. And the value NDArray should be
   * always has the same shape as being inited.
   *
   * \param keys the list of keys
   * \param values the list of values
   * \param priority Priority of the action.
   */
  virtual void Push(const std::vector<int>& keys,
                    const std::vector<NDArray>& values,
                    int priority = 0)  = 0;

  /*!
   * \brief push a list of key-value pairs into the store
   * \param keys the list of keys in string format
   * \param values the list of values
   * \param priority Priority of the action.
   */
  virtual void Push(const std::vector<std::string>& str_keys,
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
   * - when type == "local"
   * \code
   * for (auto& v : values) v.WaitToRead()
   * \endcode
   *
   * - when type == "dist"
   * \code
   * Wait(keys);
   * \endcode
   *
   * \param keys the list of keys
   * \param values the list of buffers for the pulled data, they should be preallocated
   * \param priority Priority of the action.
   */
  virtual void Pull(const std::vector<int>& keys,
                    const std::vector<NDArray*>& values,
                    int priority = 0) = 0;
  /*!
   * \brief pull a list of key-value pairs from the store
   * \param keys the list of keys in string format
   * \param values the list of buffers for the pulled data, they should be preallocated
   * \param priority Priority of the action.
   */
  virtual void Pull(const std::vector<std::string>& str_keys,
                    const std::vector<NDArray*>& values,
                    int priority = 0) = 0;


  /**
   * \brief the prototype of user-defined updater
   */
  typedef std::function<void(int, const NDArray&, NDArray*)> Updater;
  /*!
   * \brief set an updater
   *
   * Given a key, assume \a x is the received (pushed) value and \a y is the
   * value stored on the store node. The store updates \a y by `h(x, &y)`. The
   * default \a h is ASSIGN, namely `*y = x`.
   *
   * \param updater user-defined updater, default is assign
   */
  virtual void set_updater(const Updater& updater) {
    CHECK(updater) << "invalid updater";
    updater_ = updater;
  }

  /******************************************************
   * the following are used for multi-machines.
   ******************************************************/

  /**
   * \brief initalize ps-lite environment variables
   * \param envs key-value environment variables
   */
  static void InitPSEnv(const std::unordered_map<std::string, std::string>& envs) {
#if MXNET_USE_DIST_KVSTORE
    ps::Environment::Init(envs);
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to init parameter server's environment";
#endif  // MXNET_USE_DIST_KVSTORE
  }

  /**
   * \return whether or not this process is a worker node.
   *
   * Always returns true when type == "local"
   */
  static bool IsWorkerNode() {
#if MXNET_USE_DIST_KVSTORE
    const char* role_str = ps::Environment::Get()->find("DMLC_ROLE");
    return (role_str == nullptr) || (!strcmp(role_str, "worker"));
#else
    return true;
#endif  // MXNET_USE_DIST_KVSTORE
  }

  /**
   * \return whether or not this process is a server node.
   *
   * Always returns false when type == "local"
   */
  static bool IsServerNode() {
#if MXNET_USE_DIST_KVSTORE
    const char* role_str = ps::Environment::Get()->find("DMLC_ROLE");
    return (role_str != nullptr) && (!strcmp(role_str, "server"));
#else
    return false;
#endif  // MXNET_USE_DIST_KVSTORE
  }

  void set_barrier_before_exit(const bool barrier_before_exit) {
#if MXNET_USE_DIST_KVSTORE
    if (!IsWorkerNode()) LOG(FATAL) << "barrier_before_exit takes effect only on worker nodes";
    barrier_before_exit_ = barrier_before_exit;
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to enable barrier";
#endif
  }

  /**
   * \return whether or not this process is a scheduler node.
   *
   * Always returns false when type == "local"
   */
  static bool IsSchedulerNode() {
#if MXNET_USE_DIST_KVSTORE
    const char* role_str = ps::Environment::Get()->find("DMLC_ROLE");
    return (role_str != nullptr) && (!strcmp(role_str, "scheduler"));
#else
    return false;
#endif  // MXNET_USE_DIST_KVSTORE
  }

  /*!
   * \return The rank of this node in its group, which is in [0,
   * GroupSize).
   *
   * Always return 0 when type == "local"
   */
  virtual int get_rank() const {
    return 0;
  }

  /*!
   * \return The number of worker nodes
   */
  virtual int get_group_size() const {
    return 1;
  }

  /*!
   * \return the number of dead node(s) specified by {node_id}
   * \param node_id can be a node group or a single node
   * \param timeout a node fails to send heartbeart in {timeout} seconds
   *        will be presumed as 'dead'
   *
   * Always return 0 when type == "local"
   */
  virtual int get_num_dead_node(int node_id, int timeout = 60) const {
    return 0;
  }

  /*!
   * \brief global barrier among all worker machines
   *
   * But note that, this functions only blocks the main thread of workers until
   * all of them are reached this point. It doesn't guarantee that all
   * operations issued before are actually finished, such as \ref Push and \ref Pull.
   */
  virtual void Barrier() { }

  /**
   * \brief Send a command to all server nodes
   *
   * Send a command to all server nodes, which will make each server node run
   * \a controller
   *
   * This function returns after the command has been executed in all server nodes
   *
   * \param cmd_id the head of the command
   * \param cmd_body the body of the command
   */
  virtual void SendCommandToServers(int cmd_id, const std::string& cmd_body) { }

  /**
   * \brief the prototype of a server controller
   */
  typedef std::function<void(int, const std::string&)> Controller;

  /**
   * \brief Run as server (or scheduler)
   *
   * The behavior of a server:
   * \code
   * while(receive(x)) {
   *   if (IsCommand(x)) controller(x)
   *   else if (IsKeyValue(x)) updater(x)
   * }
   * \endcode
   *
   * \param controller the user-defined server controller
   */
  virtual void RunServer(const Controller& controller) { }

 protected:
  /**
   * \brief the user-defined  updater
   */
  Updater updater_;

  /**
   * \brief the kvstore type
   */
  std::string type_;

  /**
   * \brief whether to do barrier when finalize
   */
  std::atomic<bool> barrier_before_exit_{true};
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_H_
