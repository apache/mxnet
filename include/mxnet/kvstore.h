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
#include "narray.h"

namespace mxnet {

/**
 * \brief distributed key-value store
 *
 * A distributed key-value store for data synchronization over multiple
 * devices/machines. It supports aggregator and user-defined updater.
 */
class KVStore {
 public:
  /**
   * \brief get singleton instance
   */
  static KVStore* Get() { static KVStore store; return &store; }

  /**
   * \brief Init with the local devices
   */
  virtual void InitDevices(const std::vector<Context>& devices);

  /**
   * \brief  data
   *
   * Initialize a key-value pair to the store. For any \a key, this function
   * should only be called once.
   */
  virtual void Init(int key, const NArray& value) {
    get_impl()->Init(key, value);
  }

  /*!
   * \brief push data to the store
   *
   * Push the key-value pair (\a key, \a value) to the store.  This
   * function returns after adding a push operator to the engine. Any following
   * operator requiring writing \a value will be blocked until the actual push is
   * finished.
   *
   * One can wait the push is finished via `data.WaitToWrite()`
   *
   * For each push, an updater is called to merge the value to the one
   * stored. The default updater is Assign.
   *
   * One must call Init() on \a key before. And the \a value should be always
   * has the same size as being inited.
   *
   * \param key the key for pushing
   * \param value the value for pushing
   */
  virtual void Push(int key, const NArray& value) {
    get_impl()->Push(key, value);
  }

  /*!
   * \brief pull data from the store
   *
   * Pull the \a value associated with the \a key from the store.  This
   * function returns after adding a pull operator to the engine. Any following
   * operator requiring reading \a data will be blocked until the actual pull is
   * finished.
   *
   * One can wait the pull is finished via `data.WaitToRead()`
   *
   * Before sending back the value, the store will wait all pushed issued by
   * this worker on \a key have been applied (updater has been applied). One
   * must call Init() on \a key before.
   *
   * \param key the key for pulling
   * \param value data for pulling, should be pre-allocated
   */
  virtual void Pull(int key, NArray* value) {
    get_impl()->Pull(key, value);
  }

  /**
   * \brief clear all key-value pairs stored, updater, and devices binded
   */
  virtual void Stop() { get_impl()->Stop(); delete impl_; impl_ = NULL; }

#if DMLC_USE_CXX11
  /**
   * \brief the prototype of user-defined updater
   */
  using Updater = std::function<void(const NArray&, NArray*)>;

  /*! \brief returns the default updater, which is ASSIGN */
  Updater DefaultUpdater() {
    return [](const NArray& a, NArray* b) { CopyFromTo(a, b); };
  }

  /**
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
   *
   * - no: \a h is called every time when \a x is received from a worker. It
   * is often used for asynchronous optimization.
   *
   * \param batch true for batch, false for online
   * \param updt user-defined updater, default is assign
   */
  virtual void set_updater(const Updater& updater) {
    get_impl()->set_updater(updater);
  }

#endif  // DMLC_USE_CXX11

  /**
   * \brief set aggregator
   *
   * The aggregator is enabled in default
   *
   * \param aggregator false to disable
   */
  virtual void set_aggregator(bool aggregator) {
    get_impl()->set_aggregator(aggregator);
  }

  /*!
   * \brief Gets rank of this node in its group, which is in [0, GroupSize).
   */
  virtual int get_rank() const {
    return get_impl()->get_rank();
  }


  /*!
   * \brief Get the number of nodes in this group.
   */
  virtual int get_group_size() const {
    return get_impl()->get_group_size();
  }

 protected:
  KVStore() : impl_(NULL) { }
  virtual ~KVStore() { delete impl_; impl_ = NULL; }

 private:
  inline KVStore* get_impl() const {
    CHECK(impl_) << "call InitDevices() first";
    return impl_;
  }
  KVStore* impl_;
  DISALLOW_COPY_AND_ASSIGN(KVStore);
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_H_
