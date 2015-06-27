/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray.h
 * \brief narray interface that dynamically schedules operations
 */
#ifndef MXNET_NARRAY_H_
#define MXNET_NARRAY_H_
#include <memory>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include "./base.h"
#include "./storage.h"
#include "./tensor_blob.h"
#include "./dag_engine.h"
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for narray module"
#endif

namespace mxnet {
/*!
 * \brief ndarray interface
 */
class NArray {
 public:
  /*! \brief default cosntructor */
  NArray() {}
  /*!
   * \brief constructing a new dynamic NArray
   * \param shape the shape of array
   * \param ctx context of NArray
   */
  NArray(const TShape &shape, Context ctx)
      : ptr_(new Chunk(shape, ctx, false)) {
  }
  /*!
   * \brief constructing a static NArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NArray for each TBlob,
   *  make sure the memory region is available through out the life of NArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NArray(const TBlob &data, int dev_id)
      : ptr_(new Chunk(data, dev_id)) {
  }
  /*!
   * \return the shape of current NArray
   */
  inline const TShape &shape() const {
    return ptr_->data.shape_;
  }
  /*!
   * \return the context of NArray, this function is only valid when the NArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*! \return whether this narray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }

 private:
  /*! \brief the real data chunk that backs NArray */
  struct Chunk {
    /*! \brief storage handlefrom storage engine */
    StorageManager::Handle shandle;
    /*! \brief variable from DAG engine */
    DAGEngine::Variable var;
    /*! \brief holds the data content */
    TBlob data;
    /*!
     * \brief if this is true, this means the data do not come
     * from StorageManager, and do not need to be freed
     */
    bool static_data;
    /*! \brief whether allocation is delayed */
    bool delay_alloc;
    /*! \brief default cosntructor */
    Chunk() : static_data(true), delay_alloc(false) {
      var  = DAGEngine::Get()->NewVar();
    }
    /*! \brief construct from static data */
    Chunk(const TBlob &data, int dev_id)
        : data(data),
          static_data(true),
          delay_alloc(false) {
      var = DAGEngine::Get()->NewVar();
      shandle.ctx = Context(data.dev_mask_, dev_id);
    }
    /*! \brief construct a new chunk */
    Chunk(const TShape &shape, Context ctx, bool delay_alloc_)
        : static_data(false), delay_alloc(true) {
      var = DAGEngine::Get()->NewVar();
      data.shape_ = shape;
      shandle.ctx = ctx;
      if (!delay_alloc_) this->CheckAndAlloc();
    }
    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = StorageManager::Get()->Alloc(data.shape_.Size() * sizeof(real_t), shandle.ctx);
        data = TBlob(static_cast<real_t*>(shandle.dptr), data.shape_, shandle.ctx.dev_mask);
        delay_alloc = false;
      }
    }
    /*! \brief destructor */
    ~Chunk() {
      if (static_data) {
        DAGEngine::Get()->PushDelete([](RunContext s) {}, var);
      } else {
        CHECK(!delay_alloc) << "deleted before allocation";
        StorageManager::Handle h = this->shandle;
        DAGEngine::Get()->PushDelete([h](RunContext s) {
            StorageManager::Get()->Free(h);
          }, var);
      }
    }
  };
  /*! \brief internal data of NArray */
  std::shared_ptr<Chunk> ptr_;
  /*!
   * \brief constructing a new dynamic NArray
   * \param shape the shape of array
   * \param ctx context of NArray
   * \param delay_alloc whether delay the allocation
   */
  NArray(const TShape &shape, Context ctx, bool delay_alloc)
      : ptr_(new Chunk(shape, ctx, delay_alloc)) {
  }
  // add friend to helper functions
  template<typename OP>
  friend void BinaryEWise(const NArray &lhs, const NArray &rhs, NArray *out);
};
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator+(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator-(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator*(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator/(const NArray &lhs, const NArray &rhs);
}  // namespace mxnet
#endif  // MXNET_NARRAY_H_
