/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray.h
 * \brief narray interface that dynamically schedules operations
 */
#ifndef MXNET_NARRAY_H_
#define MXNET_NARRAY_H_
#include <memory>
#include <dmlc/base.h>
#include <mshadow/tensor.h>
#include "./base.h"
#include "./storage.h"
#include "./dag_engine.h"

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
   * \param dev_mask the device type of the NArray can be cpu::kDevMask or gpu::kDevMask
   * \param dev_id the device id of the specific device, needed for GPU
   */
  NArray(const mshadow::TShape &shape, int dev_mask, int dev_id)
      : ptr_(new Chunk(shape, dev_mask, dev_id)) {
  }
  /*!
   * \brief constructing a static NArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NArray for each TBlob, 
   *  make sure the memory region is available through out the life of NArray
   * \param data the memory content of static data
   */  
  NArray(const mshadow::TBlob &data)
      : ptr_(new Chunk(data)) {
  }
  /*!
   * \return the shape of current NArray    
   */
  inline const mshadow::TShape &shape() const {
    return ptr_->data.shape_;
  }
  /*!
   * \return the device mask of NArray,
   * this indicate which device the memory resides, can be cpu::kDevMask or gpu::kDevMask
   */
  inline int dev_mask() const {
    return ptr_->data.dev_mask_;
  }
  /*! \return whether this narray is not initialized */
  inline bool is_empty() const {
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
    mshadow::TBlob data;
    /*!
     * \brief if this is true, this means the data do not come
     * from StorageManager, and do not need to be freed
     */
    bool static_data;
    /*! \brief default cosntructor */
    Chunk() : static_data(true) {
      var  = DAGEngine::Get()->NewVar();
    }
    /*! \brief construct from static data */    
    Chunk(const mshadow::TBlob &data)
        : data(data), static_data(true) {
      var  = DAGEngine::Get()->NewVar();
    }
    /*! \brief construct a new chunk */
    Chunk(const mshadow::TShape &shape, int dev_mask, int dev_id)
        : static_data(false) {
      shandle = StorageManager::Get()->Alloc(shape.Size(), dev_mask, dev_id);
      var  = DAGEngine::Get()->NewVar();
      data = mshadow::TBlob(shandle.dptr, shape, dev_mask);      
    }
    /*! \brief destructor */
    ~Chunk() {
      if (static_data) { 
        DAGEngine::Get()->PushDelete([]{}, var);
      } else {
        StorageManager::Handle h = this->shandle;
        DAGEngine::Get()->PushDelete([h] {
            StorageManager::Get()->Free(h);
          }, var);
      }
    }
  };
  /*! \brief internal data of NArray */
  std::shared_ptr<Chunk> ptr_;
  // list of friend functions
  friend NArray operator+(const NArray &lhs, const NArray &rhs);
  friend NArray operator*(const NArray &lhs, const NArray &rhs);
};

NArray operator+(const NArray &lhs, const NArray &rhs);
NArray operator*(const NArray &lhs, const NArray &rhs);

}  // namespace mxnet
#endif  // MXNET_NARRAY_H_
