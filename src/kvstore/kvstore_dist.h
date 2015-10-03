/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_

#include "./kvstore_local.h"
#include "ps.h"

namespace mxnet {
namespace kvstore {

class KVStoreDist : public KVStoreLocal {
 public:

  virtual int get_group_size() const {
    return ps::NodeInfo::RankSize();
  }
  virtual int get_rank() const {
    return ps::NodeInfo::MyRank();
  }

  virtual bool is_distributed() const {
    return true;
  }
};

}  // namespace kvstore
}  // namespace mxnet


#endif /* MXNET_KVSTORE_KVSTORE_DIST_H_ */
