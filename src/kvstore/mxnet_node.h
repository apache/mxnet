/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_MXNET_NODE_H_
#define MXNET_KVSTORE_MXNET_NODE_H_
#include "ps.h"
#include "mxnet/kvstore.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief a server node on ps
 */
class MXNetServer : public ps::App {
 public:
  MXNetServer() { }
  virtual ~MXNetServer() { }

  void set_controller(const KVStore::Controller& ctrl) {
    controller_ = ctrl;
  }

  void ProcessRequest(ps::Message* request) override {
    int head = request->task.cmd();
    const char* body = request->task.msg().c_str();
    CHECK(controller_);
    controller_(head, body);
  }

 private:
  KVStore::Controller controller_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif /* MXNET_KVSTORE_MXNET_NODE_H_ */
