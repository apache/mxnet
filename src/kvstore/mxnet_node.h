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
 * \brief encode/decode a command
 */
struct KVStoreCommand {
  KVStoreCommand() : cmd(0) { }
  KVStoreCommand(int cmd_) : cmd(cmd_) { }
  int cmd;
  void set_barrier(int count) {
    cmd = - count - 10;
  }

  bool get_barrier(int *count) {
    if (cmd <= 10) {
      *count = - cmd - 10;
      return true;
    }
    return false;
  }
};

#define APP_ID 10
/**
 * \brief a server node on ps
 */
class MXNetServer : public ps::App {
 public:
  MXNetServer() : App(APP_ID) { }
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

/**
 * \brief a worker node on ps
 */
class MXNetWorker : public ps::App {
 public:
  MXNetWorker() : App(APP_ID) { }
  virtual ~MXNetWorker() { }

};

/**
 * \brief a scheduler node on ps
 */
class MXNetScheduler : public ps::App {
 public:
  MXNetScheduler() : App(APP_ID) { }
  virtual ~MXNetScheduler() { }

  void ProcessRequest(ps::Message* request) override {
    KVStoreCommand cmd(request->task.cmd());
    int count;
    if (cmd.get_barrier(&count)) {
      if (barrier_.msgs.empty()) {
        barrier_.barrier_count = count;
      } else {
        CHECK_EQ(barrier_.barrier_count, count);
      }
      barrier_.msgs.push_back(*request);
      // disable automatical reply so the worker node will be sleeped on node->Wait()
      request->replied = true;

      if (++barrier_.num_nodes == ps::NodeInfo::NumWorkers()) {
        for (auto& m : barrier_.msgs) {
          Reply(&m);
        }
        barrier_.clear();
      }
    }
  }

 private:
  // a simple barrier. probably move to ps-lite later
  struct Barrier {
    Barrier() { clear(); }
    void clear() {
      barrier_count = -1;
      num_nodes = 0;
      msgs.clear();
    }
    int num_nodes;
    int barrier_count;
    std::vector<ps::Message> msgs;
  };

  Barrier barrier_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif /* MXNET_KVSTORE_MXNET_NODE_H_ */
