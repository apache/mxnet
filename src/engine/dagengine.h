#include <stdint.h>
#include <functional>
#pragma once

#include <vector>

typedef uint64_t NodeId;

class DAGEngine {
protected:
  typedef std::function<void()> Op;
  struct DAGNode {
    NodeId id;
    Op op;
    DAGNode * children;
  };

public:
  // push a new operation to execute, specifying preceeding nodes
  virtual NodeId PushNode(Op op, std::vector<NodeId> & pre) = 0;
  // wait for one node to finish
  virtual void WaitForNode(NodeId id) = 0;
  // wait for all nodes
  virtual void WaitForAll() = 0;

private:
  // worker thread will pop nodes and call its execute()
  //virtual DAGNode * PopNodeForExecute() = 0;
  // after node finish
  //virtual void FinishNode(DAGNode * node) = 0;
};

class SingleThreadDAGEngine : public DAGEngine {
public:
  // push a new operation to execute, specifying preceeding nodes
  NodeId PushNode(Op op, std::vector<NodeId> & pre) override {
    op();
    return currNodeId++;
  }

  // wait for one node to finish
  void WaitForNode(NodeId id) override
  {}

  // wait for all nodes
  void WaitForAll() override
  {}
private:
  NodeId currNodeId = 1;  // default dependency id starts from 0
};

inline void WaitForAll() {}