/*!
 * Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>
#include <dmlc/logging.h>

#include "mxnet/dag_engine.h"

void Foo(mxnet::RunContext, int i) {
  printf("The fox says %d\n", i);
}

int main() {
  auto&& engine = mxnet::DAGEngine::Get();
  auto&& var = engine->NewVar();
  std::vector<mxnet::DAGEngine::Operator> oprs;
  LOG(INFO) << "pointer " << var;

  // Test #1
  printf("============= Test #1 ==============\n");
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::DAGEngine::Callback cb) {
          Foo(ctx, i);
          cb();
        },
        {var}, {}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  engine->WaitForAll();
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  engine->PushDelete([](mxnet::RunContext){}, mxnet::Context{}, var);

  printf("============= Test #2 ==============\n");
  var = engine->NewVar();
  oprs.clear();
  for (int i = 0; i < 2; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::DAGEngine::Callback cb) {
          Foo(ctx, i);
          cb();
        },
        {}, {var}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  engine->WaitForAll();
  engine->PushDelete([](mxnet::RunContext){}, mxnet::Context{}, var);
  engine->WaitForAll();

  return 0;
}
