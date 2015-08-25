/*!
 * Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>

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
          std::this_thread::sleep_for(std::chrono::seconds{1});
          cb();
        },
        {var}, {}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  engine->WaitForAll();
  printf("Going to push delete\n");
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->PushDelete([](mxnet::RunContext){}, mxnet::Context{}, var);

  printf("============= Test #2 ==============\n");
  var = engine->NewVar();
  oprs.clear();
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::DAGEngine::Callback cb) {
          Foo(ctx, i);
          std::this_thread::sleep_for(std::chrono::milliseconds{500});
          cb();
        },
        {}, {var}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  engine->WaitForAll();
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->PushDelete([](mxnet::RunContext){}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #3 ==============\n");
  var = engine->NewVar();
  oprs.clear();
  engine->WaitForVar(var);

  return 0;
}
