#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>

#include <mxnet/engine.h>

void Foo(mxnet::RunContext, int i) { printf("The fox says %d\n", i); }

TEST(Engine, basics) {
  auto&& engine = mxnet::Engine::Get();
  auto&& var = engine->NewVariable();
  std::vector<mxnet::Engine::OprHandle> oprs;

  // Test #1
  printf("============= Test #1 ==============\n");
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::Engine::CallbackOnComplete cb) {
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
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #2 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::Engine::CallbackOnComplete cb) {
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
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);

  printf("============= Test #3 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  engine->WaitForVar(var);
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #4 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  oprs.push_back(engine->NewOperator(
      [](mxnet::RunContext ctx, mxnet::Engine::CallbackOnComplete cb) {
        std::this_thread::sleep_for(std::chrono::seconds{2});
        Foo(ctx, 42);
        cb();
      },
      {}, {var}, mxnet::FnProperty::kCopy));
  engine->Push(oprs.at(0), mxnet::Context{});
  LOG(INFO) << "IO operator pushed, should wait for 2 seconds.";
  engine->WaitForVar(var);
  LOG(INFO) << "OK, here I am.";
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #5 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  oprs.push_back(engine->NewOperator(
      [](mxnet::RunContext ctx, mxnet::Engine::CallbackOnComplete cb) {
        Foo(ctx, 42);
        std::this_thread::sleep_for(std::chrono::seconds{2});
        cb();
      },
      {var}, {}));
  engine->Push(oprs.at(0), mxnet::Context{});
  LOG(INFO) << "Operator pushed, should not wait.";
  engine->WaitForVar(var);
  LOG(INFO) << "OK, here I am.";
  engine->WaitForAll();
  LOG(INFO) << "That was 2 seconds.";
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();
  var = nullptr;
  oprs.clear();
  LOG(INFO) << "All pass";
}
