/*!
 * Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
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
  // Context exec_ctx;

  // Test #1
  printf("============= Test #1 ==============\n");
  auto&& var = engine->NewVar();
  std::vector<mxnet::DAGEngine::Operator> oprs;
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::DAGEngine::Callback cb) {
          Foo(ctx, i);
          cb();
        },
        {var}, {}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  std::this_thread::sleep_for(std::chrono::seconds{1});

  printf("============= Test #2 ==============\n");
  var = engine->NewVar();
  oprs.clear();
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx, mxnet::DAGEngine::Callback cb) {
          Foo(ctx, i);
          cb();
        },
        {}, {var}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  std::this_thread::sleep_for(std::chrono::seconds{1});

  // usleep(1000000);

  // // Test #2
  // cout << "============= Test #2 ==============" << endl;
  // for (int i = 0; i < 10; ++i) {
  //   engine->Push([i] (RunContext rctx) { Foo(rctx, i); },
  //       exec_ctx, {}, vars);
  // }

  // usleep(1000000);

  // // Test #3
  return 0;
}
