// Copyright (c) 2015 by Contributors
#include <unistd.h>
#include <iostream>
#include <vector>

#include "mxnet/dag_engine.h"

using namespace std;
using namespace mxnet;

void Foo(RunContext rctx, int i) {
  cout << "say: " << i << endl;
}

int main() {
  DAGEngine* engine = DAGEngine::Get();
  Context exec_ctx;

  // Test #1
  cout << "============= Test #1 ==============" << endl;
  vector<DAGEngine::Variable> vars;
  for (int i = 0; i < 10; ++i) {
    vars.push_back(engine->NewVar());
  }
  for (int i = 0; i < 10; ++i) {
    engine->Push([i] (RunContext rctx) { Foo(rctx, i); },
        exec_ctx, vars, {});
  }

  usleep(1000000);

  // Test #2
  cout << "============= Test #2 ==============" << endl;
  for (int i = 0; i < 10; ++i) {
    engine->Push([i] (RunContext rctx) { Foo(rctx, i); },
        exec_ctx, {}, vars);
  }

  usleep(1000000);

  // Test #3
  return 0;
}
