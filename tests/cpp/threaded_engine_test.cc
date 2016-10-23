#include <time.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>

#include <mxnet/engine.h>
#include "../src/engine/engine_impl.h"
#include <dmlc/timer.h>

/**
 * present the following workload
 *  n = reads.size()
 *  data[write] = (data[reads[0]] + ... data[reads[n]]) / n
 *  std::this_thread::sleep_for(std::chrono::microsecons(time));
 */
struct Workload {
  std::vector<int> reads;
  int write;
  int time;
};

/**
 * generate a list of workloads
 */
void GenerateWorkload(int num_workloads, int num_var,
                      int min_read, int max_read,
                      int min_time, int max_time,
                      std::vector<Workload>* workloads) {
  workloads->clear();
  workloads->resize(num_workloads);
  for (int i = 0; i < num_workloads; ++i) {
    auto& wl = workloads->at(i);
    wl.write = rand() % num_var;
    int r = rand();
    int num_read = min_read + (r % (max_read - min_read));
    for (int j = 0; j < num_read; ++j) {
      wl.reads.push_back(rand() % num_var);
    }
    wl.time = min_time + rand() % (max_time - min_time);
  }
}

/**
 * evaluate a single workload
 */
void EvaluateWorload(const Workload& wl, std::vector<double>* data) {
  double tmp = 0;
  for (int i : wl.reads) tmp += data->at(i);
  data->at(wl.write) = tmp / (wl.reads.size() + 1);
  if (wl.time > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(wl.time));
  }
}

/**
 * evaluate a list of workload, return the time used
 */
double EvaluateWorloads(const std::vector<Workload>& workloads,
                      mxnet::Engine* engine,
                      std::vector<double>* data) {
  using namespace mxnet;
  double t = dmlc::GetTime();
  std::vector<Engine::VarHandle> vars;
  if (engine) {
    for (size_t i = 0; i < data->size(); ++i) {
      vars.push_back(engine->NewVariable());
    }
  }

  for (const auto& wl : workloads) {
    if (wl.reads.size() == 0) continue;
    if (engine == NULL) {
      EvaluateWorload(wl, data);
    } else {
      auto func = [wl,data](RunContext ctx, Engine::CallbackOnComplete cb) {
        EvaluateWorload(wl, data); cb();
      };
      std::vector<Engine::VarHandle> reads;
      for (auto i : wl.reads) {
        if (i != wl.write) reads.push_back(vars[i]);
      }
      engine->PushAsync(func, Context::CPU(), reads, {vars[wl.write]});
    }
  }

  if (engine) {
    engine->WaitForAll();
  }
  return dmlc::GetTime() - t;
}

TEST(Engine, RandSumExpr) {
  std::vector<Workload> workloads;
  int num_repeat = 5;
  const int num_engine = 4;

  std::vector<double> t(num_engine, 0.0);
  std::vector<mxnet::Engine*> engine(num_engine);

  engine[0] = NULL;
  engine[1] = mxnet::engine::CreateNaiveEngine();
  engine[2] = mxnet::engine::CreateThreadedEnginePooled();
  engine[3] = mxnet::engine::CreateThreadedEnginePerDevice();

  for (int repeat = 0; repeat < num_repeat; ++repeat) {
    srand(time(NULL) + repeat);
    int num_var = 100;
    GenerateWorkload(10000, num_var, 2, 20, 1, 10, &workloads);
    std::vector<std::vector<double>> data(num_engine);
    for (int i = 0; i < num_engine; ++i) {
      data[i].resize(num_var, 1.0);
      t[i] += EvaluateWorloads(workloads, engine[i], &data[i]);
    }

    for (int i = 1; i < num_engine; ++i) {
      for (int j = 0; j < num_var; ++j) EXPECT_EQ(data[0][j], data[i][j]);
    }
    LOG(INFO) << "data: " << data[0][1] << " " << data[0][2] << "...";
  }


  LOG(INFO) << "baseline\t\t"  << t[0] << " sec";
  LOG(INFO) << "NaiveEngine\t\t"  << t[1] << " sec";
  LOG(INFO) << "ThreadedEnginePooled\t" << t[2] << " sec";
  LOG(INFO) << "ThreadedEnginePerDevice\t" << t[3] << " sec";
}

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
      {}, {var}, mxnet::FnProperty::kCopyFromGPU));
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

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
