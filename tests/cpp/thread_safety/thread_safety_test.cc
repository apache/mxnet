/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  \file thread_safety_test.cc
 *  \brief test thread safety at the dependency engine level and cached op level
 */

#if MXNET_USE_CPP_PACKAGE == 1
#include <stdio.h>
#include <gtest/gtest.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/ndarray.h>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "../src/engine/engine_impl.h"
#include "../src/imperative/imperative_utils.h"
#include "../include/test_util.h"
#include "mxnet-cpp/MxNetCpp.h"
/*
 * Prepares input data for the ops/models used in this file
 */
void prepare_input_data(const mxnet::cpp::Shape& shape, const mxnet::cpp::Context& ctx,
                        int num_threads,
                        std::vector<mxnet::cpp::NDArray>* data_arr,
                        bool random_uniform = false) {
  for (size_t i = 0; i < num_threads; ++i) {
    data_arr->emplace_back(shape, ctx, false, 0);
    int begin = i * 100;
    int end = begin + 100;
    if (random_uniform) {
      mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke((*data_arr)[i]);
    }
    mxnet::cpp::NDArray::WaitAll();
  }
}

void prepare_output_data(const mxnet::cpp::Shape& shape, const mxnet::cpp::Context& ctx,
                         int num_threads,
                         std::vector<mxnet::cpp::NDArray>* output_arr) {
    for (size_t i = 0; i < num_threads; ++i) {
        output_arr->emplace_back(shape, ctx, false, 0);
        mxnet::cpp::NDArray::WaitAll();
    }
}

/*
 * Prepare backend ndarrays from cpp frontend ndarrays
 */
void prepare_backend_data(const std::vector<mxnet::cpp::NDArray> &input_cpp_arrs,
                          int num_threads,
                          std::vector<mxnet::NDArray *> *output_backend_arrs) {
  output_backend_arrs->resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    (*output_backend_arrs)[i] = static_cast<NDArray *>(input_cpp_arrs[i].GetHandle());
  }
}

/*
 * Create and Invoke CachedOp for given data
 */
void get_expected_results(const mxnet::cpp::Symbol &sym,
                          const std::vector<std::string> &flag_keys,
                          const std::vector<std::string> &flag_vals,
                          int num_threads,
                          std::vector<std::vector<NDArrayHandle>> *arr_handles,
                          std::vector<mxnet::NDArray*> *result_expected,
                          CachedOpHandle* hdl) {
  // prepare flag_keys and flag_vals
  std::vector<const char *> flag_key_cstrs, flag_val_cstrs;
  flag_key_cstrs.reserve(flag_keys.size());
  for (size_t i = 0; i < flag_keys.size(); ++i) {
    flag_key_cstrs.emplace_back(flag_keys[i].c_str());
  }
  for (size_t i = 0; i < flag_vals.size(); ++i) {
    flag_val_cstrs.emplace_back(flag_vals[i].c_str());
  }

  // Create CachedOp
  int ret1 = MXCreateCachedOpEx(sym.GetHandle(), flag_keys.size(),
                                flag_key_cstrs.data(), flag_val_cstrs.data(),
                                hdl);
  if (ret1 < 0) {
    LOG(FATAL) << MXGetLastError();
  }

  std::vector<NDArrayHandle *> nd_ptrs(num_threads);

  // Invoke CachedOp same number of times as number of threads
  for (size_t i = 0; i < num_threads; ++i) {
    int num_output = 0;
    const int *stypes;
    int ret4 = MXInvokeCachedOpEx(*hdl, (*arr_handles)[i].size(), (*arr_handles)[i].data(),
                                  &num_output, &nd_ptrs[i], &stypes);
    if (ret4 < 0) {
      LOG(FATAL) << MXGetLastError();
    }
    mxnet::cpp::NDArray::WaitAll();
    (*result_expected)[i] = static_cast<NDArray*>(*nd_ptrs[i]);
  }
}

/*
 * Create and Invoke CachedOp for multiple threads, each thread with multiple
 * inferences
 */
inline void get_expected_results_multiple(
    const mxnet::cpp::Symbol &sym,
    const std::vector<std::string> &flag_keys, const std::vector<std::string> &flag_vals,
    std::vector<std::vector<std::vector<NDArrayHandle>>> *arr_handles,
    int num_threads,
    std::vector<std::vector<mxnet::NDArray *>> *result_expected,
    CachedOpHandle *hdl) {
  // prepare flag_keys and flag_vals
  std::vector<const char *> flag_key_cstrs, flag_val_cstrs;
  flag_key_cstrs.reserve(flag_keys.size());
  flag_val_cstrs.reserve(flag_vals.size());
  for (size_t i = 0; i < flag_keys.size(); ++i) {
    flag_key_cstrs.emplace_back(flag_keys[i].c_str());
  }
  for (size_t i = 0; i < flag_vals.size(); ++i) {
    flag_val_cstrs.emplace_back(flag_vals[i].c_str());
  }

  // Create CachedOp
  int ret1 =
      MXCreateCachedOpEX(sym.GetHandle(), flag_keys.size(),
                         flag_key_cstrs.data(), flag_val_cstrs.data(), hdl, false);
  if (ret1 < 0) {
    LOG(FATAL) << MXGetLastError();
  }
  std::vector<std::vector<NDArrayHandle *>> nd_ptrs((*arr_handles).size());

  // Invoke CachedOp same number of times as number of threads
  for (size_t i = 0; i < (*arr_handles).size(); ++i) {
    nd_ptrs[i].resize(num_threads);
    (*result_expected)[i].resize(num_threads);
    for (size_t j = 0; j < num_threads; ++j) {
      int num_output = 0;
      const int *stypes;
      int ret4 = MXInvokeCachedOpEx(*hdl, (*arr_handles)[i][j].size(),
                                    (*arr_handles)[i][j].data(), &num_output,
                                    &nd_ptrs[i][j], &stypes);
      if (ret4 < 0) {
        LOG(FATAL) << MXGetLastError();
      }
      mxnet::cpp::NDArray::WaitAll();
      (*result_expected)[i][j] = static_cast<NDArray *>(*nd_ptrs[i][j]);
    }
  }
}

void run_inference(const std::string& model,
                   int num_inf_per_thread = 1, bool random_sleep = false,
                   int num_threads = 1, bool static_alloc = false,
                   bool static_shape = false) {
    // Load model
    LOG(INFO) << "Running inference for " + model +
                 " num_threads: " + std::to_string(num_threads) +
                 " num_inf_per_thread: " + std::to_string(num_inf_per_thread) +
                 " random_sleep: " + std::to_string(random_sleep) +
                 " static_alloc: " + std::to_string(static_alloc) +
                 " static_shape: " + std::to_string(static_shape);
    auto out = mxnet::cpp::Symbol::Load(model + "-symbol.json");
    std::string static_alloc_str = static_alloc ? "true" : "false";
    std::string static_shape_str = static_shape ? "true" : "false";

    // Prepare context
#if MXNET_USE_CUDA == 1
    Context backend_ctx;
    mxnet::cpp::Context ctx = mxnet::cpp::Context::gpu(0);
    if (!mxnet::test::thread_safety_force_cpu) {
      backend_ctx = Context::GPU(0);
      ctx = mxnet::cpp::Context::gpu(0);
    } else {
      backend_ctx = Context::CPU();
      ctx = mxnet::cpp::Context::cpu();
    }
#else
    Context backend_ctx = Context::CPU(0);
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu(0);
#endif

    // Prepare input data and parameters
    std::vector<std::vector<mxnet::cpp::NDArray>> data_arr(num_inf_per_thread);
    std::vector<std::vector<mxnet::cpp::NDArray>> softmax_arr(num_inf_per_thread);
    std::vector<mxnet::cpp::NDArray> params;
    mxnet::cpp::Shape data_shape = mxnet::cpp::Shape(1, 3, 224, 224);
    mxnet::cpp::Shape softmax_shape = mxnet::cpp::Shape(1);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
     prepare_input_data(data_shape, ctx, num_threads, &(data_arr[i]), true);
     prepare_input_data(softmax_shape, ctx, num_threads, &(softmax_arr[i]));
    }
    std::map<std::string, mxnet::cpp::NDArray> parameters;
    mxnet::cpp::NDArray::Load(model + "-0000.params", 0, &parameters);

    for (std::string name : out.ListInputs()) {
        if (name == "arg:data") {
            continue;
        }
        if (parameters.find("arg:" + name) != parameters.end()) {
            params.push_back(parameters["arg:" + name].Copy(ctx));
        } else if (parameters.find("aux:" + name) != parameters.end()) {
            params.push_back(parameters["aux:" + name].Copy(ctx));
        }
    }

    // Prepare data_indices, param_indices and get_expected_results
    std::vector<std::string> flag_keys{"data_indices", "param_indices",
                                       "static_alloc", "static_shape"};
    std::string param_indices = "[";
    std::vector<std::vector<mxnet::NDArray*>> result_expected(num_inf_per_thread);
    int num_inputs = out.ListInputs().size();
    for (size_t i = 1; i < num_inputs; ++i) {
      param_indices += std::to_string(i);
      param_indices += std::string(", ");
    }
    param_indices += "]";
    std::vector<std::string> flag_vals{"[0]", param_indices, static_alloc_str, static_shape_str};
    std::vector<std::vector<std::vector<NDArrayHandle>>> arr_handles(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
      arr_handles[i].resize(num_threads);
      for (size_t j = 0; j < num_threads; ++j) {
        arr_handles[i][j].push_back(data_arr[i][j].GetHandle());
        for (size_t k = 1; k < num_inputs - 1; k++) {
          arr_handles[i][j].push_back(params[k - 1].GetHandle());
        }
        arr_handles[i][j].push_back(softmax_arr[i][j].GetHandle());
      }
    }
    CachedOpHandle hdl = CachedOpHandle();
    get_expected_results_multiple(out, flag_keys, flag_vals, &arr_handles,
                                  num_threads, &result_expected, &hdl);


    // Create thread safe cahced op
    CachedOpHandle hdl2 = CachedOpHandle();
    std::vector<const char *> flag_key_cstrs, flag_val_cstrs;
    flag_key_cstrs.reserve(flag_keys.size());
    for (size_t i = 0; i < flag_keys.size(); ++i) {
      flag_key_cstrs.emplace_back(flag_keys[i].c_str());
    }
    for (size_t i = 0; i < flag_vals.size(); ++i) {
      flag_val_cstrs.emplace_back(flag_vals[i].c_str());
    }

    int ret1 = MXCreateCachedOpEX(out.GetHandle(), flag_keys.size(),
                                  flag_key_cstrs.data(), flag_val_cstrs.data(),
                                  &hdl2, true);
    if (ret1 < 0) {
      LOG(FATAL) << MXGetLastError();
    }


    // Prepare data structures and lambda to run in different threads
    std::vector<NDArrayHandle *> cached_op_handles(num_threads * num_inf_per_thread);
    std::vector<std::vector<std::vector<mx_float>>> temp(num_inf_per_thread);
    std::vector<std::vector<mxnet::NDArray*>> output_mx_arr(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; i++) {
        output_mx_arr[i].resize(num_threads);
        temp[i].resize(num_threads);
        for (size_t j = 0; j < num_threads; ++j) {
            temp[i][j].resize(1000);
        }
    }

    std::vector<std::vector<std::vector<NDArrayHandle>>> arr_handles2(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
        arr_handles2[i].resize(num_threads);
        for (size_t j = 0; j < num_threads; ++j) {
            arr_handles2[i][j].reserve(num_inputs);
            arr_handles2[i][j].emplace_back(data_arr[i][j].GetHandle());
            for (size_t k = 1; k < num_inputs - 1; ++k) {
                arr_handles2[i][j].emplace_back(params[k - 1].GetHandle());
            }
            arr_handles2[i][j].emplace_back(softmax_arr[i][j].GetHandle());
        }
    }
    std::vector<mxnet::NDArray> data(num_inf_per_thread * num_threads);
    auto func = [&](int num) {
      unsigned next = num;
      for (size_t i = 0; i < num_inf_per_thread; ++i) {
        if (random_sleep) {
            int sleep_time = rand_r(&next) % 5;
            std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
        }
        int num_output = 0;
        const int *stypes;
        int ret = MXInvokeCachedOpEx(
            hdl2, arr_handles2[i][num].size(), arr_handles2[i][num].data(),
            &num_output, &(cached_op_handles[i * num_threads + num]), &stypes);
        if (ret < 0) {
            LOG(FATAL) << MXGetLastError();
        }
        output_mx_arr[i][num] = static_cast<mxnet::NDArray *>(
            *cached_op_handles[i * num_threads + num]);
      }
    };

    // Spawn multiple threads, join and wait for all threads to complete
    std::vector<std::thread> worker_threads(num_threads);
    int count = 0;
    for (auto &&i : worker_threads) {
      i = std::thread(func, count);
      count++;
    }

    for (auto &&i : worker_threads) {
      i.join();
    }

    mxnet::cpp::NDArray::WaitAll();
    for (size_t i = 0; i < num_inf_per_thread; i++) {
      mxnet::test::AssertEqual(output_mx_arr[i], result_expected[i], 1e-2, 1e-5);
    }
    mxnet::cpp::NDArray::WaitAll();
    int ret2 = MXFreeCachedOp(hdl);
    if (ret2 < 0) {
      LOG(FATAL) << MXGetLastError();
    }

    ret2 = MXFreeCachedOp(hdl2);
    if (ret2 < 0) {
      LOG(FATAL) << MXGetLastError();
    }
}

void run_inference_unsupported(const std::string& model,
                   int num_inf_per_thread = 1, bool random_sleep = false,
                   int num_threads = 1, bool static_alloc = false,
                   bool static_shape = false) {
    // Load model
    LOG(INFO) << "Running inference for " + model +
                 " num_threads: " + std::to_string(num_threads) +
                 " num_inf_per_thread: " + std::to_string(num_inf_per_thread) +
                 " random_sleep: " + std::to_string(random_sleep) +
                 " static_alloc: " + std::to_string(static_alloc) +
                 " static_shape: " + std::to_string(static_shape);
    auto out = mxnet::cpp::Symbol::Load(model + "-symbol.json");
    std::string static_alloc_str = static_alloc ? "true" : "false";
    std::string static_shape_str = static_shape ? "true" : "false";

    // Prepare context
#if MXNET_USE_CUDA == 1
    Context backend_ctx;
    mxnet::cpp::Context ctx = mxnet::cpp::Context::gpu(0);
    if (!mxnet::test::thread_safety_force_cpu) {
      backend_ctx = Context::GPU(0);
      ctx = mxnet::cpp::Context::gpu(0);
    } else {
      backend_ctx = Context::CPU();
      ctx = mxnet::cpp::Context::cpu();
    }
#else
    Context backend_ctx = Context::CPU(0);
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu(0);
#endif

    // Prepare input data and parameters
    std::vector<std::vector<mxnet::cpp::NDArray>> data_arr(num_inf_per_thread);
    std::vector<std::vector<mxnet::cpp::NDArray>> softmax_arr(num_inf_per_thread);
    std::vector<mxnet::cpp::NDArray> params;
    mxnet::cpp::Shape data_shape = mxnet::cpp::Shape(1, 3, 224, 224);
    mxnet::cpp::Shape softmax_shape = mxnet::cpp::Shape(1);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
     prepare_input_data(data_shape, ctx, num_threads, &(data_arr[i]), true);
     prepare_input_data(softmax_shape, ctx, num_threads, &(softmax_arr[i]));
    }
    std::map<std::string, mxnet::cpp::NDArray> parameters;
    mxnet::cpp::NDArray::Load(model + "-0000.params", 0, &parameters);

    for (std::string name : out.ListInputs()) {
        if (name == "arg:data") {
            continue;
        }
        if (parameters.find("arg:" + name) != parameters.end()) {
            params.push_back(parameters["arg:" + name].Copy(ctx));
        } else if (parameters.find("aux:" + name) != parameters.end()) {
            params.push_back(parameters["aux:" + name].Copy(ctx));
        }
    }

    // Prepare data_indices, param_indices and get_expected_results
    std::vector<std::string> flag_keys{"data_indices", "param_indices",
                                       "static_alloc", "static_shape"};
    std::string param_indices = "[";
    std::vector<std::vector<mxnet::NDArray*>> result_expected(num_inf_per_thread);
    int num_inputs = out.ListInputs().size();
    for (size_t i = 1; i < num_inputs; ++i) {
      param_indices += std::to_string(i);
      param_indices += std::string(", ");
    }
    param_indices += "]";
    std::vector<std::string> flag_vals{"[0]", param_indices, static_alloc_str, static_shape_str};
    std::vector<std::vector<std::vector<NDArrayHandle>>> arr_handles(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
      arr_handles[i].resize(num_threads);
      for (size_t j = 0; j < num_threads; ++j) {
        arr_handles[i][j].push_back(data_arr[i][j].GetHandle());
        for (size_t k = 1; k < num_inputs - 1; k++) {
          arr_handles[i][j].push_back(params[k - 1].GetHandle());
        }
        arr_handles[i][j].push_back(softmax_arr[i][j].GetHandle());
      }
    }
    CachedOpHandle hdl = CachedOpHandle();
    get_expected_results_multiple(out, flag_keys, flag_vals, &arr_handles,
                                  num_threads, &result_expected, &hdl);


    // Create thread safe cahced op
    CachedOpHandle hdl2 = CachedOpHandle();


    // Prepare data structures and lambda to run in different threads
    std::vector<NDArrayHandle *> cached_op_handles(num_threads * num_inf_per_thread);
    std::vector<std::vector<mxnet::NDArray*>> output_mx_arr(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; i++) {
        output_mx_arr[i].resize(num_threads);
    }

    std::vector<std::vector<std::vector<NDArrayHandle>>> arr_handles2(num_inf_per_thread);
    for (size_t i = 0; i < num_inf_per_thread; ++i) {
        arr_handles2[i].resize(num_threads);
        for (size_t j = 0; j < num_threads; ++j) {
            arr_handles2[i][j].reserve(num_inputs);
            arr_handles2[i][j].emplace_back(data_arr[i][j].GetHandle());
            for (size_t k = 1; k < num_inputs - 1; ++k) {
                arr_handles2[i][j].emplace_back(params[k - 1].GetHandle());
            }
            arr_handles2[i][j].emplace_back(softmax_arr[i][j].GetHandle());
        }
    }
    std::vector<mxnet::NDArray> data(num_inf_per_thread * num_threads);
    std::mutex mutex_;
    auto func = [&](int num) {
      std::vector<const char *> flag_key_cstrs, flag_val_cstrs;
      flag_key_cstrs.reserve(flag_keys.size());
      for (size_t i = 0; i < flag_keys.size(); ++i) {
        flag_key_cstrs.emplace_back(flag_keys[i].c_str());
      }
      for (size_t i = 0; i < flag_vals.size(); ++i) {
        flag_val_cstrs.emplace_back(flag_vals[i].c_str());
      }

      {
      // Uncomment these lines for a workaround around the same
      /*
      std::lock_guard<std::mutex> lock{mutex_};
      */

      if (hdl2 == nullptr) {
        int ret1 = MXCreateCachedOpEX(out.GetHandle(), flag_keys.size(),
                                      flag_key_cstrs.data(),
                                      flag_val_cstrs.data(), &hdl2, true);
        if (ret1 < 0) {
          LOG(FATAL) << MXGetLastError();
        }
      }
      }

      unsigned next = num;
      for (size_t i = 0; i < num_inf_per_thread; ++i) {
        if (random_sleep) {
          int sleep_time = rand_r(&next) % 5;
          std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
        }
        int num_output = 0;
        const int *stypes;
        int ret = MXInvokeCachedOpEx(
            hdl2, arr_handles2[i][num].size(), arr_handles2[i][num].data(),
            &num_output, &(cached_op_handles[i * num_threads + num]), &stypes);
        if (ret < 0) {
          LOG(FATAL) << MXGetLastError();
        }
        mxnet::cpp::NDArray::WaitAll();
        output_mx_arr[i][num] = static_cast<mxnet::NDArray *>(
            *cached_op_handles[i * num_threads + num]);
      }
    };

    // Spawn multiple threads, join and wait for all threads to complete
    std::vector<std::thread> worker_threads(num_threads);
    int count = 0;
    for (auto &&i : worker_threads) {
      i = std::thread(func, count);
      count++;
    }

    for (auto &&i : worker_threads) {
      i.join();
    }

    mxnet::cpp::NDArray::WaitAll();
    for (size_t i = 0; i < num_inf_per_thread; i++) {
      mxnet::test::AssertEqual(output_mx_arr[i], result_expected[i], 1e-2, 1e-5);
    }
    mxnet::cpp::NDArray::WaitAll();
    int ret2 = MXFreeCachedOp(hdl);
    if (ret2 < 0) {
      LOG(FATAL) << MXGetLastError();
    }

    ret2 = MXFreeCachedOp(hdl2);
    if (ret2 < 0) {
      LOG(FATAL) << MXGetLastError();
    }
}

/**
 * Verifying engine thread safety by pushing ops from multiple threads to the
 * dependency engine
 */
TEST(ThreadSafety, Engine) {
  int num_threads = 20;
#if MXNET_USE_CUDA == 1
  Context backend_ctx;
  mxnet::cpp::Context ctx = mxnet::cpp::Context::gpu(0);
  DispatchMode dispatch_mode;
  if (!mxnet::test::thread_safety_force_cpu) {
    backend_ctx = Context::GPU(0);
    ctx = mxnet::cpp::Context::gpu(0);
    dispatch_mode = DispatchMode::kFCompute;
  } else {
    backend_ctx = Context::CPU();
    ctx = mxnet::cpp::Context::cpu();
    dispatch_mode = DispatchMode::kFComputeEx;
  }
#else
  Context backend_ctx = Context::CPU(0);
  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu(0);
  DispatchMode dispatch_mode = DispatchMode::kFComputeEx;
#endif
  // Prepare convolution op and parse attrs
  const nnvm::Op *op = Op::Get("Convolution");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  attrs.name = "conv_node1";
  std::unordered_map<std::string, std::string> params = {
      {"kernel", "(2,2)"}, {"no_bias", "0"},    {"dilate", "(1,1)"},
      {"num_group", "1"},  {"layout", "NCHW"},  {"stride", "(1,1)"},
      {"pad", "(0,0)"},    {"num_filter", "10"}};
  attrs.dict = params;
  op->attr_parser(&attrs);

  // Prepare input data
  std::vector<mxnet::cpp::NDArray> data_arr, weight_arr, bias_arr, output_arr;
  mxnet::cpp::Shape data_shape(2, 4, 10, 10);
  mxnet::cpp::Shape weight_shape(10, 4, 2, 2);
  mxnet::cpp::Shape bias_shape(10);
  mxnet::cpp::Shape output_shape(2, 10, 9, 9);

  prepare_input_data(data_shape, ctx, num_threads, &data_arr, true);
  prepare_input_data(weight_shape, ctx, num_threads, &weight_arr, true);
  prepare_input_data(bias_shape, ctx, num_threads, &bias_arr, true);
  prepare_output_data(output_shape, ctx, num_threads, &output_arr);

  // Prepare symbol
  mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
  mxnet::cpp::Symbol weight = mxnet::cpp::Symbol::Variable("weight");
  mxnet::cpp::Symbol bias = mxnet::cpp::Symbol::Variable("bias");
  auto out = mxnet::cpp::Operator("Convolution")
      .SetParam("kernel", mxnet::cpp::Shape(2, 2))
      .SetParam("no_bias", false)
      .SetParam("dilate", mxnet::cpp::Shape(1, 1))
      .SetParam("num_group", 1)
      .SetParam("layout", "NCHW")
      .SetParam("stride", mxnet::cpp::Shape(1, 1))
      .SetParam("pad", mxnet::cpp::Shape(0, 0))
      .SetParam("num_filter", 10)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .SetInput("bias", bias)
      .CreateSymbol("fwd");

  // Prepare data_indices, param_indices and get_expected_results
  std::vector<std::string> flag_keys{"data_indices", "param_indices"};
  std::vector<std::string> flag_vals{"[0]", "[1,2]"};
  std::vector<mxnet::NDArray*> result_expected(num_threads);

  std::vector<std::vector<NDArrayHandle>> arr_handles(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
      arr_handles[i].push_back(data_arr[i].GetHandle());
      arr_handles[i].push_back(weight_arr[i].GetHandle());
      arr_handles[i].push_back(bias_arr[i].GetHandle());
  }
  CachedOpHandle hdl = CachedOpHandle();
  get_expected_results(out, flag_keys, flag_vals, num_threads,
                       &arr_handles, &result_expected, &hdl);

  // Prepare backend NDArray inputs
  std::vector<mxnet::NDArray*> data_mx_arr, weight_mx_arr, bias_mx_arr, output_mx_arr;
  prepare_backend_data(data_arr, num_threads, &data_mx_arr);
  prepare_backend_data(weight_arr, num_threads, &weight_mx_arr);
  prepare_backend_data(bias_arr, num_threads, &bias_mx_arr);
  prepare_backend_data(output_arr, num_threads, &output_mx_arr);

  // Prepare func which Invokes op
  auto func = [&](int num) {
    std::vector<mxnet::NDArray *> tmp_inputs, tmp_outputs;
    tmp_inputs.emplace_back(data_mx_arr[num]);
    tmp_inputs.emplace_back(weight_mx_arr[num]);
    tmp_inputs.emplace_back(bias_mx_arr[num]);
    tmp_outputs.emplace_back(output_mx_arr[num]);
    std::vector<OpReqType> reqs;
    reqs.push_back(kWriteTo);
    Imperative::Get()->InvokeOp(backend_ctx, attrs, tmp_inputs, tmp_outputs,
                                reqs, dispatch_mode, OpStatePtr());
  };

  // Spawn multiple threads
  std::vector<std::thread> worker_threads(num_threads);
  int count = 0;
  for (auto &&i : worker_threads) {
    i = std::thread(func, count);
    count++;
  }

  for (auto &&i : worker_threads) {
    i.join();
  }

  mxnet::cpp::NDArray::WaitAll();
  mxnet::test::AssertEqual(output_mx_arr, result_expected, 1e-2, 1e-5);
  mxnet::cpp::NDArray::WaitAll();
}

TEST(ThreadSafety, CachedOpFullModel) {
  std::vector<std::string> models_list = {
      "imagenet1k-resnet-18", "imagenet1k-resnet-152", "imagenet1k-resnet-50"};
  if (mxnet::test::thread_safety_force_cpu) {
    models_list.push_back("imagenet1k-resnet-152-subgraph");
  }
  for (const auto &model : models_list) {
    run_inference(model, 1, true, 20);
    run_inference(model, 2, true, 20);
    run_inference(model, 4, true, 5);
    run_inference(model, 4, true, 20);
    run_inference(model, 4, false, 20);
    run_inference(model, 8, true, 20);
    // static_alloc = true
    run_inference(model, 2, true, 20, true);
    run_inference(model, 4, true, 5, true);
    run_inference(model, 4, true, 20, true);
    run_inference(model, 8, true, 20, true);
    // static_alloc = true, static_shape = true
    run_inference(model, 4, true, 20, true, true);
    run_inference(model, 8, true, 20, true, true);
    // the below line may hang
    // run_inference_unsupported(model, 32, false, 20);
  }
}
#endif
